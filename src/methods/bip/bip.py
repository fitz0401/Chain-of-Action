"""
BIP — Bidirectional Policy (stop_head redesign).

Architecture:
  * Single shared CoA ImageEncoder (spatial backbone).
  * Single CoA-REVERSE ActorModel: generates the full segment in reverse
    (keyframe → interaction start) from any observation in the segment.
  * stop_head: Linear(action_dim → 1) applied per decoded step; predicts
    whether that step is the interaction-start boundary (stop=1 / continue=0).

Training:
  * Full segment as reverse target (no manual interval truncation).
  * Offline _mark_stop_labels annotates each segment with stop_k via the
    inflection point of the cross-demo SE(3)-normalised variance curve.
  * Loss = L_action (masked L1) + L_latent (masked L1) + λ · L_stop (BCE).
  * stop_head is supervised on positions 0 … stop_k only (stop_mask).

Inference (closed-loop, re-plan every step):
  * Run CoA REVERSE AR to get full a_hat (b, L, action_dim).
  * Apply stop_head → sigmoid → find first position where p > stop_threshold.
  * Execute that action (the estimated interaction start / current target).
  * On next timestep: re-plan from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from transformers.optimization import get_scheduler

from src.methods.base import BaseMethod, BatchedActionSequence
from src.methods.utils import (
    extract_many_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
)

VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
VISUAL_OBS_STD  = [0.229, 0.224, 0.225]


class BIP(BaseMethod):
    def __init__(
        self,
        encoder_model,          # CoA spatial ImageEncoder
        coa_actor_model,        # CoA REVERSE ActorModel
        lr: float,
        lr_backbone: float,
        weight_decay: float,
        num_train_steps: int,
        adaptive_lr: bool,
        actor_grad_clip,
        action_dim: int,
        hidden_dim: int,
        execution_length: int,
        loss_type: str,
        latent_loss_type: str,
        lambda_stop: float,           # weight of BCE stop loss
        stop_threshold: float,        # sigmoid threshold for inference stopping
        plan_reach_threshold: float,  # EE dist (normalised) to switch plan→exec phase
        chunk_size: int,              # actions returned per act() call (action chunking)
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_dim            = action_dim
        self.loss_type             = loss_type
        self.latent_loss_type      = latent_loss_type
        self.actor_grad_clip       = actor_grad_clip
        self.lambda_stop           = lambda_stop
        self.stop_threshold        = stop_threshold
        self.plan_reach_threshold  = plan_reach_threshold
        self.chunk_size            = chunk_size
        self.hidden_dim            = hidden_dim

        self.device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # shared spatial backbone
        self.encoder = encoder_model().to(self.device)

        # CoA REVERSE planner (execute_threshold=0 → never stops internally;
        # stop_head controls generation length at inference)
        self.coa_actor = coa_actor_model(
            action_order="REVERSE",
            execute_threshold=0.0,
            execution_length=execution_length,
        ).to(self.device)

        # per-step stop classifier on decoder hidden state (richer than 8D action)
        self.stop_head = nn.Linear(hidden_dim, 1).to(self.device)

        self.img_normalizer = tvf.Normalize(mean=VISUAL_OBS_MEAN, std=VISUAL_OBS_STD)

        param_dicts = [
            {"params": [p for n, p in self.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters()
                        if "backbone" in n and p.requires_grad],
             "lr": lr_backbone},
        ]
        self.opt = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
        if adaptive_lr:
            self.lr_scheduler = get_scheduler(
                "cosine", optimizer=self.opt,
                num_warmup_steps=100, num_training_steps=num_train_steps,
            )
        self.prepare_accelerator()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _prepare_img(self, batch):
        raw = extract_many_from_batch(batch, "rgb")
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw, dim=1))
        return self.img_normalizer(img / 255.0)

    @staticmethod
    def _strip_mtp(a):
        return a[:, :, 0, :] if a.dim() == 4 else a

    def training_mode(self, training: bool = True):
        self.encoder.train(training)
        self.coa_actor.train(training)
        self.stop_head.train(training)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("BIP uses update() / act().")

    # ── training ─────────────────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        self.training_mode(True)
        img      = self._prepare_img(batch)
        proprio  = batch["low_dim_state"]
        task_emb = batch.get("task_emb", None)

        obs_feat = self.encoder(img)

        # CoA forward pass (teacher-forced; full segment as target)
        a_hat, x_hat, x_gt = self.coa_actor(
            obs_feat, proprio, task_emb,
            batch["action"], batch["is_pad"], training=True,
        )
        a_hat = self._strip_mtp(a_hat)                             # (b, L, action_dim)
        x_hat = x_hat[:, :, 0, :] if x_hat.dim() == 4 else x_hat  # (b, L, hidden_dim)

        is_pad = batch["is_pad"]          # (b, L) bool
        valid  = ~is_pad.unsqueeze(-1)    # (b, L, 1)

        # action regression (L1, masked to non-padded positions)
        loss_fn  = F.l1_loss if self.loss_type == "l1" else F.mse_loss
        coa_elem = loss_fn(a_hat, batch["action"], reduction="none") * valid
        coa_loss = coa_elem.sum() / valid.expand_as(coa_elem).sum().clamp(min=1)

        # latent reconstruction loss
        lat_fn   = F.l1_loss if self.latent_loss_type == "l1" else F.mse_loss
        lat_elem = lat_fn(x_hat, x_gt, reduction="none") * valid
        latent_loss = lat_elem.sum() / valid.expand_as(lat_elem).sum().clamp(min=1)

        # stop_head BCE on decoder hidden state; soft Gaussian labels over all valid steps
        stop_logits = self.stop_head(x_hat.detach()).squeeze(-1)    # (b, L)
        stop_mask   = batch["stop_mask"].bool()                      # (b, L), ~is_pad
        if stop_mask.any():
            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logits[stop_mask],
                batch["soft_stop_label"][stop_mask],
            )
        else:
            stop_loss = stop_logits.new_zeros(())

        total_loss = coa_loss + latent_loss + self.lambda_stop * stop_loss
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=-100.0)

        self.opt.zero_grad(set_to_none=True)
        if self.accelerator:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.actor_grad_clip)
        self.opt.step()
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

        return {
            "total_loss":  total_loss.detach(),
            "coa_loss":    coa_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "stop_loss":   stop_loss.detach(),
        }

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, batch) -> BatchedActionSequence:
        self.training_mode(False)
        img      = self._prepare_img(batch)
        proprio  = batch["low_dim_state"]
        task_emb = batch.get("task_emb", None)

        obs_feat = self.encoder(img)

        # CoA reverse AR: position 0 = keyframe, position stop_k = interaction start
        a_hat, x_hat, _ = self.coa_actor(obs_feat, proprio, task_emb, None, None, training=False)
        a_hat = self._strip_mtp(a_hat)                             # (b, L, action_dim)
        x_hat = x_hat[:, :, 0, :] if x_hat.dim() == 4 else x_hat  # (b, L, hidden_dim)

        # Find interaction-start via stop_head on hidden states;
        # argmax of raw logits as fallback when nothing crosses the sigmoid threshold
        stop_logits = self.stop_head(x_hat).squeeze(-1)   # (b, L)
        triggered   = stop_logits.sigmoid() > self.stop_threshold
        has_stop    = triggered.any(dim=1)
        stop_pos = torch.where(
            has_stop,
            triggered.float().argmax(dim=1),   # first threshold crossing
            stop_logits.argmax(dim=1),          # fallback: highest logit position
        )  # (b,)

        cur = proprio[:, -1] if proprio.dim() == 3 else proprio   # (b, dim)

        # Build action chunks from the reverse AR path treated as a continuous forward path.
        #
        # For each item:
        #   1. rev_seq = a_hat[i, :stop_pos+1]   — reverse order (keyframe … interaction_start)
        #   2. fwd_seq = flip(rev_seq)            — forward order (interaction_start … keyframe)
        #   3. start_idx = argmin EE dist to fwd_seq[:, :3] — where robot is on this path
        #   4. return fwd_seq[start_idx : start_idx + chunk_size], padded with keyframe
        #
        # This subsumes both "far" (start_idx≈0, motion-plan toward start) and "close"
        # (start_idx>0, advance through interaction) without a hard distance branch.
        chunks = []
        for i in range(int(stop_pos.shape[0])):
            sp      = int(stop_pos[i].item())
            rev_seq = a_hat[i, :sp + 1, :]                          # (sp+1, action_dim)
            fwd_seq = torch.flip(rev_seq, dims=[0])                  # (sp+1, action_dim)

            dists     = (fwd_seq[:, :3] - cur[i, :3]).norm(dim=-1)  # (sp+1,)
            start_idx = int(dists.argmin().item())
            tail      = fwd_seq[start_idx:]                          # remaining steps

            # Pad to chunk_size with the keyframe (last step of fwd_seq)
            if tail.shape[0] < self.chunk_size:
                pad  = fwd_seq[-1].unsqueeze(0).expand(self.chunk_size - tail.shape[0], -1)
                tail = torch.cat([tail, pad], dim=0)

            chunks.append(tail[:self.chunk_size])                    # (chunk_size, action_dim)

        return torch.stack(chunks, dim=0)   # (b, chunk_size, action_dim)

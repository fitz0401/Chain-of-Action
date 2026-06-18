"""
BIP — Bidirectional Policy.

A single shared visual backbone feeds two action heads:

  1. CoA-REVERSE head (planner) — identical to coa.py.
        Autoregressively generates the *full* remaining sub-trajectory in
        REVERSE order (keyframe a_T -> current a_idx). The m steps closest to
        the goal, a_hat[:, :m] = [a_T, a_{T-1}, ..., a_{T-m+1}], form the
        "milestone" that anchors the forward head.

  2. DP-FORWARD head (controller) — identical to dp.py.
        A conditional diffusion model that predicts the immediate forward chunk
        [a_{idx+1}, ..., a_{idx+dp_chunk}] actually executed on the robot.
        Conditioned on (pooled image feature, projected milestone).

Train / test milestone consistency
----------------------------------
At inference only the CoA *prediction* is available, so during training the
milestone fed to the DP head is a scheduled mix of ground truth and the
(detached) CoA prediction:

        p         = max(p_min, 1 - step / schedule_steps)
        milestone = p * GT + (1 - p) * CoA_pred.detach()

p starts at 1 (pure teacher forcing) and anneals towards p_min, so the DP head
gradually learns to trust the planner's own output — matching inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from diffusers import DDIMScheduler
from transformers.optimization import get_scheduler

from src.methods.base import BaseMethod, BatchedActionSequence
from src.methods.dp.diffusion import Actor as DPActor
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
        encoder_model,          # shared CoA-style spatial image encoder
        coa_actor_model,        # CoA transformer planner
        dp_actor_model,         # DP ConditionalUnet1D controller
        # optimiser / scheduler
        lr: float,
        lr_backbone: float,
        weight_decay: float,
        num_train_steps: int,
        adaptive_lr: bool,
        actor_grad_clip,
        # BIP params
        m: int,
        dp_action_sequence: int,
        action_dim: int,
        hidden_dim: int,
        milestone_embed_dim: int,
        milestone_schedule_steps: int,
        milestone_p_min: float,
        num_diffusion_iters: int,
        # CoA params injected at actor construction time
        execute_threshold: float,
        execution_length: int,
        loss_type: str,
        latent_loss_type: str,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.m                        = m
        self.dp_action_sequence       = dp_action_sequence
        self.action_dim               = action_dim
        self.loss_type                = loss_type
        self.latent_loss_type         = latent_loss_type
        self.actor_grad_clip          = actor_grad_clip
        self.milestone_schedule_steps = milestone_schedule_steps
        self.milestone_p_min          = milestone_p_min

        self.device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Shared visual backbone (spatial features for CoA, pooled for DP) ──
        self.encoder = encoder_model().to(self.device)

        # ── CoA planner head (REVERSE, full sub-trajectory) ──────────────────
        self.coa_actor = coa_actor_model(
            action_order="REVERSE",
            execute_threshold=execute_threshold,
            execution_length=execution_length,
        ).to(self.device)

        # ── Conditioning projections for the DP head ─────────────────────────
        # milestone: the m near-goal actions -> a single embedding
        self.milestone_proj = nn.Linear(m * action_dim, milestone_embed_dim).to(self.device)
        # pooled visual feature -> its own learnable space
        self.dp_visual_proj = nn.Linear(hidden_dim, hidden_dim).to(self.device)

        # ── DP controller head ────────────────────────────────────────────────
        dp_feature_dim = hidden_dim + milestone_embed_dim
        self.dp_noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        unet = dp_actor_model(
            input_shapes={"actions": (action_dim,), "features": (dp_feature_dim,)},
            output_shape=action_dim,
            sequence_length=dp_action_sequence,
        ).to(self.device)
        self.dp_actor = DPActor(
            action_sequence=dp_action_sequence,
            action_dim=action_dim,
            actor_model=unet,
            noise_scheduler=self.dp_noise_scheduler,
            num_diffusion_iters=num_diffusion_iters,
        ).to(self.device)

        self.img_normalizer = tvf.Normalize(mean=VISUAL_OBS_MEAN, std=VISUAL_OBS_STD)
        # checkpointed training-step counter that drives the milestone schedule
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

        # ── Unified AdamW (backbone group at lower lr; exclude DP EMA shadow) ──
        param_dicts = [
            {"params": [p for n, p in self.named_parameters()
                        if "backbone" not in n and "ema_actor" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters()
                        if "backbone" in n and "ema_actor" not in n and p.requires_grad],
             "lr": lr_backbone},
        ]
        self.opt = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
        if adaptive_lr:
            self.lr_scheduler = get_scheduler(
                "cosine", optimizer=self.opt,
                num_warmup_steps=100, num_training_steps=num_train_steps,
            )

        self.prepare_accelerator()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _prepare_img(self, batch) -> torch.Tensor:
        raw = extract_many_from_batch(batch, "rgb")
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw, dim=1))
        return self.img_normalizer(img / 255.0)

    def _pool_visual(self, img_feat: torch.Tensor) -> torch.Tensor:
        # (b, hidden_dim, h, w) -> (b, hidden_dim)
        return self.dp_visual_proj(img_feat.mean(dim=[-2, -1]))

    def _milestone_embed(self, seg: torch.Tensor) -> torch.Tensor:
        # (b, m, action_dim) -> (b, milestone_embed_dim)
        return self.milestone_proj(seg.reshape(seg.shape[0], -1))

    @staticmethod
    def _strip_mtp(a: torch.Tensor) -> torch.Tensor:
        return a[:, :, 0, :] if a.dim() == 4 else a

    def _milestone(self, a_hat: torch.Tensor, is_pad: torch.Tensor = None) -> torch.Tensor:
        """Build the m-step near-goal milestone in REVERSE order (pos 0 = keyframe a_T).

        Any step beyond the real sub-trajectory is filled with the *keyframe* action
        (the goal pose), not zeros, so the DP conditioning stays smooth and meaningful
        as the robot closes in on the keyframe.

          - training : a_hat is full length L (>= m); is_pad[:, :m] marks the tail.
          - inference: a_hat has the decoder's generated length k; keyframe-pad if k < m.
        """
        keyframe = a_hat[:, 0:1]                      # a_T (always valid: pos 0)
        if is_pad is not None:
            seg = a_hat[:, :self.m]
            pad = is_pad[:, :self.m].unsqueeze(-1)    # (b, m, 1), True = beyond traj
            return torch.where(pad, keyframe, seg)
        seq_len = a_hat.shape[1]
        if seq_len >= self.m:
            return a_hat[:, :self.m]
        pad = keyframe.expand(-1, self.m - seq_len, -1)
        return torch.cat([a_hat, pad], dim=1)

    def _milestone_p(self) -> float:
        if self.milestone_schedule_steps <= 0:
            return self.milestone_p_min
        frac = float(self._step.item()) / float(self.milestone_schedule_steps)
        return max(self.milestone_p_min, 1.0 - frac)

    def training_mode(self, training: bool = True):
        self.encoder.train(training)
        self.coa_actor.train(training)
        self.milestone_proj.train(training)
        self.dp_visual_proj.train(training)
        self.dp_actor.train(training)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, batch, training: bool = True):
        img      = self._prepare_img(batch)
        proprio  = batch["low_dim_state"]
        task_emb = batch.get("task_emb", None)

        obs_feat = self.encoder(img)          # (img_feat, pos)
        img_feat = obs_feat[0]

        if training:
            action_coa = batch["action"]      # (b, L, 8) REVERSE, pos 0 = keyframe
            is_pad_coa = batch["is_pad"]       # (b, L)
            a_hat_coa, x_hat, x_gt = self.coa_actor(
                obs_feat, proprio, task_emb, action_coa, is_pad_coa, training=True,
            )

            # milestone = scheduled mix of GT and detached CoA prediction;
            # both keyframe-padded on the (is_pad) tail beyond the real trajectory.
            gt_milestone   = self._milestone(action_coa, is_pad_coa)
            pred_milestone = self._milestone(self._strip_mtp(a_hat_coa).detach(), is_pad_coa)
            p = self._milestone_p()
            milestone = p * gt_milestone + (1.0 - p) * pred_milestone

            dp_features = torch.cat(
                [self._pool_visual(img_feat), self._milestone_embed(milestone)], dim=-1,
            )
            noise_pred, noise = self.dp_actor(dp_features, batch["action_dp"])
            return a_hat_coa, x_hat, x_gt, action_coa, is_pad_coa, noise_pred, noise

        # ── inference ──
        a_hat_coa, _, _ = self.coa_actor(
            obs_feat, proprio, task_emb, None, None, training=False,
        )
        milestone = self._milestone(self._strip_mtp(a_hat_coa))   # m near-goal, keyframe-padded
        dp_features = torch.cat(
            [self._pool_visual(img_feat), self._milestone_embed(milestone)], dim=-1,
        )
        a_hat_dp = self.dp_actor.infer(dp_features)           # (b, dp_chunk, 8)
        return a_hat_dp, a_hat_coa

    # ── Training update ─────────────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        self.training_mode(True)
        a_hat_coa, x_hat, x_gt, action_coa, is_pad_coa, noise_pred, noise = \
            self.forward(batch, training=True)

        valid = ~is_pad_coa.unsqueeze(-1)

        # CoA action loss (masked, same as coa.py)
        loss_fn = F.l1_loss if self.loss_type == "l1" else F.mse_loss
        coa_elem = loss_fn(a_hat_coa, action_coa, reduction="none") * valid
        coa_loss = coa_elem.sum() / valid.expand_as(coa_elem).sum().clamp(min=1)

        # CoA latent loss (masked, same as coa.py)
        lat_fn = F.l1_loss if self.latent_loss_type == "l1" else F.mse_loss
        lat_elem = lat_fn(x_hat, x_gt, reduction="none") * valid
        latent_loss = lat_elem.sum() / valid.expand_as(lat_elem).sum().clamp(min=1)

        # DP noise loss (same as dp.py)
        dp_loss = F.mse_loss(noise_pred, noise, reduction="none").mean(-1).mean(-1).mean()

        total_loss = coa_loss + latent_loss + dp_loss
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

        # Keep DP EMA synced so checkpoints are always inference-ready
        self.dp_actor.ema.step(self.dp_actor.actor.parameters())
        self.dp_actor.ema.copy_to(self.dp_actor.ema_actor.parameters())

        self._step += 1

        return {
            "total_loss":  total_loss.detach(),
            "coa_loss":    coa_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "dp_loss":     dp_loss.detach(),
            "milestone_p": torch.tensor(self._milestone_p(), device=self.device),
        }

    # ── Inference ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, batch) -> BatchedActionSequence:
        """Return the DP forward chunk (conditioned on the CoA milestone)."""
        self.training_mode(False)
        a_hat_dp, _ = self.forward(batch, training=False)
        return a_hat_dp

"""
BIP — Bidirectional Policy (segment + motion-planning hand-off).

Pipeline (data segmented by gripper-only keyframes; per segment an offline
variance analysis marks the near-goal low-variance interval [variance_start, keyframe]):

  * CoA-REVERSE head (planner): predicts the small-variance terminal segment in
    REVERSE (keyframe -> interval start). Trained on obs anywhere in the segment;
    target near-end = max(obs_idx, variance_start).
  * DP-FORWARD head (controller): plain Diffusion Policy, image-conditioned, predicts
    the executed forward chunk. Trained only INSIDE the interval (dp_valid mask).

Inference (stateless phase switch):
  CoA predicts the segment -> its start point. If the current end-effector is far from
  that start, output the start pose as a single action (RLBench motion-plans there);
  once close, switch to the DP head's forward chunk.
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
        lr: float,
        lr_backbone: float,
        weight_decay: float,
        num_train_steps: int,
        adaptive_lr: bool,
        actor_grad_clip,
        dp_action_sequence: int,
        action_dim: int,
        hidden_dim: int,
        num_diffusion_iters: int,
        execute_threshold: float,
        execution_length: int,
        loss_type: str,
        latent_loss_type: str,
        plan_reach_threshold: float,   # normalized EE distance to switch plan -> DP
        pad_eps: float,                # ||action|| below this is treated as padding
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dp_action_sequence   = dp_action_sequence
        self.action_dim           = action_dim
        self.loss_type            = loss_type
        self.latent_loss_type     = latent_loss_type
        self.actor_grad_clip      = actor_grad_clip
        self.plan_reach_threshold = plan_reach_threshold
        self.pad_eps              = pad_eps

        self.device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── shared visual backbone ──
        self.encoder = encoder_model().to(self.device)

        # ── CoA planner head (REVERSE) ──
        self.coa_actor = coa_actor_model(
            action_order="REVERSE",
            execute_threshold=execute_threshold,
            execution_length=execution_length,
        ).to(self.device)

        # ── DP controller head (plain DP: pooled image feature only) ──
        self.dp_visual_proj = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.dp_noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        unet = dp_actor_model(
            input_shapes={"actions": (action_dim,), "features": (hidden_dim,)},
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

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _prepare_img(self, batch):
        raw = extract_many_from_batch(batch, "rgb")
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw, dim=1))
        return self.img_normalizer(img / 255.0)

    def _pool_visual(self, img_feat):
        return self.dp_visual_proj(img_feat.mean(dim=[-2, -1]))   # (b, hidden_dim)

    @staticmethod
    def _strip_mtp(a):
        return a[:, :, 0, :] if a.dim() == 4 else a

    def _segment_start(self, a_hat):
        """Last non-padding REVERSE action = the interval start (furthest from keyframe).
        a_hat: (b, L, 8) REVERSE (pos 0 = keyframe, trailing positions ~0 = padding)."""
        n_valid = (a_hat.norm(dim=-1) > self.pad_eps).sum(1).clamp(min=1)   # (b,)
        last = (n_valid - 1).long()
        return a_hat[torch.arange(a_hat.shape[0], device=a_hat.device), last]   # (b, 8)

    def training_mode(self, training: bool = True):
        self.encoder.train(training)
        self.coa_actor.train(training)
        self.dp_visual_proj.train(training)
        self.dp_actor.train(training)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("BIP uses update() for training and act() for inference.")

    # ── training ──────────────────────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        self.training_mode(True)
        img      = self._prepare_img(batch)
        proprio  = batch["low_dim_state"]
        task_emb = batch.get("task_emb", None)

        obs_feat = self.encoder(img)
        img_feat = obs_feat[0]

        # CoA planner loss (masked L1 + latent)
        action_coa = batch["action"]
        is_pad_coa = batch["is_pad"]
        a_hat_coa, x_hat, x_gt = self.coa_actor(
            obs_feat, proprio, task_emb, action_coa, is_pad_coa, training=True)

        valid = ~is_pad_coa.unsqueeze(-1)
        loss_fn = F.l1_loss if self.loss_type == "l1" else F.mse_loss
        coa_elem = loss_fn(a_hat_coa, action_coa, reduction="none") * valid
        coa_loss = coa_elem.sum() / valid.expand_as(coa_elem).sum().clamp(min=1)
        lat_fn = F.l1_loss if self.latent_loss_type == "l1" else F.mse_loss
        lat_elem = lat_fn(x_hat, x_gt, reduction="none") * valid
        latent_loss = lat_elem.sum() / valid.expand_as(lat_elem).sum().clamp(min=1)

        # DP controller loss (noise MSE, masked to in-interval samples)
        noise_pred, noise = self.dp_actor(self._pool_visual(img_feat), batch["action_dp"])
        dp_se = F.mse_loss(noise_pred, noise, reduction="none").mean(-1).mean(-1)   # (b,)
        dp_valid = batch["dp_valid"]
        dp_loss = (dp_se * dp_valid).sum() / dp_valid.sum().clamp(min=1)

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

        self.dp_actor.ema.step(self.dp_actor.actor.parameters())
        self.dp_actor.ema.copy_to(self.dp_actor.ema_actor.parameters())

        return {
            "total_loss":  total_loss.detach(),
            "coa_loss":    coa_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "dp_loss":     dp_loss.detach(),
        }

    # ── inference ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, batch) -> BatchedActionSequence:
        self.training_mode(False)
        img      = self._prepare_img(batch)
        proprio  = batch["low_dim_state"]
        task_emb = batch.get("task_emb", None)

        obs_feat = self.encoder(img)
        img_feat = obs_feat[0]

        a_hat_coa, _, _ = self.coa_actor(obs_feat, proprio, task_emb, None, None, training=False)
        a_hat_coa = self._strip_mtp(a_hat_coa)
        start = self._segment_start(a_hat_coa)                    # (b, 8) interval start pose

        # current end-effector position vs. the segment start (normalized space)
        cur = proprio[:, -1] if proprio.dim() == 3 else proprio   # (b, dim)
        dist = (cur[:, :3] - start[:, :3]).norm(dim=-1)           # (b,)

        if bool((dist > self.plan_reach_threshold).any()):
            # PLAN phase: hand the start pose to the env's motion planner (one step)
            return start.unsqueeze(1)                             # (b, 1, 8)
        # CONTROL phase: DP forward chunk
        return self.dp_actor.infer(self._pool_visual(img_feat))  # (b, dp_chunk, 8)

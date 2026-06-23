"""
BIP — Bidirectional Policy.

A single shared visual backbone feeds two action heads:

  1. CoA-REVERSE head (planner) — identical to coa.py.
        Autoregressively generates the full remaining sub-trajectory in REVERSE
        order (keyframe a_T -> current a_idx).

  2. DP-FORWARD head (controller) — identical to dp.py.
        A conditional diffusion model that predicts the immediate forward chunk
        [a_{idx+1}, ..., a_{idx+dp_chunk}] actually executed on the robot,
        conditioned ONLY on the pooled image feature (plain DP — no milestone).

The two heads are trained independently (each with its own loss). They are
coupled ONLY at inference, through CoA-guided diffusion: while the DP head
denoises from pure noise, every DDIM step blends the predicted clean sample
toward the CoA guess for the same chunk (reconstruction guidance). See
`_coa_guess_chunk` and `Actor.infer_guided`.
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
        dp_action_sequence: int,
        action_dim: int,
        hidden_dim: int,
        num_diffusion_iters: int,
        # CoA-guided diffusion (inference only)
        use_coa_guidance: bool,
        guidance_strength: float,
        # CoA params injected at actor construction time
        execute_threshold: float,
        execution_length: int,
        loss_type: str,
        latent_loss_type: str,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dp_action_sequence = dp_action_sequence
        self.action_dim         = action_dim
        self.loss_type          = loss_type
        self.latent_loss_type   = latent_loss_type
        self.actor_grad_clip    = actor_grad_clip
        self.use_coa_guidance   = use_coa_guidance
        self.guidance_strength  = guidance_strength

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

        # ── DP controller head (plain DP: image-conditioned only) ─────────────
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

    @staticmethod
    def _strip_mtp(a: torch.Tensor) -> torch.Tensor:
        return a[:, :, 0, :] if a.dim() == 4 else a

    def _coa_guess_chunk(self, a_hat_coa: torch.Tensor) -> torch.Tensor:
        """CoA's guess for the DP forward chunk [a_{idx+1}, ..., a_{idx+dp_chunk}].

        a_hat_coa is REVERSE (pos 0 = keyframe a_T, last valid = current a_idx). Flip
        to forward [a_idx, a_{idx+1}, ...] and drop a_idx (the current action) to align
        with action_dp. Keyframe-pad the tail if the planner generated fewer steps.
        """
        fwd = torch.flip(a_hat_coa, dims=[1])             # [a_idx, a_{idx+1}, ..., a_T]
        guess = fwd[:, 1:1 + self.dp_action_sequence]     # [a_{idx+1}, ..., a_{idx+dp_chunk}]
        if guess.shape[1] < self.dp_action_sequence:
            keyframe = a_hat_coa[:, 0:1]                  # a_T
            pad = keyframe.expand(-1, self.dp_action_sequence - guess.shape[1], -1)
            guess = torch.cat([guess, pad], dim=1)
        return guess

    def training_mode(self, training: bool = True):
        self.encoder.train(training)
        self.coa_actor.train(training)
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

            # plain DP: image-conditioned only
            dp_features = self._pool_visual(img_feat)
            noise_pred, noise = self.dp_actor(dp_features, batch["action_dp"])
            return a_hat_coa, x_hat, x_gt, action_coa, is_pad_coa, noise_pred, noise

        # ── inference ──
        a_hat_coa, _, _ = self.coa_actor(
            obs_feat, proprio, task_emb, None, None, training=False,
        )
        a_hat_coa = self._strip_mtp(a_hat_coa)
        dp_features = self._pool_visual(img_feat)
        if self.use_coa_guidance:
            # warm-start the diffusion from the CoA guess for this same chunk (SDEdit)
            guess = self._coa_guess_chunk(a_hat_coa)          # (b, dp_chunk, 8)
            a_hat_dp = self.dp_actor.infer_guided(
                dp_features, guess, self.guidance_strength)
        else:
            a_hat_dp = self.dp_actor.infer(dp_features)        # (b, dp_chunk, 8)
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

        return {
            "total_loss":  total_loss.detach(),
            "coa_loss":    coa_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "dp_loss":     dp_loss.detach(),
        }

    # ── Inference ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, batch) -> BatchedActionSequence:
        """Return the DP forward chunk (CoA-guided diffusion)."""
        self.training_mode(False)
        a_hat_dp, _ = self.forward(batch, training=False)
        return a_hat_dp

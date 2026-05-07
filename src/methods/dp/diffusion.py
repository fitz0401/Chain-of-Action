"""
Diffusion Policy for Chain-of-Action.
Based on https://diffusion-policy.cs.columbia.edu using DDIM scheduler.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from typing import Optional

from diffusers import DDIMScheduler
from diffusers.training_utils import EMAModel
from transformers.optimization import get_scheduler

from src.methods.base import BaseMethod, BatchedActionSequence
from src.methods.backbone import build_backbone
from src.methods.utils import (
    extract_many_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
)
from src.models.diffusion_models import ConditionalUnet1D, replace_bn_with_gn


VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
VISUAL_OBS_STD = [0.229, 0.224, 0.225]


class ImageEncoder(nn.Module):
    """Image encoder that returns a flat feature vector for diffusion conditioning.
    Uses the same ResNet backbone as ACT but pools spatially.
    """

    def __init__(
        self,
        input_shape,
        hidden_dim=512,
        position_embedding="sine",
        lr_backbone=1e-5,
        masks=False,
        backbone="resnet18",
        dilation=False,
        use_frozen_bn=True,
    ):
        super().__init__()
        assert len(input_shape) == 4, f"Expected (V, C, H, W), got {input_shape}"
        self._input_shape = tuple(input_shape)
        self.hidden_dim = hidden_dim
        self.num_views = input_shape[0]

        self.backbone = build_backbone(
            hidden_dim=hidden_dim,
            position_embedding=position_embedding,
            lr_backbone=lr_backbone,
            masks=masks,
            backbone=backbone,
            dilation=dilation,
            use_frozen_bn=use_frozen_bn,
        )
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        # Replace BN with GN for stable diffusion training
        replace_bn_with_gn(self)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, v, c, h, w) - already normalized
        Returns:
            features: (b, hidden_dim) - global average pooled
        """
        assert self._input_shape == x.shape[1:], f"Expected {self._input_shape}, got {x.shape[1:]}"

        all_cam_features = []
        for cam_id in range(self._input_shape[0]):
            cur_x = x[:, cam_id].reshape(-1, 3, *self._input_shape[2:])
            feat, _ = self.backbone(cur_x)
            feat = self.input_proj(feat[0])  # (b, hidden_dim, h', w')
            all_cam_features.append(feat)

        # Average pool each camera feature then concatenate across views
        pooled = [f.mean(dim=[-2, -1]) for f in all_cam_features]  # list of (b, hidden_dim)
        # Mean across views to get (b, hidden_dim)
        features = torch.stack(pooled, dim=1).mean(dim=1)
        return features


class Actor(nn.Module):
    """Wraps ConditionalUnet1D with DDIM noise scheduler and EMA."""

    def __init__(
        self,
        action_sequence: int,
        action_dim: int,
        actor_model: ConditionalUnet1D,
        noise_scheduler: DDIMScheduler,
        num_diffusion_iters: int,
    ):
        super().__init__()
        self.action_sequence = action_sequence
        self.action_dim = action_dim
        self.actor = actor_model
        self.noise_scheduler = noise_scheduler
        self.num_diffusion_iters = num_diffusion_iters

        self.ema = EMAModel(parameters=self.actor.parameters(), power=0.75)
        self.ema_actor = copy.deepcopy(self.actor)

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        """Training forward: add noise and predict it.
        Args:
            features: (b, feature_dim)
            actions: (b, seq, action_dim)
        Returns:
            noise_pred, noise
        """
        b = features.shape[0]
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b,), device=features.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        noise_pred = self.actor({
            "actions": noisy_actions,
            "features": features,
            "timestep": timesteps,
        })
        return noise_pred, noise

    @torch.no_grad()
    def infer(self, features: torch.Tensor) -> torch.Tensor:
        """Inference: iterative DDIM denoising.
        Args:
            features: (b, feature_dim)
        Returns:
            actions: (b, seq, action_dim)
        """
        b = features.shape[0]
        # ema_actor is kept synced after each update() step and correctly
        # restored from state_dict on checkpoint load — do not overwrite here.

        noisy_action = torch.randn(
            (b, self.action_sequence, self.action_dim), device=features.device
        )
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            noise_pred = self.ema_actor({
                "actions": noisy_action,
                "features": features,
                "timestep": k,
            })
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action
            ).prev_sample

        return noisy_action


class DiffusionPolicy(BaseMethod):
    """Diffusion Policy agent for Chain-of-Action.
    Interface matches CoA: update(batch) and act(batch).
    """

    def __init__(
        self,
        encoder_model,
        actor_model,
        lr,
        action_sequence,
        action_dim,
        num_diffusion_iters=50,
        num_train_steps=200000,
        adaptive_lr=True,
        actor_grad_clip=None,
        weight_decay=1e-6,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.action_sequence = action_sequence
        self.action_dim = action_dim
        self.num_diffusion_iters = num_diffusion_iters
        self.num_train_steps = num_train_steps
        self.adaptive_lr = adaptive_lr
        self.actor_grad_clip = actor_grad_clip
        self.weight_decay = weight_decay

        self.device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build image encoder
        self.encoder_model = encoder_model().to(self.device)
        feature_dim = self.encoder_model.output_dim

        # Build DDIM noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Build actor (ConditionalUnet1D)
        unet = actor_model(
            input_shapes={
                "actions": (action_dim,),
                "features": (feature_dim,),
            },
            output_shape=action_dim,
            sequence_length=action_sequence,
        ).to(self.device)

        self.actor = Actor(
            action_sequence=action_sequence,
            action_dim=action_dim,
            actor_model=unet,
            noise_scheduler=self.noise_scheduler,
            num_diffusion_iters=num_diffusion_iters,
        ).to(self.device)

        self.img_normalizer = tvf.Normalize(mean=VISUAL_OBS_MEAN, std=VISUAL_OBS_STD)

        # Optimizer - use AdamW for UNet, Adam for encoder
        self.actor_opt = self.actor.actor.preferred_optimiser(lr=self.lr)
        self.encoder_opt = torch.optim.Adam(self.encoder_model.parameters(), lr=self.lr)

        if self.adaptive_lr:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.actor_opt,
                num_warmup_steps=100,
                num_training_steps=self.num_train_steps,
            )

        self.prepare_accelerator()

    def _encode_images(self, batch_input: dict) -> torch.Tensor:
        """Extract and encode multi-camera images to flat features."""
        raw_img = extract_many_from_batch(batch_input, "rgb")
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw_img, dim=1))
        img = self.img_normalizer(img / 255.0)
        return self.encoder_model(img)  # (b, feature_dim)

    def forward(self, batch_input: dict, training: bool = True):
        features = self._encode_images(batch_input)

        if training:
            actions = batch_input["action"]  # (b, seq, action_dim)
            noise_pred, noise = self.actor(features, actions)
            return noise_pred, noise, features
        else:
            return self.actor.infer(features), features

    def training_mode(self, training: bool = True):
        self.encoder_model.train(training)
        self.actor.train(training)

    @torch.no_grad()
    def act(self, batch_input: dict) -> BatchedActionSequence:
        self.training_mode(False)
        actions, _ = self.forward(batch_input, training=False)
        return actions

    def update(self, batch_input: dict) -> dict:
        self.training_mode(True)
        noise_pred, noise, _ = self.forward(batch_input, training=True)

        mse_loss = F.mse_loss(noise_pred, noise, reduction="none").mean(-1).mean(-1)
        actor_loss = mse_loss.mean()
        actor_loss = torch.nan_to_num(actor_loss, nan=0.0, posinf=100.0, neginf=-100.0)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)

        if self.accelerator:
            self.accelerator.backward(actor_loss)
        else:
            actor_loss.backward()

        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
            nn.utils.clip_grad_norm_(self.encoder_model.parameters(), self.actor_grad_clip)

        self.actor_opt.step()
        self.encoder_opt.step()

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

        # Update EMA weights and sync ema_actor so it's always checkpoint-ready
        self.actor.ema.step(self.actor.actor.parameters())
        self.actor.ema.copy_to(self.actor.ema_actor.parameters())

        return {"total_loss": actor_loss.detach(), "actor_loss": actor_loss.detach()}

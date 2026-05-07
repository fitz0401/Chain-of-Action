"""
ACT (Action Chunking with Transformers) policy for Chain-of-Action.
Based on https://github.com/tonyzhaozh/act with CVAE latent token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as tvf
from typing import Optional, Tuple, List
from torch.autograd import Variable

from transformers.optimization import get_scheduler

from src.methods.base import BaseMethod, BatchedActionSequence
from src.methods.backbone import build_backbone
from src.methods.act.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
from src.methods.utils import (
    extract_many_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
)


VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
VISUAL_OBS_STD = [0.229, 0.224, 0.225]


def kl_divergence(mu, logvar):
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dim_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dim_wise_kld, mean_kld


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ImageEncoder(nn.Module):
    """Image encoder using ResNet backbone with sinusoidal position embeddings.
    Returns (img_feat, pos_embed, task_emb) to be compatible with ACT and CoA.
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
        use_lang_cond=False,
        use_frozen_bn=False,
    ):
        super().__init__()
        assert len(input_shape) == 4, f"Expected (V, C, H, W), got {input_shape}"
        self._input_shape = tuple(input_shape)
        self.use_lang_cond = use_lang_cond

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

    def forward(self, x: torch.Tensor, task_emb: Optional[torch.Tensor] = None):
        """
        Args:
            x: (b, v, c, h, w) - already normalized
            task_emb: optional (b, lang_dim)
        Returns:
            img_feat: (b, hidden_dim, 3, 3*v)
            pos: (b, hidden_dim, 3, 3*v)
            task_emb: passthrough
        """
        assert self._input_shape == x.shape[1:], f"Expected {self._input_shape}, got {x.shape[1:]}"

        all_cam_features = []
        all_cam_pos = []
        shape = x.shape

        for cam_id in range(self._input_shape[0]):
            cur_x = x[:, cam_id].reshape(-1, 3, *self._input_shape[2:])
            feat, pos = self.backbone(cur_x)
            feat = self.input_proj(feat[0])
            pos = pos[0]
            all_cam_features.append(feat)
            all_cam_pos.append(pos)

        img_feat = torch.cat(all_cam_features, dim=3)
        img_feat = img_feat.reshape(shape[0], -1, *img_feat.shape[2:])
        pos = torch.cat(all_cam_pos, dim=3)

        return img_feat, pos, task_emb


class ActorModel(nn.Module):
    """ACT actor model with CVAE latent token.
    Combines a VAE-style encoder (for training) with a transformer decoder.
    """

    def __init__(
        self,
        hidden_dim=512,
        dropout=0.1,
        nheads=8,
        dim_feedforward=3200,
        enc_layers=4,
        dec_layers=7,
        pre_norm=False,
        state_dim=8,
        action_dim=8,
        num_queries=100,
        kl_weight=10,
        use_lang_cond=False,
        latent_token=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = 32
        self.kl_weight = kl_weight
        self.latent_token = latent_token
        self.use_lang_cond = use_lang_cond

        # CVAE encoder: encodes (actions, qpos) -> latent z during training
        encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, "relu", pre_norm)
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        self.encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)

        # Main transformer decoder
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=pre_norm,
            return_intermediate_dec=True,
        )

        # Output heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # CVAE encoder components
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )

        # Position embeddings for [latent, proprio] (and optionally task_emb)
        num_cond_tokens = 3 if use_lang_cond else 2
        self.additional_pos_embed = nn.Embedding(num_cond_tokens, hidden_dim)
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

    def _style_variable_encoder(self, bs, actions, qpos, is_pad):
        """Encode (actions, qpos) -> latent z via CVAE encoder."""
        action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)  # (bs, 1, hidden)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # (bs, 1, hidden)

        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # (bs, seq+2, hidden)
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+2, bs, hidden)

        cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
        is_pad_full = torch.cat([cls_joint_is_pad, is_pad], dim=1)

        pos_embed = self.pos_table[:, : encoder_input.shape[0]].clone().detach().permute(1, 0, 2).repeat(1, bs, 1)
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)
        return encoder_output[0]  # CLS token only

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        qpos: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (img_feat, pos) from image encoder
            qpos: (b, state_dim)
            actions: (b, seq, action_dim) - training only
            is_pad: (b, seq) bool - training only
            task_emb: optional (b, lang_dim)
        Returns:
            a_hat: (b, num_queries, action_dim)
            is_pad_hat: (b, num_queries, 1)
            [mu, logvar]: VAE parameters (None during inference)
        """
        bs = x[0].shape[0]
        proprio_input = self.input_proj_robot_state(qpos)  # (b, hidden)

        if self.training and actions is not None and self.latent_token:
            actions_trunc = actions[:, : self.num_queries]
            is_pad_trunc = is_pad[:, : self.num_queries]
            encoder_out = self._style_variable_encoder(bs, actions_trunc, qpos, is_pad_trunc)
            latent_info = self.latent_proj(encoder_out)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)

        hs = self.transformer(
            x[0],
            None,
            self.query_embed.weight,
            x[1],
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            task_emb=task_emb,
        )[-1]  # Last decoder layer output: (b, num_queries, hidden)

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

    def compute_loss(self, a_hat, is_pad_hat, mu, logvar, actions, is_pad):
        """Compute L1 action loss + KL divergence."""
        seq_len = min(actions.shape[1], a_hat.shape[1])
        actions_trunc = actions[:, :seq_len]
        is_pad_trunc = is_pad[:, :seq_len]
        a_hat_trunc = a_hat[:, :seq_len]

        all_l1 = F.l1_loss(actions_trunc, a_hat_trunc, reduction="none")
        l1 = (all_l1 * ~is_pad_trunc.unsqueeze(-1)).mean()

        if mu is not None and logvar is not None:
            total_kld, _, _ = kl_divergence(mu, logvar)
            kl_loss = total_kld[0]
        else:
            kl_loss = torch.tensor(0.0, device=a_hat.device)

        loss = l1 + kl_loss * self.kl_weight
        return loss, {"loss": loss, "l1": l1, "kl": kl_loss}


class ACT(BaseMethod):
    """ACT behavioral cloning agent for Chain-of-Action.
    Interface matches CoA: update(batch) and act(batch).
    """

    def __init__(
        self,
        encoder_model,
        actor_model,
        lr,
        lr_backbone,
        weight_decay,
        num_train_steps,
        adaptive_lr,
        use_lang_cond,
        loss_type="l1",
        action_sequence=100,
        actor_grad_clip=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.num_train_steps = num_train_steps
        self.adaptive_lr = adaptive_lr
        self.use_lang_cond = use_lang_cond
        self.loss_type = loss_type
        self.actor_grad_clip = actor_grad_clip

        self.device = (
            self.accelerator.device
            if self.accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.encoder_model = encoder_model().to(self.device)
        self.actor_model = actor_model().to(self.device)

        self.img_normalizer = tvf.Normalize(mean=VISUAL_OBS_MEAN, std=VISUAL_OBS_STD)

        param_dicts = [
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        self.opt = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        if self.adaptive_lr:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.opt,
                num_warmup_steps=100,
                num_training_steps=self.num_train_steps,
            )

        self.prepare_accelerator()

    def _preprocess_images(self, batch_input: dict) -> torch.Tensor:
        """Extract and normalize multi-camera images from batch."""
        raw_img = extract_many_from_batch(batch_input, "rgb")
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw_img, dim=1))
        img = self.img_normalizer(img / 255.0)
        return img

    def forward(self, batch_input: dict, training: bool = True):
        img = self._preprocess_images(batch_input)
        proprio = batch_input["low_dim_state"]
        if proprio.dim() == 3:  # (b, t, d) -> (b, d)
            proprio = proprio[:, -1]

        task_emb = batch_input.get("task_emb", None)

        img_feat, pos, task_emb = self.encoder_model(img, task_emb=task_emb)
        x = (img_feat, pos)

        if training:
            actions = batch_input["action"]  # (b, seq, action_dim)
            is_pad = batch_input["is_pad"]   # (b, seq) bool
        else:
            actions = is_pad = None

        a_hat, is_pad_hat, (mu, logvar) = self.actor_model(x, proprio, actions, is_pad, task_emb)
        return a_hat, is_pad_hat, (mu, logvar), actions, is_pad

    def training_mode(self, training: bool = True):
        self.encoder_model.train(training)
        self.actor_model.train(training)

    @torch.no_grad()
    def act(self, batch_input: dict) -> BatchedActionSequence:
        self.training_mode(False)
        a_hat, _, _, _, _ = self.forward(batch_input, training=False)
        return a_hat

    def update(self, batch_input: dict) -> dict:
        self.training_mode(True)
        a_hat, is_pad_hat, (mu, logvar), actions, is_pad = self.forward(batch_input, training=True)

        loss, loss_dict = self.actor_model.compute_loss(a_hat, is_pad_hat, mu, logvar, actions, is_pad)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=-100.0)

        self.opt.zero_grad(set_to_none=True)
        if self.accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.actor_grad_clip)

        self.opt.step()
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

        return {
            "total_loss": loss.detach(),
            **{k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()},
        }

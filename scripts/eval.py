# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
Evaluation script for Chain-of-Action models.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workspace import Workspace
import hydra
import torch


def load_checkpoint(snapshot_path):
    # Checkpoints in this project contain config objects, so load the full pickle payload.
    try:
        return torch.load(snapshot_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only.
        return torch.load(snapshot_path, map_location="cpu")

@hydra.main(config_path="../src/cfgs/", config_name="launch", version_base=None)
def main(cfg):
    # load the checkpoint
    checkpoint = load_checkpoint(cfg.snapshot)

    # merge the cfg from checkpoint and the cfg from command line
    cfg_ckpt = checkpoint["config"]
    cfg.env.env_name = cfg_ckpt.env.env_name
    cfg.method = cfg_ckpt.method
    cfg.action_sequence = cfg_ckpt.action_sequence
    cfg.method_name = cfg_ckpt.method_name
    cfg.wandb.use = False

    workspace = Workspace(cfg, train=False)
    workspace.eval()

if __name__ == "__main__":
    main() 

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

"""
RLBench Dataset for Chain-of-Action

This dataset loads and preprocesses demonstration data from RLBench,
supporting different action sequence orderings.

Key features:
- Loads demonstrations from RLBench format
- Configurable action sequence ordering (reverse/forward/hybrid)
- Handles padding and masking of variable length sequences
- Provides batched data loading for efficient training

"""

from __future__ import annotations
from torch.utils.data import Dataset
from enum import Enum  # Use Enum to define enumeration types
import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.envs.rlbench.rlbench_env import ActionModeType


# Define enumeration type for action order
class ActionOrder(Enum):
    REVERSE = 1  # reverse, default setting for CoA
    FORWARD = 2  # forward, default setting for ACT and DP, for ablaiton study of CoA
    HYBRID = 3   # hybrid, for ablaiton study of CoA

class RLBenchDataset(Dataset):
    """
    RLBenchDataset is used to load and preprocess demonstration data (demos),
    supporting different action sequence ordering (forward/reverse).
    """
    def __init__(self, cfg, seq_len):
        """
        Initialize dataset configuration.

        Args:
            cfg: An object or dict containing configuration information, must include the following fields:
                - action_sequence: maximum length of action sequence
                - k.act_padding: padding method, 'repeat' or 'zero'
                - k.sample_threshold: sampling threshold
                - k.REVERSE: string, specifies action order (e.g. "REVERSE")
                - k.full_traj_training: whether to use full trajectory training
                - keyframe_only: whether to sample only nbp part
                - env.action_mode: action mode in the environment (e.g. ActionModeType.ABS_JOINT_POSITION, etc.)
        """
        self._nstep = 1
        self._demos = None  
        self._frame_stacks = 1
        self._action_seq_len_max = seq_len
        self.action_padding = cfg.method.action_padding
        self.traj_sample_margin = cfg.method.traj_sample_margin if cfg.method_name == "coa" else None
        
        # Determine action order by enumeration and YAML config string
        self.action_order = ActionOrder[cfg.method.action_order]  # 例如 cfg.method.REVERSE 为 "REVERSE"
        self.cfg = cfg

        # Decide whether to perform last action correction for different action modes
        if cfg.env.action_mode == ActionModeType.ABS_JOINT_POSITION and \
           cfg.env.action_mode == ActionModeType.HYBRID:  
            self.last_action_correction = True
        else:
            self.last_action_correction = False

    def __len__(self):
        # Return the number of demos
        return len(self._demos)

    def get_observation(self, episode: dict, idx: int) -> dict:
        """
        Sample observation data according to the current idx.
        
        Args:
            episode: A single demonstration (contains multiple fields, such as action, desc, etc.)
            idx: current sampling index
        
        Returns:
            A dict containing the sampled observation
        """
        ep_len = len(episode[ActionModeType[self.cfg.env.action_mode].value]) - 1
        obs_start_idx = idx - self._frame_stacks + 1
        obs_idxs = [np.clip(i, 0, ep_len) for i in range(obs_start_idx, idx + 1)]
        sample = {name: episode[name][obs_idxs] for name in episode.keys()}
        return sample


    def get_action(self, actions: np.ndarray, ep_len: int, idx: int) -> tuple:
        """
        Get the action sequence for ACT/DP mode for forward prediction.
        
        Args:
            actions: all actions in the current episode
            ep_len: length of the current episode
            idx: current sampling start index
        
        Returns:
            action_seq: processed action sequence (l, 8)
            is_pad: padding flag array (l,)
        """
        # Calculate the start and end index of the action sequence
        action_start_idx = idx + 1
        action_end_idx = min(idx + self._action_seq_len_max + 1, ep_len)
        
        # Extract action sequence
        action_idxs = list(range(action_start_idx, action_end_idx))
        action_seq = actions[action_idxs]
        
        # Calculate the number of actions to pad
        num_actions_extracted = len(action_seq)
        num_actions_to_pad = self._action_seq_len_max - num_actions_extracted
        
        # Build padding flag
        is_pad = np.array([False] * num_actions_extracted + [True] * num_actions_to_pad)
        
        # Pad insufficient part
        if num_actions_to_pad > 0:
            if self.action_padding == 'zero':
                # Use zero padding
                padding_shape = (num_actions_to_pad,) + action_seq.shape[1:]
                action_padding = np.zeros(padding_shape, dtype=action_seq.dtype)
            elif self.action_padding == 'repeat':
                # Repeat the last action
                action_padding = np.tile(action_seq[-1:], (num_actions_to_pad, 1))
            else:
                raise ValueError(f"Unknown action_padding type: {self.action_padding}")
            
            # Concatenate action sequence and padding
            action_seq = np.concatenate([action_seq, action_padding], axis=0)
        
        # Ensure the array is contiguous
        action_seq = np.ascontiguousarray(action_seq)
        
        return action_seq, is_pad


    def get_action_coa(self, actions: np.ndarray, ep_len: int, idx: int) -> tuple:
        """
        Extract and process action sequence according to config, supporting forward or reverse order.

        Args:
            actions: all actions in the current episode
            ep_len: length of the current episode (note: len(actions)-1)
            idx: current sampling start index
        
        Returns:
            action_seq: processed action sequence (order determined by config)
            is_pad: padding flag array
            is_stop: stop flag array
        """
        # Determine action start index according to whether full trajectory training is used
        action_start_idx = 0 if self.cfg.method.full_traj_training else idx
        action_end_idx = ep_len
        action_idxs = list(range(action_start_idx, action_end_idx + 1))
        action_seq = actions[action_idxs]
        
        # Build padding and stop flags
        is_pad = np.array([False] * len(action_idxs) + [True] * (self._action_seq_len_max - len(action_idxs)))

        
        # Pad insufficient part
        assert self.action_padding == 'zero', f"Unknown act_padding type: {self.action_padding}"
        num_action_to_pad = self._action_seq_len_max - len(action_seq)
        if self.action_padding == 'repeat':
            action_padding = action_seq[0:1, ...].repeat(num_action_to_pad, axis=0)
        elif self.action_padding == 'zero':
            action_padding = np.zeros_like(action_seq[0:1, ...].repeat(num_action_to_pad, axis=0))
        else:
            raise ValueError(f"Unknown act_padding type: {self.action_padding}")
        
        # Determine the order of action sequence by enumeration
        if self.action_order == ActionOrder.REVERSE:
            # Reverse: pad first, then concatenate, then reverse order
            action_seq = np.concatenate([action_padding, action_seq], axis=0)
            action_seq = np.flip(action_seq, axis=0)
        elif self.action_order == ActionOrder.FORWARD:
            # Forward: action first, then padding
            action_seq = np.concatenate([action_seq, action_padding], axis=0)
        elif self.action_order == ActionOrder.HYBRID:
            # HYBRID mode example: nbp first (assume idx part is nbp), remaining part in forward order
            # Further design needed according to specific requirements
            action_seq = np.roll(action_seq, 1, axis=0)
            action_seq = np.concatenate([action_seq, action_padding], axis=0)
        else:
            raise ValueError("Unknown action order")
        
        action_seq = np.ascontiguousarray(action_seq) if self._action_seq_len_max > 1 else action_seq

        return action_seq, is_pad

    def convert_to_mtp_actions(self, actions: np.array, is_pad: np.array) -> tuple(np.array, np.array):
        mtp_size = self.cfg.method.actor_model.nmtpheads
        l, d = actions.shape

        mtp_actions = np.zeros((l, mtp_size, d))
        mtp_is_pad = np.ones((l, mtp_size), dtype=bool)  # By default, the padding part is True (i.e., padding)

        for i in range(l):
            end_idx = min(i + mtp_size, l)  # Avoid out of bounds
            extracted = actions[i:end_idx]  
            extracted_pad = is_pad[i:end_idx]  # Extract corresponding `is_pad` values

            # If the extracted data is less than mtp_size, need to pad
            if extracted.shape[0] < mtp_size:
                pad_count = mtp_size - extracted.shape[0]
                extracted = np.vstack((extracted, np.zeros((pad_count, d))))  # Pad actions with 0
                extracted_pad = np.hstack((extracted_pad, np.ones(pad_count, dtype=bool)))  # Mark the padded part with 1

            mtp_actions[i] = extracted
            mtp_is_pad[i] = extracted_pad

        return mtp_actions, mtp_is_pad
    

    def convert_dtype(self, data: dict) -> dict:
        new_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.dtype == float:
                new_data[k] = np.array(v, dtype=np.float32)
            else:
                new_data[k] = v
        return new_data

    def __getitem__(self, episode_idx: int) -> dict:
        """
        Get a sample.
        self._demos is a list of demos, provided by the env factory. if data is loaded for CoA, demos are split into sub-trajectories according to the keyframe action. if for ACT and DP, trajectories are loaded without split.

        Args:
            episode_idx: Index of the current episode
        
        Returns:
            Sample dictionary containing observation, action sequence and related flags
            
        """

        if self.cfg.method_name == "coa":
            return self.get_sample_coa(episode_idx)
        elif self.cfg.method_name in ("act", "dp"):
            return self.get_sample(episode_idx)
        elif self.cfg.method_name == "bip":
            return self.get_sample_bip(episode_idx)
        else:
            raise ValueError(f"Unknown method name: {self.cfg.method_name}")
        


    def get_sample(self, episode_idx: int) -> dict:
        """
        Get a sample for ACT/DP mode.
        
        Args:
            episode_idx: Index of the current episode
        
        Returns:
            sample: Sample dictionary containing observation, action sequence and related flags
            
            **Image observation fields** (CHW format, dtype=uint8):
            - 'left_shoulder_rgb': (1, 3, 128, 128) - Left shoulder camera RGB image  
            - 'right_shoulder_rgb': (1, 3, 128, 128) - Right shoulder camera RGB image
            - 'wrist_rgb': (1, 3, 128, 128) - Wrist camera RGB image
            - 'front_rgb': (1, 3, 128, 128) - Front camera RGB image
            Format: (frame_stack, channels, height, width)
            
            **State fields** (dtype=float32):
             - 'low_dim_state': (8,) - Robot low-dimensional state (e.g. gripper pose + gripper state)
             
             **Action-related fields** (dtype=float32/bool):
             - 'action': (l, 8) - Action sequence
               * l: action_sequence_length (determined by config and trajectory length)
               * 8: action_dim (3 position + 4 orientation + 1 gripper)
             - 'is_pad': (l,) - Padding flags, bool type
               * True indicates this position is padded, False indicates valid action
            
            **Language condition fields** (only when use_lang_cond=True):
            - 'desc': (77,) - CLIP tokenized text description
            
        """
        episode = self._demos[episode_idx]

        actions = episode[ActionModeType[self.cfg.env.action_mode].value]
        ep_len = len(actions) 

        # Determine the sampling idx according to config
        min_idx, max_idx = 0, ep_len - 1
        if self.cfg.debug:
            idx = 0
        else:
            idx = np.random.randint(min_idx, max_idx)
        
        # Get observation sample according to the idx, used for all keys except for action
        sample = self.get_observation(episode, idx)

        # Get action sequence according to the idx
        action_seq, is_pad = self.get_action(actions, ep_len, idx)

        # By default, do not use MTP, use standard format directly
        sample['action'] = action_seq
        sample['is_pad'] = is_pad

        # Squeeze desc dimension (assuming desc is part of observation)
        if self.cfg.method.use_lang_cond:
            sample['desc'] = sample['desc'].squeeze(0) 
        
        # Remove original action as it's only used for action generation
        del sample[ActionModeType[self.cfg.env.action_mode].value]
        
        # convert
        sample = self.convert_dtype(sample)
        
        return sample



    def get_sample_coa(self, episode_idx: int) -> dict:
        """
        Args:
            episode_idx: Index of the current episode
        
        Returns:
            sample: Sample dictionary containing observation, action sequence and related flags
            
            **Image observation fields** (CHW format, dtype=uint8):
            - 'left_shoulder_rgb': (1, 3, 128, 128) - Left shoulder camera RGB image  
            - 'right_shoulder_rgb': (1, 3, 128, 128) - Right shoulder camera RGB image
            - 'wrist_rgb': (1, 3, 128, 128) - Wrist camera RGB image
            - 'front_rgb': (1, 3, 128, 128) - Front camera RGB image
            Format: (frame_stack, channels, height, width)
            
            **State fields** (dtype=float32):
             - 'low_dim_state': (8,) - Robot low-dimensional state (e.g. gripper pose + gripper state)
             
             **Action-related fields** (MTP format, dtype=float32/bool):
             - 'action': (l, nmtpheads, 8) - Action sequence in MTP format
               * l: action_sequence_length (determined by config and trajectory length)
               * nmtpheads (number of multi-head predictions)  
               * 8: action_dim (3 position + 4 orientation + 1 gripper)
             - 'is_pad': (l, nmtpheads) - Padding flags, bool type
               * True indicates this position is padded, False indicates valid action
            
            **Language condition fields** (only when use_lang_cond=True):
            - 'desc': (77,) - CLIP tokenized text description
            
            **Key Features**:
            1. MTP conversion: convert_to_mtp_actions() transforms regular action sequences to multi-head prediction format
               Each timestep predicts future nmtpheads action steps
            2. Action Order: Supports REVERSE/FORWARD/HYBRID three arrangement modes
            3. Padding handling: Uses zero padding for sequences with insufficient length
            4. Data types: Images are uint8, other numerical values are float32
            5. Image format: CHW format (Channels, Height, Width), suitable for PyTorch CNN input
        """

        episode = self._demos[episode_idx]

        actions = episode[ActionModeType[self.cfg.env.action_mode].value]
        ep_len = len(actions) 

        # Determine the sampling idx according to config
        min_idx, max_idx = 0, ep_len - 1
        if self.cfg.method.keyframe_only:
            idx = max_idx
        elif self.cfg.debug:
            idx = 0
        else:
            idx = np.random.randint(min_idx, max_idx - self.traj_sample_margin)
        
        # Get observation sample according to the idx, used for all keys except for action
        sample = self.get_observation(episode, idx)

        # Get action sequence according to the idx
        action_seq, is_pad = self.get_action_coa(actions, max_idx, idx)

        # convert to mtp actions
        assert self.cfg.method.mtp, "mtp must be True"
        if self.cfg.method.mtp:
            action_seq_mtp, is_pad_mtp = self.convert_to_mtp_actions(action_seq, is_pad)
            sample['action'] = action_seq_mtp
            sample['is_pad'] = is_pad_mtp
        else:
            sample['action'] = action_seq
            sample['is_pad'] = is_pad
        
        # Squeeze desc dimension (assuming desc is part of observation)
        if self.cfg.method.use_lang_cond:
            sample['desc'] = sample['desc'].squeeze(0) 
        
        # Remove original action as it's only used for action generation
        del sample[ActionModeType[self.cfg.env.action_mode].value]
        
        #convert
        sample = self.convert_dtype(sample)


        
        return sample



    def get_sample_bip(self, episode_idx: int) -> dict:
        """
        Dataset sample for BIP.

        Keys returned:
          'action'    : (L, 8)        CoA-REVERSE planner target — full remaining
                                       sub-trajectory [a_T, a_{T-1}, ..., a_idx] padded
                                       to L = max sub-trajectory length. Position 0 is
                                       the keyframe. At inference its forward flip seeds
                                       the DP head's diffusion (CoA-guided / SDEdit).
          'is_pad'    : (L,)          padding mask for the planner target
          'action_dp' : (dp_chunk, 8) DP-FORWARD executed chunk [a_{idx+1}, ..., a_{idx+dp_chunk}];
                                       steps past the keyframe hold the keyframe pose a_T.
        """
        episode = self._demos[episode_idx]

        actions = episode[ActionModeType[self.cfg.env.action_mode].value]
        ep_len  = len(actions)
        max_idx = ep_len - 1

        idx = 0 if (self.cfg.debug or ep_len <= 1) else np.random.randint(0, max_idx)

        sample = self.get_observation(episode, idx)

        dp_chunk = self.cfg.method.dp_action_sequence

        # ── CoA planner target: full remaining sub-trajectory in REVERSE order ──
        # Identical to CoA: from current idx up to the keyframe, reversed so that
        # position 0 = a_T (keyframe).
        action_seq, is_pad = self.get_action_coa(actions, max_idx, idx)
        sample['action'] = action_seq
        sample['is_pad'] = is_pad

        # ── DP controller target: FORWARD chunk [a_{idx+1}, ..., a_{idx+dp_chunk}] ──
        # Beyond the keyframe, HOLD the keyframe pose a_T (not zeros) so the controller
        # drives to the goal and stays — zeros would denormalize to a center pose and,
        # since temporal ensembling only filters exactly-zero entries, drag the executed
        # action away from the keyframe (a jump/drift right at the keyframe).
        keyframe_action = actions[max_idx]
        fwd = np.zeros((dp_chunk, actions.shape[-1]), dtype=np.float32)
        for i in range(dp_chunk):
            t = idx + 1 + i
            fwd[i] = actions[t] if t < ep_len else keyframe_action
        sample['action_dp'] = fwd

        if self.cfg.method.use_lang_cond:
            sample['desc'] = sample['desc'].squeeze(0)

        del sample[ActionModeType[self.cfg.env.action_mode].value]
        sample = self.convert_dtype(sample)
        return sample


@hydra.main(config_path="../cfgs/", config_name="launch_act", version_base=None)
def main(cfg: DictConfig):
    """
    Test function for RLBenchDataset with a minimal training loop
    
    This function creates an instance of RLBenchDataset and tests its functionality
    by loading data and running a minimal training loop.
    """
    from src.envs.rlbench.rlbench_env import RLBenchEnvFactory
    env_factory = RLBenchEnvFactory()
    eval_env = env_factory.make_eval_env(cfg)
    demos = env_factory._load_demos(cfg)
    dataset = RLBenchDataset(cfg)
    dataset._demos = demos

    # 测试数据集基本功能
    print(f"Dataset loaded with {len(dataset)} demonstrations")
    
    # 测试第一个样本
    sample = dataset[0]
    print("Successfully loaded sample with keys:", list(sample.keys()))
    print("Action shape:", sample['action'].shape)
    print("Is_pad shape:", sample['is_pad'].shape)

if __name__ == "__main__":
    main()

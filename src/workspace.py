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
Training workflow.
"""
import os
import random
import pickle
from pathlib import Path
from itertools import cycle

import hydra
import torch
import wandb
import numpy as np
import warnings

from typing import Dict, Any, Tuple, Union, Optional

from omegaconf import DictConfig
from natsort import natsorted

from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from accelerate import Accelerator, DistributedDataParallelKwargs

from src.envs.rlbench.wrappers.rescale_from_tanh import MinMaxNorm
from src.envs.rlbench.rlbench_env import RLBenchEnvFactory
from src.envs.base import EnvFactory
from src.dataset.rlbench_dataset import RLBenchDataset
from src.logger import Logger
from src.video import VideoRecorder
from src.utils import make_pcd, merge_pcds



def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class Workspace:
    def __init__(self, cfg, train: bool = True):


        set_seed_everywhere(1)

                # initialize the agent or load the agent from snapshot

        self.cfg = cfg
            
        # get work dir from hydra
        self.work_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        print(f"work dir: {self.work_dir}")
        
        
        # initialize the accelerator
        self.accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        self.device = self.accelerator.device

        # initialize logger
        self.logger = Logger(cfg, self.work_dir)

        # initialize the env
        self.env_factory = RLBenchEnvFactory()
        
        
        # initialize video recorder
        video_dir = self.work_dir / 'eval_videos' if cfg.get('log_eval_video', False) else None
        self.eval_video_recorder = VideoRecorder(video_dir)

        # initialize the dataset and dataloader for training mode
        # or only read the action sequence for eval mode
        if train:
            # initialize the dataset for visualization
            vis_demos, action_sequence_vis = self.env_factory._load_demos(cfg, training=False)
            self.dataset_vis = RLBenchDataset(cfg, seq_len=action_sequence_vis)
            setattr(self.dataset_vis, '_demos', vis_demos)

            # initialize the dataset
            demos, action_sequence_train = self.env_factory._load_demos(cfg, training=True)
            cfg.action_sequence = action_sequence_train
            if cfg.wandb.use:
                wandb.config.update({"action_sequence": action_sequence_train}, allow_val_change=True)
            self.dataset = RLBenchDataset(cfg, seq_len=action_sequence_train)
            setattr(self.dataset, '_demos', demos)

            # Create dataloader for visualization dataset
            self.dataloader_vis = DataLoader(
                self.dataset_vis,
                batch_size=1,
            )
            self.dataloader_vis = cycle(self.dataloader_vis)

            self._current_step = 0
            sampler = RandomSampler(self.dataset, replacement=True, num_samples=cfg.batch_size)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,   
            )

        # initialize the agent and load the agent from snapshot if snapshot is provided
        self.agent = hydra.utils.instantiate(cfg.method, accelerator=self.accelerator)    
        if cfg.snapshot is not None:
            self.load_snapshot(cfg.snapshot)
            
        # initialize the eval env
        # the action_sequence wrapper depends on the action sequence_length, so we need to initialize the eval env after the dataset is initialized
        self.eval_env = self.env_factory.make_eval_env(cfg)

        
    def _get_next_batch(self):
        batch = next(iter(self.dataloader))
        return batch, self.dataloader


    def loop(self):
        """Main training loop"""
        num_train_steps = self.cfg.num_train_steps


        self.logger.info(f"Start training, total steps: {num_train_steps}")
        
        for step in range(num_train_steps):
            batch, _ = self._get_next_batch()
            
            self._current_step += 1
            
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Update model
            updated_metrics = self.agent.update(batch)
            
            
            if self._current_step % self.cfg.log_every_steps == 0:
                self.logger.log_metrics(updated_metrics, step=self._current_step)
                print(f"Step {self._current_step}/{num_train_steps}, Loss: {updated_metrics.get('total_loss', 0):.4f}")
            
            '''
            # The dataset in Huggingface omits point cloud observations to reduce storage requirements.
            # and the point cloud observation is neccessary for the visualization.
            # As a result, the visualization is not available now or you can generate the full eval dataset manually.
            # TODO: upload full eval dataset to huggingface
            '''            

            # if  self._current_step % self.cfg.vis_every_steps == 0:
            #     updated_metrics = self.vis()
            #     self.logger.log_metrics(updated_metrics, step=self._current_step)

            if self._current_step % self.cfg.eval_every_steps == 0:
                updated_metrics = self.eval()
                self.logger.log_metrics(updated_metrics, step=self._current_step)

            if self._current_step % self.cfg.save_every_steps == 0:
                self.save_snapshot()

        self.logger.info("Finished training!")

    def _vis(
        self,
        pcds_ori: Union[np.ndarray, torch.Tensor],
        rgbs_ori: Union[np.ndarray, torch.Tensor],
        predicted_trajs: Union[np.ndarray, torch.Tensor],
        gt_trajs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> dict:
        """
        Generates a dictionary containing visualization metrics for predicted trajectories and point clouds observation.

        Parameters:
            pcds_ori (np.ndarray): The point clouds. Shape: (N, 3).
            rgbs_ori (np.ndarray): The RGB values. Shape: (N, 3).
            predicted_trajs (np.ndarray): The predicted trajectories. Shape: (M, T, 3).
            gt_trajs (np.ndarray, optional): The ground truth trajectories. Shape: (M, T, 3).

        Returns:
            dict: A dictionary containing visualization metrics.
        """
        if isinstance(pcds_ori, torch.Tensor):
            pcds_ori = pcds_ori.cpu().numpy()
        if isinstance(rgbs_ori, torch.Tensor):
            rgbs_ori = rgbs_ori.cpu().numpy()
        if isinstance(predicted_trajs, torch.Tensor):
            predicted_trajs = predicted_trajs.cpu().numpy()
        if gt_trajs is not None and isinstance(gt_trajs, torch.Tensor):
            gt_trajs = gt_trajs.cpu().numpy()
        import plotly.graph_objects as go
        metrics = {}

        sampled_trajs = predicted_trajs
        tx, ty, tz = (
            sampled_trajs[:, 0],
            sampled_trajs[:, 1],
            sampled_trajs[:, 2],
        )

        if gt_trajs is not None:
            gx, gy, gz = gt_trajs[:, 0], gt_trajs[:, 1], gt_trajs[:, 2]

        pcds = pcds_ori.reshape(-1, 3)

        bound_min, bound_max = 0, 0.5

        # Ensure rgbs_ori is numpy array
        if isinstance(rgbs_ori, torch.Tensor):
            rgbs_ori = rgbs_ori.cpu().numpy()
        rgbs = (255*rgbs_ori).reshape(-1, 3).astype(np.uint8)

        bound = np.array([bound_min, bound_max], dtype=np.float32)

        pcd_mask = (pcds > bound[0:1]) * (pcds < bound[1:2])
        pcd_mask = np.all(pcd_mask, axis=1)
        indices = np.where(pcd_mask)[0]

        rgb_strings = [
            f"rgb{rgbs[i][0],rgbs[i][1],rgbs[i][2]}" for i in range(len(rgbs))
        ]

        pcd_plots = [
            go.Scatter3d(
                x=pcds[:, 0],
                y=pcds[:, 1],
                z=pcds[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=rgb_strings,
                ),
            )
        ]

        plot_data = pcd_plots
        for i in range(len(sampled_trajs)):
            tx, ty, tz = (
                sampled_trajs[i:i+1, 0],
                sampled_trajs[i:i+1, 1],
                sampled_trajs[i:i+1, 2],
            )
            color = f"rgba(0, 255, 0, {1 - i / len(sampled_trajs)})"
            plot_data = [
                go.Scatter3d(
                    x=tx,
                    y=ty,
                    z=tz,
                    mode="markers",
                    marker=dict(size=6, color=color),
                )] + plot_data

        gt_plot = plot_data
        if gt_trajs is not None:
            for i in range(len(gt_trajs)):
                gx, gy, gz = gt_trajs[i:i+1, 0], gt_trajs[i:i+1, 1], gt_trajs[i:i+1, 2]
                color = f"rgba(255, 0, 0, {1 - i / len(gt_trajs)})"
                gt_plot = [
                    go.Scatter3d(
                        x=gx,
                        y=gy,
                        z=gz,
                        mode="markers",
                        marker=dict(size=6, color=color),
                    )
                ] + gt_plot

        fig = go.Figure(gt_plot)
        fig.update_layout(
            scene=dict(
                aspectmode='cube',  # Fix aspect ratio to be the same for all axes
                xaxis=dict(range=[-0.3, 0.7]),  # Set x-axis range
                yaxis=dict(range=[-0.5, 0.5]),  # Set y-axis range
                zaxis=dict(range=[0.76, 1.6])        # Set z-axis range (adjust based on your data)
            )
        )
        # fig.show()
        # store fig in disk
        # fig.write_image("traj.png")
        metrics[f"trajectories"] = fig
        return metrics


    def vis(self,) -> dict[str, Any]:
        """
        Visualize offline demo loaded from RLBenchDataset. This version loads demo steps via RLBenchDataset,
        and uses _extract_obs(training=False) to retain point cloud etc. for visualization.
        """

        batch = next(self.dataloader_vis)
        
        if self.cfg.debug:
            idx = 0
        else:
            idx = random.randint(0, batch['action'].shape[0] - 1)
        # Automatically collect all existing point_cloud/rgb pairs
        pcd_list = []
        for k in batch.keys():
            if k.endswith('_point_cloud'):
                cam = k[:-12]  # Remove _point_cloud suffix
                rgb_key = f'{cam}_rgb'
                if rgb_key in batch:
                    pcd_input = batch[k][idx].permute(0, 2, 3, 1).reshape(-1, 3)
                    rgb_input = batch[rgb_key][idx].permute(0, 2, 3, 1).reshape(-1, 3)
                    pcd = make_pcd(pcd_input, rgb_input)
                    pcd_list.append(pcd)
        if not pcd_list:
            raise RuntimeError('No valid point cloud and rgb pairs found in batch!')
        pcd_o3d = merge_pcds(pcd_list)
        pcd = np.asarray(pcd_o3d.points)
        pcd_color = np.asarray(pcd_o3d.colors)
        # pcd_rgb = np.concatenate([pcd, pcd_color], axis=-1)

        gt_trajs = batch['action'][idx][:,0,:]
        if isinstance(gt_trajs, torch.Tensor):
            gt_trajs = gt_trajs.cpu().detach().numpy()
        # print(gt_trajs[:,2][::-1])
        gt_trajs = MinMaxNorm.denormalize(gt_trajs, self.env_factory.get_action_space(self.cfg))

        valid_mask = ~batch['is_pad'][idx][:,0]
        gt_trajs = gt_trajs[valid_mask]

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) and  'point_cloud' not in k else v for k, v in batch.items()}
        predicted_trajs = self.agent.act(batch)
        if isinstance(predicted_trajs, torch.Tensor):
            predicted_trajs = predicted_trajs.cpu().detach().numpy()
        predicted_trajs = MinMaxNorm.denormalize(predicted_trajs, self.env_factory.get_action_space(self.cfg))


        metrics = self._vis(pcd, pcd_color, predicted_trajs[0], gt_trajs)
        return metrics

    def _perform_env_steps(
        self,
        observations: Dict[str, np.ndarray],
        info: Dict[str, Any],
        env,
        eval_mode: bool = True,
    ) -> Tuple[np.ndarray, Tuple, Dict[str, Any]]:
        """
        Perform environment interaction step (只支持eval模式).
        
        Args:
            observations: Current observations from environment
            info: Environment info
            env: Environment instance
            eval_mode: Should always be True for this implementation
            
        Returns:
            Tuple of (action, env_step_result, metrics)
        """
        metrics = {}
        
        with torch.no_grad():
            # Convert observations to torch tensors
            torch_observations = {
                k: torch.from_numpy(v).to(self.device) for k, v in observations.items()
            }
            
            # Handle description if present
            if "desc" in info:
                torch_observations["desc"] = torch.from_numpy(info["desc"]).to(self.device)
            
            # Add batch dimension for eval
            if eval_mode:
                torch_observations = {
                    k: v.unsqueeze(0) for k, v in torch_observations.items()
                }
            
            # Get action from agent
            action = self.agent.act(torch_observations)
            
            # Handle agent info if returned as tuple
            if isinstance(action, tuple):
                action, act_info = action
                metrics["agent_act_info"] = act_info
                
            # Convert to numpy
            action = action.cpu().detach().numpy()
            
            # Validate action shape
            if action.ndim != 3:
                raise ValueError(
                    "Expected actions from `agent.act` to have shape "
                    "(Batch, Timesteps, Action Dim)."
                )
            
            # Remove batch dimension for eval
            if eval_mode:
                action = action[0]  # expecting batch size of 1 for eval

        # Execute action in environment
        next_observation, reward, termination, truncation, next_info = env.step(action)
        
        return action, (next_observation, reward, termination, truncation, next_info), metrics

    def eval(self) -> Dict[str, Any]:
        """
        Complete evaluation loop with environment interaction.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        step, episode, total_reward, successes = 0, 0, 0, 0
        eval_episodes = self.cfg.get('num_eval_episodes')
        metrics = {}

        
        pkl_list = []
        # Get task name from config 
        task_name = self.cfg.env.task_name
        if self.cfg.get('debug', False):
            dataset_root = self.cfg.dataset_root_train
        else:
            dataset_root = self.cfg.dataset_root_eval
        
        if not os.path.exists(dataset_root):
            print(f"Warning: Eval dataset root {dataset_root} does not exist")
            return {"episode_success": 0.0, "episode_length": 0.0}
            
        for i in range(eval_episodes):
            episode_dir = os.path.join(dataset_root, task_name, 'variation0', 'episodes', f'episode{i}')
            pkl_path = os.path.join(episode_dir, 'low_dim_obs.pkl')
            if os.path.exists(pkl_path):
                pkl_list.append(pkl_path)

        pkl_list = natsorted(pkl_list)
        
        if not pkl_list:
            print(f"Warning: No evaluation episodes found in {dataset_root}/{task_name}")
            return {"episode_success": 0.0, "episode_length": 0.0}

        video_success = []
        video_fail = []
        
        episodes_completed = 0
        ik_error = 0
        num_episodes = min(eval_episodes, len(pkl_list))
        for episode_idx in range(num_episodes):
            try:
                # Load and reset environment to demo
                with open(pkl_list[episode_idx], 'rb') as f:
                    pkl = pickle.load(f)
                # Use reset_to_demo method, pass pkl data
                observation, info = self.eval_env.reset_to_demo(pkl)
                
                # Initialize video recording
                enabled = self.cfg.get('log_eval_video') 

                self.eval_video_recorder.init(self.eval_env, enabled=enabled)
                
                # Episode loop
                termination, truncation = False, False
                episode_steps = 0
                
                while not (termination or truncation):
                    # Perform environment step
                    action, (next_observation, reward, termination, truncation, next_info), env_metrics = \
                        self._perform_env_steps(observation, info, self.eval_env, eval_mode=True)
                    
                    # Update for next step
                    observation = next_observation
                    info = next_info
                    metrics.update(env_metrics)
                    
                    # Record video frame
                    self.eval_video_recorder.record(self.eval_env)
                    
                    # Update counters
                    total_reward += reward
                    step += 1
                    episode_steps += 1

                # Save video and categorize by success/failure
                video = np.array(self.eval_video_recorder.frames)
                if len(video) > 0:
                    if reward == 1:
                        video_success.append(video)
                    else:
                        video_fail.append(video)
                        
                self.eval_video_recorder.save(f"episode_{episode_idx}.mp4")
                
                # Track success
                success = info.get("task_success", reward)
                successes += int(success > 0)
                if success > 0:
                    print(f"episode {episode_idx} success")
                else:
                    print(f"episode {episode_idx} fail")
                
                
                
            except Exception as e:
                print(f"episode {episode_idx} failed for: {e}")
                import traceback
                traceback.print_exc()
                ik_error += 1


            episodes_completed += 1
        # Compute final metrics
        if episodes_completed > 0:
            metrics.update({
                "success_rate": successes / episodes_completed,
                # "sr_no_ik_error": (successes - ik_error) / (episodes_completed - ik_error),
                "episode_length": step / episodes_completed,
                "ik_error_rate": ik_error / episodes_completed,
                "num_episodes": num_episodes,
            })
        else:
            metrics.update({
                "success_rate": 0.0,
                # "sr_no_ik_error": 0.0,
                "episode_length": 0.0,
                "ik_error_rate": 0.0,
            })

        self.logger.info(f"{self.cfg.method_name} on {task_name}: {metrics}") 
        
        # Add video rollout if enabled
        if self.cfg.get('log_eval_video', False):
            metrics["eval_rollout"] = {
                "video_success": video_success,
                "video_fail": video_fail,
                "fps": 10
            }
            
        return metrics

    def save_snapshot(self):
        """Save model checkpoint"""
        # Use experiment directory created by Hydra
        save_dir = self.work_dir / 'checkpoints'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'config': self.cfg,
            'step': self._current_step,
            # 'optimizer_state_dict': self.agent.opt.state_dict(),
            # 'scheduler_state_dict': self.agent.lr_scheduler.state_dict() if hasattr(self.agent, 'lr_scheduler') else None,
        }
            
        # Save checkpoint
        checkpoint_path = save_dir / f'{self.cfg.method_name}_{self._current_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_snapshot(self, path: str) :
        """Load model checkpoint and return a new Workspace instance initialized with ckpt config."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint file {checkpoint_path} does not exist")
            return None

        # Load checkpoint: this project stores OmegaConf objects in checkpoints.
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            # Backward compatibility for older PyTorch versions.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Compatible with old ckpt without cfg field
        self.cfg_ckpt = checkpoint.get("config", None)
        if self.cfg_ckpt is None:
            raise RuntimeError("Checkpoint does not contain config. Cannot initialize Workspace.")


        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        if hasattr(self.agent, "opt") and "optimizer_state_dict" in checkpoint:
            self.agent.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if hasattr(self.agent, "lr_scheduler") and "scheduler_state_dict" in checkpoint:
            self.agent.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore step
        self._current_step = checkpoint.get("step", 0)
        print(f"Checkpoint loaded from {checkpoint_path}")

@hydra.main(config_path="cfgs/", config_name="launch", version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.loop()

if __name__ == "__main__":
    main()
    # 

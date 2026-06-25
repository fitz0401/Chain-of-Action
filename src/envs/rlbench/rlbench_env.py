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

from scipy.spatial.transform import Rotation as R
import wandb
import clip 
import os
import time
import importlib
import warnings
from typing import List
from enum import Enum
from functools import partial
import copy

from omegaconf import DictConfig
from pyrep.const import RenderMode
from pyrep.objects import Dummy, VisionSensor
from gymnasium.spaces import Box
from src.utils import DemoStep

from src.envs.rlbench.wrappers import (
    FrameStack,
    RescaleFromTanh,
    MinMaxNorm,
    RescaleFromTanhWithStandardization,
    RescaleFromTanhWithMinMax,
    ActionSequence,
    ReverseTemporalEnsemble,
    TemporalEnsemble,
    AppendDemoInfo,
    LangWrapper,
    TimeLimitX,
)


from src.utils import (
    rescale_demo_actions,
)
from src.envs.base import EnvFactory, Demo
from src.envs.rlbench.wrappers.rescale_from_tanh import MinMaxNorm
from src.envs.rlbench.wrappers.rescale_from_tanh import get_action_space_from_cfg

from src.envs.rlbench.rlbench_utils import get_stored_demos
from src.envs.rlbench.rlbench_utils import get_stored_demos_in_pkl
from src.envs.rlbench.arm_action_modes import EndEffectorPoseViaPlanningX
try:
    from rlbench import ObservationConfig, Environment, CameraConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK

    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.backend.observation import Observation
    from rlbench.backend.exceptions import InvalidActionError

except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ActionModeType(Enum):
    ABS_END_EFFECTOR_POSE = "abs_ee_pose"
    ABS_JOINT_POSITION = "abs_joint_pos"
    DEL_END_EFFECTOR_POSE = "del_ee_pose"
    DEL_JOINT_POSITION = "del_joint_pos"
    HYBRID = "hybrid"

ACTION_BOUNDS = {
    ActionModeType.ABS_END_EFFECTOR_POSE :(
                    np.array(
                        [-0.28, -0.66, 0.75] + 3 * [-1.0] + 2 * [0.0],
                        dtype=np.float32,
                    ),
                    np.array([0.78, 0.66, 1.75] + 4 * [1.0] + [1.0], dtype=np.float32),
                ),
    ActionModeType.ABS_JOINT_POSITION: (
                    np.array(7 * [-np.pi] + [0.0], dtype=np.float32), #  I'm not sure, to be fixed
                    np.array(7 * [np.pi] + [1.0], dtype=np.float32),
                ),
    ActionModeType.DEL_END_EFFECTOR_POSE: ( # TODO: to be fixed
                    np.array(
                        [-0.28, -0.66, 0.75] + 3 * [-1.0] + 2 * [0.0],
                        dtype=np.float32,
                    ),
                    np.array([0.78, 0.66, 1.75] + 4 * [1.0] + [1.0], dtype=np.float32),
                ),
    ActionModeType.DEL_JOINT_POSITION: (
                    np.array(7 * [-np.pi] + [0.0], dtype=np.float32),
                    np.array(7 * [np.pi] + [1.0], dtype=np.float32),
                ),
    ActionModeType.HYBRID: (
                    np.array(
                        [-0.28, -0.66, 0.75] + 3 * [-1.0] + 2 * [0.0],
                        dtype=np.float32,
                    ),
                    np.array([0.78, 0.66, 1.75] + 4 * [1.0] + [1.0], dtype=np.float32),
                ),
}


ROBOT_STATE_KEYS = [
    "joint_velocities",
    "joint_positions",
    "joint_forces",
    "gripper_open",
    "gripper_pose",
    "gripper_matrix",
    "gripper_joint_positions",
    "gripper_touch_forces",
    "task_low_dim_state",
    "misc",
]

TASK_TO_LOW_DIM_SIM = {
    "reach_target": 3,
    "pick_and_lift": 6,
    "take_lid_off_saucepan": 7,
}

def _get_cam_observation_elements(camera: CameraConfig, prefix: str):
    space_dict = {}
    img_s = camera.image_size
    if camera.rgb:
        space_dict["%s_rgb" % prefix] = spaces.Box(
            0, 255, shape=(3,) + img_s, dtype=np.uint8
        )

    # TODO: point cloud is not supported yet
    # if camera.point_cloud: 
    #     space_dict["%s_point_cloud" % prefix] = spaces.Box(
    #         -np.inf, np.inf, shape=(3,) + img_s, dtype=np.float32
    #     )
    #     space_dict["point_cloud_merge"] = spaces.Box(
    #         -np.inf, np.inf, shape=(4096, 3), dtype=np.float32 # to be fixed
    #     )
    #     space_dict["point_cloud_merge_color"] = spaces.Box(
    #         -np.inf, np.inf, shape=(4096, 6), dtype=np.float32 # to be fixed
    #     )
        # space_dict["%s_camera_extrinsics" % prefix] = spaces.Box(
        #     -np.inf, np.inf, shape=(4, 4), dtype=np.float32
        # )
        # space_dict["%s_camera_intrinsics" % prefix] = spaces.Box(
        #     -np.inf, np.inf, shape=(3, 3), dtype=np.float32
        # )

    if camera.depth:
        space_dict["%s_depth" % prefix] = spaces.Box(
            0, np.inf, shape=(1,) + img_s, dtype=np.float32
        )
    if camera.mask:
        raise NotImplementedError()
    return space_dict


def _observation_config_to_gym_space(observation_config, action_mode_type) -> spaces.Dict:
    space_dict = {}
    if action_mode_type == ActionModeType.HYBRID:
         robot_state_len = 15
    elif action_mode_type == ActionModeType.ABS_JOINT_POSITION or action_mode_type == ActionModeType.ABS_END_EFFECTOR_POSE:
        robot_state_len = 8
    if robot_state_len > 0:
        space_dict["low_dim_state"] = spaces.Box(
            -np.inf, np.inf, shape=(robot_state_len,), dtype=np.float32
        )
    for cam, name in [
        (observation_config.left_shoulder_camera, "left_shoulder"),
        (observation_config.right_shoulder_camera, "right_shoulder"),
        (observation_config.front_camera, "front"),
        (observation_config.wrist_camera, "wrist"),
        (observation_config.overhead_camera, "overhead"),
    ]:
        space_dict.update(_get_cam_observation_elements(cam, name))
    return spaces.Dict(space_dict)


def _name_to_task_class(task_file: str):
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    try:
        mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise ValueError(
            "The env file '%s' does not exist or cannot be compiled." % name
        ) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise ValueError(
            "Cannot find the class name '%s' in the file '%s'." % (class_name, name)
        ) from e
    return task_class




def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(
    demo: Demo, stopping_delta: float = 0.01, method: str = "heuristic",
    gripper_only: bool = False,
) -> list[int]:
    """Discover next-best-pose keypoints in a demonstration based on specified method.

    Args:
        demo: A demonstration represented as a list of Observation objects.
        stopping_delta: Tolerance for considering joint
            velocities as "stopped". Defaults to 0.1.
        method: The method for discovering keypoints.
            - "heuristic": Uses a heuristic approach.
            - "random": Randomly selects keypoints.
            - "fixed_interval": Selects keypoints at fixed intervals.
            Defaults to "heuristic".

    Returns:
        List of indices representing the discovered keypoints in the demonstration.
    """
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            # gripper_only: ignore the low-velocity "stopped" criterion (BIP).
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or (stopped and not gripper_only)):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        # print('Found %d keypoints.' % len(episode_keypoints),
        #               episode_keypoints)
        return episode_keypoints
    elif method == "random":
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo._observations)), size=20, replace=False
        )
        episode_keypoints.sort()
        return episode_keypoints

    elif method == "fixed_interval":
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo._observations) // 20
        for i in range(0, len(demo._observations), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


class RLBenchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        task_name: str,
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        action_mode_type: ActionModeType = ActionModeType.DEL_JOINT_POSITION,
        arm_max_velocity: float = 1.0,
        arm_max_acceleration: float = 4.0,
        dataset_root_train: str = "",
        dataset_root_eval: str = "",
        renderer: str = "opengl",
        headless: bool = True,
        render_mode: str = None,
        use_lang_cond: bool = False,
        cfg: DictConfig = None,
    ):
        self._task_name = task_name
        self._observation_config = observation_config
        self._action_mode = action_mode
        self._action_mode_type = action_mode_type
        self._arm_max_velocity = arm_max_velocity
        self._arm_max_acceleration = arm_max_acceleration
        self._dataset_root_train = dataset_root_train
        self._dataset_root_eval = dataset_root_eval
        self._headless = headless
        self._use_lang_cond = use_lang_cond
        self._rlbench_env = None
        self.action_mode_type = action_mode_type
        self._cfg = cfg
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_space = _observation_config_to_gym_space(
            observation_config, action_mode_type
        )
        minimum, maximum = action_mode.action_bounds()
        self.action_space = spaces.Box(
            minimum, maximum, shape=maximum.shape, dtype=maximum.dtype
        )
        if renderer == "opengl":
            self.renderer = RenderMode.OPENGL
        elif renderer == "opengl3":
            self.renderer = RenderMode.OPENGL3
        else:
            raise ValueError(self.renderer)

    def get_observation_config(self):
        return self._observation_config
    
    def get_action_mode_type(self):
        return self._action_mode_type
    
    def _launch(self):
        task_class = _name_to_task_class(self._task_name)
        self._rlbench_env = Environment(
            action_mode=self._action_mode,
            obs_config=self._observation_config,
            dataset_root=self._dataset_root_train,
            headless=self._headless,
            arm_max_velocity=self._arm_max_velocity,
            arm_max_acceleration=self._arm_max_acceleration,
        )
        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(task_class)
        if self.render_mode is not None:
            self._add_video_camera()

    def _add_video_camera(self):
        cam_placeholder = Dummy("cam_cinematic_placeholder")
        self._cam = VisionSensor.create([320, 192], explicit_handling=True)
        self._cam.set_pose(cam_placeholder.get_pose())
        self._cam.set_render_mode(self.renderer)

    def _render_frame(self) -> np.ndarray:
        self._cam.handle_explicitly()
        frame = self._cam.capture_rgb()
        frame = np.clip((frame * 255.0).astype(np.uint8), 0, 255)
        return frame

        rlb_obs, reward, term = self._task.step(action)

        return rlb_obs, reward, term

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def step(self, action):
        # assert self.cfg.action_mode == ActionModeType:
        # if self._action_mode_type == ActionModeType.HYBRID:
        #     action = np.concatenate((action[:7], action[-1:]), axis=-1)
        if self._action_mode_type == ActionModeType.HYBRID:
            action = np.concatenate((action[:7], action[-1:]), axis=-1)
        rlb_obs, reward, term = self._task.step(action)
        obs = _extract_obs(rlb_obs, self._observation_config, training=False, cfg=self._cfg, action_space=self.action_space)
        return obs, reward, term, False, {"demo": 0, "task_success": int(reward > 0)}

    def reset(self, seed=None, options=None, robot_state_keys: dict = None):
        super().reset(seed=seed)
        if self._rlbench_env is None:
            self._launch()
        desc, rlb_obs = self._task.reset()

        obs = _extract_obs(rlb_obs, self._observation_config, robot_state_keys, training=False, cfg=self._cfg, action_space=self.action_space)
        info = {"demo": 0}
        if self._use_lang_cond:
            info["desc"] = desc
        return obs, info

    def reset_to_demo(self, demo: Demo, seed=None, options=None, robot_state_keys: dict = None):
        super().reset(seed=seed)
        if self._rlbench_env is None:
            self._launch()
        demo.restore_state()
        variation_index = demo._observations[0].misc["variation_index"]
        self._task.set_variation(variation_index)
        desc, rlb_obs = self._task.reset(demo)
        obs = _extract_obs(rlb_obs, self._observation_config, robot_state_keys, training=False, cfg=self._cfg, action_space=self.action_space)
        info = {"demo": 0}
        if self._use_lang_cond:
            info["desc"] = desc
        return obs, info
    
    def close(self):
        if self._rlbench_env is not None:
            self._rlbench_env.shutdown()

def _make_obs_config(cfg: DictConfig):
    pixels = cfg.pixels

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(False)
    obs_config.set_all_high_dim(False)
    obs_config.gripper_open = True
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_pose = True

    if pixels:
        if cfg.env.renderer == "opengl":
            renderer = RenderMode.OPENGL
        elif cfg.env.renderer == "opengl3":
            renderer = RenderMode.OPENGL3
        else:
            raise ValueError(cfg.env.renderer)

        for camera in cfg.env.cameras:
            camera_config = getattr(obs_config, f"{camera}_camera")
            setattr(camera_config, "rgb", True)
            # setattr(camera_config, "point_cloud", True) # TODO: point cloud is not supported yet
            setattr(camera_config, "image_size", cfg.visual_observation_shape)
            setattr(camera_config, "render_mode", renderer)
            setattr(obs_config, f"{camera}_camera", camera_config)
    else:
        obs_config.task_low_dim_state = True

    return obs_config






def _get_spaces(cfg, space_list):
    obs_config = _make_obs_config(cfg)
    rlb_env = _make_env(cfg, obs_config)
    space_list.append(
        (rlb_env.unwrapped.observation_space, rlb_env.unwrapped.action_space)
    )
    rlb_env.close()



def _get_action_mode(action_mode_type: ActionModeType):
    
    class CustomMoveArmThenGripper(MoveArmThenGripper):
        def action_bounds(self):
            return ACTION_BOUNDS[action_mode_type]
    
    # Select arm action mode according to action mode type
    if action_mode_type ==  ActionModeType.ABS_END_EFFECTOR_POSE:
        arm_mode = EndEffectorPoseViaPlanningX(absolute_mode=True)
    elif action_mode_type == ActionModeType.DEL_END_EFFECTOR_POSE:
        arm_mode = EndEffectorPoseViaPlanningX(absolute_mode=False)
    elif action_mode_type == ActionModeType.DEL_JOINT_POSITION:
        arm_mode = JointPosition(False)  
    elif action_mode_type == ActionModeType.ABS_JOINT_POSITION:
        arm_mode = JointPosition(True)   
    else:
        raise ValueError(f"Unsupported action mode type: {action_mode_type}")
    
    return CustomMoveArmThenGripper(arm_mode, Discrete())


class RLBenchEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg, return_raw_spaces=False, demo_env=False):
        self.cfg = cfg
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)
        if ActionModeType[cfg.env.action_mode] == ActionModeType.ABS_END_EFFECTOR_POSE:
            rescale_from_tanh_cls = MinMaxNorm
        else:
            assert not (
                cfg.use_standardization and cfg.use_min_max_normalization
            ), "You can't use both standardization and min/max normalization."
            if cfg.use_standardization:
                # Use demo-based standardization for actions
                assert cfg.demos > 0
                rescale_from_tanh_cls = partial(
                    RescaleFromTanhWithStandardization,
                    action_stats=self._action_stats,
                )
            elif cfg.use_min_max_normalization:
                # Use demo-based min/max normalization for actions
                assert cfg.demos > 0
                rescale_from_tanh_cls = partial(
                    RescaleFromTanhWithMinMax,
                    action_stats=self._action_stats,
                    min_max_margin=cfg.min_max_margin,
                )
            else:
                rescale_from_tanh_cls = RescaleFromTanh
        # env = action_filter(env)
        env = rescale_from_tanh_cls(env)
        env = TimeLimitX(env, cfg.env.episode_length//cfg.env.episode_length_decay_rate)
        # if cfg.use_onehot_time_and_no_bootstrap:
        #     env = OnehotTime(env, cfg.env.episode_length//cfg.env.episode_length_decay_rate)

        # As RoboBase replay buffer always store single-step transitions. demo-env
        # should ignores action sequence and frame stack wrapper.
        if not demo_env:
            env = FrameStack(env, cfg.env.frame_stack)

            # If action_sequenceaction_sequence length and execution length are the same, we do not
            # use receding horizon wrapper.
            # NOTE: for RL, action_sequence == execution_length == 1, so
            #       RecedingHorizonControl won't be enabled.
            if cfg.action_sequence == cfg.execution_length:
                env = ActionSequence(
                    env,
                    cfg.action_sequence,
                )
            else:
                if cfg.method_name == "coa":
                    env = ReverseTemporalEnsemble(
                    env,
                    cfg.action_sequence,
                    cfg.env.episode_length,
                    cfg.execution_length,
                    cfg.temporal_ensemble,
                        cfg.temporal_ensemble_gain,
                        action_order="REVERSE"
                    )
                elif cfg.method_name == "bip":
                    # BIP act() returns a variable-length action: a single planned pose
                    # (move-to-segment-start phase) or a DP chunk (control phase). A plain
                    # ActionSequence executes whatever it is, open-loop, then re-plans.
                    env = ActionSequence(env, cfg.method.dp_action_sequence)
                else:
                    env = TemporalEnsemble(
                        env,
                        cfg.action_sequence,
                        cfg.env.episode_length,
                        cfg.execution_length,
                        cfg.temporal_ensemble,
                        cfg.temporal_ensemble_gain,
                    )

        env = AppendDemoInfo(env)
        if cfg.method.use_lang_cond:
            env = LangWrapper(env)
        if return_raw_spaces:
            return env, (action_space, observation_space)
        else:
            return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        obs_config = _make_obs_config(cfg)

        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(_make_env(cfg, obs_config), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        obs_config = _make_obs_config(cfg)
        # NOTE: Assumes workspace always creates eval_env in the main thread
        env, (self._action_space, self._observation_space) = self._wrap_env(
            _make_env(cfg, obs_config), cfg, return_raw_spaces=True
        )
        return env
    
    def get_action_space(self, cfg):
        return get_action_space_from_cfg(cfg)

    def _load_demos(self, cfg,training=True):
        self.training = training
        dataset_root_dir = cfg.dataset_root_train
        
        obs_config = _make_obs_config(cfg)
        obs_config_demo = copy.deepcopy(obs_config)
        num_demos = cfg.demos if training else cfg.num_eval_episodes

        # RLBench demos are all saved in same action mode (joint).
        # For conversion to an alternate action mode, additional
        # info may be required. ROBOT_STATE_KEYS is altered to
        # reflect this and ensure low_dim_state is consitent
        # for demo and rollout steps.

        action_mode_type = getattr(ActionModeType, cfg.env.action_mode)
        if action_mode_type in [ActionModeType.ABS_END_EFFECTOR_POSE, ActionModeType.HYBRID, ActionModeType.DEL_END_EFFECTOR_POSE]:
            for attr in ['joint_velocities', 'gripper_matrix', 'task_low_dim_state', 'gripper_pose']:
                setattr(obs_config_demo, attr, True)
        elif action_mode_type in [ActionModeType.DEL_JOINT_POSITION, ActionModeType.ABS_JOINT_POSITION]:
            obs_config_demo.joint_velocities = True
        else:
            raise ValueError(f"Unsupported action mode type: {cfg.env.action_mode}")


        demo_state_keys = copy.deepcopy(ROBOT_STATE_KEYS)

        '''
        get raw demos from from rlbench

        format of raw demos:
        [demo1, demo2, demo3, ...]
        each demo is a list of timesteps
        each timestep is a dict of observations, contains:
            - front_rgb
            - wrist_rgb
            - left_shoulder_rgb
            - right_shoulder_rgb
            - joint_positions
            - gripper_open
            - gripper_matrix
            - task_low_dim_state
            - gripper_pose
            ...
        '''
        if cfg.env.is_pkl:
            raw_demos = get_stored_demos_in_pkl(
                num_demos,
                False,
                dataset_root_dir,
                0,
                cfg.env.task_name,
                obs_config_demo,
                random_selection =False,
                from_episode_number=0,
            )
        else:
            raw_demos = get_stored_demos(
                num_demos,
                False,
                dataset_root_dir,
                0,
                cfg.env.task_name,
                obs_config_demo,
                random_selection =False,
                from_episode_number=0,
            )

        # Split each trajectory into a list of sub-trajectories according to the keyframe action.
        bip_seg_idx = None
        if cfg.method_name == "coa":
            demos = self._traj_split(raw_demos)
            action_sequence = self._update_action_sequence_length(cfg, demos)
        elif cfg.method_name == "bip":
            # BIP: gripper-only keyframe split + offline variance interval marking.
            demos, bip_seg_idx = self._traj_split_indexed(
                raw_demos, gripper_only=cfg.method.gripper_only_keyframe)
            action_sequence = None  # set after conversion (needs denormalized poses)
        else:
            demos = raw_demos
            action_sequence = cfg.action_sequence

        '''
        Convert raw demos to the format that is ready for loading and remove unuseful keys
        the format is:
        [demo1, demo2, demo3, ...]
        each demo is a list of timesteps
        each timestep is a dict of observations, contains:
            - front_rgb
            - wrist_rgb
            - left_shoulder_rgb
            - right_shoulder_rgb
            - joint_positions
            - gripper_open
            - gripper_matrix
            - task_low_dim_state
            - gripper_pose
            ...
        '''
        demos_to_load = self._convert_demos_to_loaded_format(demos, cfg)

        if cfg.method_name == "bip":
            action_sequence = self._mark_variance_start(demos_to_load, bip_seg_idx, cfg)

        return demos_to_load, action_sequence


    def _update_action_sequence_length(self, cfg, demos_to_load):
        '''
        Dynamic action sequence length for CoA method

        if NBP only, action sequence length is 1
        if not NBP only, action sequence length is the max length of all demos
        '''
        if cfg.method.keyframe_only:
            cfg.action_sequence = 1
        else:
            max_sequence_length = max(
                len(demo) for demo in demos_to_load
            )
            assert max_sequence_length > 0

        return max_sequence_length
    
    def _traj_split(self, demos):
        '''
        Core design of Chain-of-Action:
        split each trajectory into a list of sub-trajectories according to the keyframe action.

        keypoint_discovery: find the keyframe action in the trajectory
        keypoint_discovery is a function that takes a trajectory and returns a list of keyframe actions.
        the keyframe action is the action that is used to split the trajectory into a sub-trajectory.
        the sub-trajectory follows the same action mode as the original trajectory.
        '''

        nbp_demos = []
        for idx_demo, demo in enumerate(demos):
            episode_keypoints = keypoint_discovery(demo)
            # modify key point if it is too close to the beginning
            if episode_keypoints[0]<10:
                assert len(episode_keypoints)>1
                episode_keypoints[0] = 0
            else:
                episode_keypoints = [0,] + episode_keypoints
            for idx in range(len(episode_keypoints)-1):
                next_idx = idx + 1
                nbp_demo = demo[episode_keypoints[idx]:episode_keypoints[next_idx]+1]
                # nbp_demo[-1].grippe  r_open = nbp_demo[-2].gripper_open
                nbp_demos.append(nbp_demo)
        return nbp_demos

    def _traj_split_indexed(self, demos, gripper_only=True):
        """Like _traj_split but keyframes are split by GRIPPER state only (BIP), and
        the segment index within each demo is tracked alongside each sub-trajectory."""
        segments, seg_idx = [], []
        for demo in demos:
            episode_keypoints = keypoint_discovery(demo, gripper_only=gripper_only)
            if episode_keypoints[0] < 10:
                assert len(episode_keypoints) > 1
                episode_keypoints[0] = 0
            else:
                episode_keypoints = [0, ] + episode_keypoints
            for idx in range(len(episode_keypoints) - 1):
                segments.append(demo[episode_keypoints[idx]:episode_keypoints[idx + 1] + 1])
                seg_idx.append(idx)
        return segments, seg_idx

    def _mark_variance_start(self, demos, seg_idx, cfg):
        """Offline (BIP): per segment-index, find the near-goal low-variance interval.

        Each segment's EE positions are expressed in the keyframe's SE(3) frame
        (translate + rotate by the inverse keyframe orientation). For each offset k
        from the keyframe, the spatial variance of that aligned position across all
        demos sharing the segment index is computed; walking outward from the goal,
        the interval ends at the first offset whose variance exceeds
        `variance_threshold` (floored at `min_interval_length`). Each segment is
        annotated with 'variance_start' (the index where its small-variance region
        begins). Returns the max CoA target length (= action_sequence for CoA head).
        """
        from collections import defaultdict
        key = getattr(ActionModeType, cfg.env.action_mode).value
        action_space = self.get_action_space(cfg)
        min_len = int(cfg.method.min_interval_length)
        thr = float(cfg.method.variance_threshold)

        # EE positions in each segment's keyframe frame (full SE(3) normalization)
        local_pos = []
        for d in demos:
            raw = MinMaxNorm.denormalize(np.asarray(d[key], dtype=np.float32), action_space)
            R_kf = R.from_quat(raw[-1, 3:7])
            local_pos.append(R_kf.inv().apply(raw[:, :3] - raw[-1, :3]))   # (T, 3)

        groups = defaultdict(list)
        for i, n in enumerate(seg_idx):
            groups[n].append(i)

        interval_len = {}
        for n, members in groups.items():
            max_k = min(len(local_pos[i]) for i in members) - 1
            l_var = 0
            for k in range(1, max_k + 1):
                pts = np.stack([local_pos[i][-1 - k] for i in members], axis=0)  # (M, 3)
                if pts.var(axis=0).sum() <= thr:
                    l_var = k
                else:
                    break
            interval_len[n] = max(min_len, l_var)

        max_target = 1
        seg_lens = []
        for i, n in enumerate(seg_idx):
            T = len(demos[i][key])
            vstart = max(0, (T - 1) - interval_len[n])
            # stored full length so get_observation's per-key indexing stays valid
            demos[i]['variance_start'] = np.full((T,), vstart, dtype=np.int64)
            target = (T - 1) - vstart + 1
            seg_lens.append(T)
            max_target = max(max_target, target)
        print(f"[BIP] {len(demos)} segments; interval_len per seg-index="
              f"{ {k: interval_len[k] for k in sorted(interval_len)} }; "
              f"seg_len[min/mean/max]={min(seg_lens)}/{int(np.mean(seg_lens))}/{max(seg_lens)}; "
              f"CoA action_sequence={max_target}")
        return max_target


    def _convert_demos_to_loaded_format(self, 
        raw_demos, cfg=None
    ) -> List[List[DemoStep]]:
        """Converts demos generated in rlbench to the common DemoStep format.

        Args:
            raw_demos: raw demos generated with rlbench.

        Returns:
            List[List[DemoStep]]: demos converted to DemoSteps ready for
                augmentation and loading. First remove unuseful keys,
                e.g. unuseful states and value of None. Then convert observations
                from timestep item as list element to keys as list element.
        """

        # remove unuseful keys and normalize low dim state
        converted_demos = []
        for demo in raw_demos:
            converted_demo = []
            for timestep in demo:
                converted_demo.append(
                    DemoStep(
                        timestep.joint_positions,
                        timestep.gripper_open,
                        _extract_obs(timestep, cfg=cfg, action_space=self.get_action_space(cfg), training=self.training),
                        timestep.gripper_matrix,
                        timestep.misc,
                    )
                )
            converted_demos.append(converted_demo)
 
        # convert raw gripper matrix and gripper state to action_abs_joint and action_abs_ee
        # add descriptions 
        for demo in converted_demos:
            for i in range(len(demo)):
                demo_step = demo[i]
                # demo_step[ActionModeType.ABS_JOINT_POSITION] = observations_to_action_abs_joint(demo_step) # TODO: To be implemented
                demo_step[ActionModeType.ABS_END_EFFECTOR_POSE.value] = observations_to_action_abs_ee(demo_step)
                if "descriptions" in demo_step.misc and cfg.method.use_lang_cond:
                    descriptions = demo_step.misc["descriptions"][0]
                    # desc = descriptions[np.random.randint(len(descriptions))]
                    demo_step['desc'] = clip.tokenize(descriptions)

        # Normalize actions for the two main action types
        self._normalize_demo_actions(converted_demos, cfg)

        # convert converted_demos. Original each demo is oranlized as list of frame dict
        # keys contain obs and action, reward, terminal, truncated, info....
        # now each demo is a dict of this items stacked over all frames 
        demos_trans = []
        for demo in converted_demos:
            demo = {key: np.stack([d[key] for d in demo], axis=0) for key in demo[0]}
            demos_trans.append(demo)
      
        return demos_trans

    def _normalize_demo_actions(self, converted_demos, cfg):
        """简洁的action normalization实现"""
        action_space = self.get_action_space(cfg)
        for demo in converted_demos:
            for demo_step in demo:
                # TODO: To be implemented
                # Normalize ABS_JOINT_POSITION actions
                # if ActionModeType.ABS_JOINT_POSITION in demo_step and demo_step[ActionModeType.ABS_JOINT_POSITION] is not None:
                #     demo_step[ActionModeType.ABS_JOINT_POSITION] = MinMaxNorm.normalize(
                #         demo_step[ActionModeType.ABS_JOINT_POSITION], action_space
                #     )
                
                # Normalize ABS_END_EFFECTOR_POSE actions  
                if ActionModeType.ABS_END_EFFECTOR_POSE.value in demo_step and demo_step[ActionModeType.ABS_END_EFFECTOR_POSE.value] is not None:
                    demo_step[ActionModeType.ABS_END_EFFECTOR_POSE.value] = MinMaxNorm.normalize(
                        demo_step[ActionModeType.ABS_END_EFFECTOR_POSE.value], action_space
                    )


    def get_action_stats(self):
        return self._action_stats

        

    def _compute_action_stats(self, demos: List[List[DemoStep]]):
        """Compute statistics from demonstration actions, which could be useful for
        users that want to set action space based on demo action statistics.

        Args:
            demos: list of demo episodes

        Returns:
            Dict[str, np.ndarray]: a dictionary of numpy arrays that contain action
            statistics (i.e., mean, std, max, and min)
        """
        actions = []
        for demo in demos:
            for step in demo:
                *_, info = step
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)

        # Gripper one-hot action's stats are hard-coded
        action_mean = np.hstack([np.mean(actions, 0)[:-1], 1 / 2])
        action_std = np.hstack([np.std(actions, 0)[:-1], 1 / 6])
        action_max = np.hstack([np.max(actions, 0)[:-1], 1])
        action_min = np.hstack([np.min(actions, 0)[:-1], 0])
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats


def observations_to_action(
    ActionModeType: ActionModeType,
    observation: DemoStep,
    action_space: Box,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """
    action = np.concatenate(
        [
            (
                current_observation.misc["joint_position_action"][:-1]
                if "joint_position_action" in current_observation.misc
                else current_observation.joint_positions
            ),
            [1.0 if current_observation.gripper_open == 1 else 0.0],
        ]
    ).astype(np.float32)
    return action

def observations_to_action_with_onehot_gripper_abs_ee(
    current_observation: DemoStep,
    # next_observation: DemoStep,
    action_space: Box = None,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """

    action_trans = current_observation.gripper_matrix[:3, 3]

    rot = R.from_matrix(current_observation.gripper_matrix[:3, :3])
    action_orien = rot.as_quat(
        canonical=True
    )  # Enforces w component always positive and unit vector

    action_gripper = [1.0 if current_observation.gripper_open == 1 else 0.0]
    action = np.concatenate(
        [
            action_trans,
            action_orien,
            action_gripper,
        ]
    )
    if action_space is not None:
        if np.any(action[:-1] > action_space.high[:-1]) or np.any(
            action[:-1] < action_space.low[:-1]
        ):
            warnings.warn(
                "Action outside action space.",
                UserWarning,
            )
            return None
    return action


def _make_env(cfg: DictConfig, obs_config: dict):
    # NOTE: Completely random initialization
    # TODO: Can we make this deterministic based on cfg.seed?
    task_name = cfg.env.task_name
    action_mode = _get_action_mode(ActionModeType[cfg.env.action_mode])

    return RLBenchEnv(
        task_name,
        obs_config,
        action_mode,
        action_mode_type=ActionModeType[cfg.env.action_mode],
        arm_max_velocity=cfg.env.arm_max_velocity,
        arm_max_acceleration=cfg.env.arm_max_acceleration,
        dataset_root_train=cfg.dataset_root_train,
        dataset_root_eval=cfg.dataset_root_eval,
        render_mode="rgb_array",
        use_lang_cond=cfg.method.get("use_lang_cond", False),
        cfg=cfg,
    )
    

    
def observations_to_action_abs_joint(
    current_observation: DemoStep,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """
    action = np.concatenate(
        [
            (
                current_observation.misc["joint_position_action"][:-1]
                if "joint_position_action" in current_observation.misc
                else current_observation.joint_positions
            ),
            [1.0 if current_observation.gripper_open == 1 else 0.0],
        ]
    ).astype(np.float32)

    action_space = ACTION_BOUNDS[ActionModeType.ABS_JOINT_POSITION]

    if np.any(action[:-1] > action_space[1][:-1]) or np.any(
        action[:-1] < action_space[0][:-1]
    ):
        warnings.warn(
            "Action outside action space.",
            UserWarning,
        )
        return None
    
    return action

def observations_to_action_abs_ee(
    current_observation: DemoStep,

):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """

    action_trans = current_observation.gripper_matrix[:3, 3]

    rot = R.from_matrix(current_observation.gripper_matrix[:3, :3])
    action_orien = rot.as_quat(
        canonical=True
    )  # Enforces w component always positive and unit vector

    action_gripper = [1.0 if current_observation.gripper_open == 1 else 0.0]
    action = np.concatenate(
        [
            action_trans,
            action_orien,
            action_gripper,
        ]
    )

    action_space = ACTION_BOUNDS[ActionModeType.ABS_END_EFFECTOR_POSE]

    # TODO: not know if this is needed
    # if np.any(action[:-1] > action_space[1][:-1]) or np.any(
    #     action[:-1] < action_space[0][:-1]
    # ):
    #     warnings.warn(
    #         "Action outside action space.",
    #         UserWarning,
    #     )
    #     return None
    
    return action


def _extract_obs(obs: Observation, observation_config=None, robot_state_keys=None, training=True, cfg=None, action_space=None):
    '''
    Extract the necessary observation from offline/online raw data from rlbench env.
    Filter to keep only essential data: RGB cameras (based on config) and low_dim_state.
    Optionally filter out point cloud data when training.
    '''
    
    # Construct low-dimensional state data (joint positions + gripper state)
    assert ActionModeType[cfg.env.action_mode] in [ActionModeType.ABS_END_EFFECTOR_POSE, ActionModeType.ABS_JOINT_POSITION], "Only ABS_END_EFFECTOR_POSE and ABS_JOINT_POSITION are supported now. Low dim state norm always use abs action space. It it gets wrong with DEL_END_EFFECTOR_POSE and DEL_JOINT_POSITION for current implementation."
    if ActionModeType[cfg.env.action_mode] == ActionModeType.ABS_END_EFFECTOR_POSE or ActionModeType[cfg.env.action_mode] == ActionModeType.DEL_END_EFFECTOR_POSE:
        low_dim_state = np.concatenate([obs.gripper_pose,[obs.gripper_open]],dtype=np.float32)
        low_dim_state = MinMaxNorm.normalize(low_dim_state, action_space)
    elif ActionModeType[cfg.env.action_mode] == ActionModeType.ABS_JOINT_POSITION or ActionModeType[cfg.env.action_mode] == ActionModeType.DEL_JOINT_POSITION:
        low_dim_state = np.concatenate([obs.joint_positions,[obs.gripper_open]],dtype=np.float32)
        low_dim_state = RescaleFromTanh.transform_to_tanh(low_dim_state, action_space)
    else:
        raise ValueError(f"Unsupported action mode type: {ActionModeType[cfg.env.action_mode]}")
    
    # Get all observation data
    obs_dict = vars(obs)
    
    # Filter data: keep only RGB camera data and low-dimensional state
    filtered_obs_dict = {"low_dim_state": low_dim_state}
    
    # Dynamically determine which cameras to use based on config
    if cfg is not None and hasattr(cfg.env, 'cameras'):
        enabled_cameras = cfg.env.cameras
    else:
        # By default, use all available cameras
        enabled_cameras = ["front", "wrist", "left_shoulder", "right_shoulder", "overhead"]
    
    # Add specified RGB camera data, filter out depth, mask, point_cloud, etc.
    for k, v in obs_dict.items():
        if v is not None and k not in ROBOT_STATE_KEYS:
            # Check if the camera name is in the enabled_cameras list from config
            if '_rgb' in k:
                camera_name = k.replace('_rgb', '')
                if camera_name in enabled_cameras:
                    filtered_obs_dict[k] = v
            # Filtering logic for point cloud, depth, mask, etc.
            elif (not training) and (('_point_cloud' in k) or ('_depth' in k) or ('_mask' in k)):
                # Only keep point cloud, depth, mask, etc. during eval/vis
                filtered_obs_dict[k] = v
            # During training, filter out all point cloud, depth, mask
            # Other types of data can be added as needed
    
    # Data format conversion: convert images from (H,W,C) to (C,H,W), add batch dimension to low_dim_state
    final_obs_dict = {}
    for k, v in filtered_obs_dict.items():
        if hasattr(v, 'ndim'):
            if v.ndim == 3:  # RGB image (H,W,C) -> (C,H,W)
                final_obs_dict[k] = v.transpose((2, 0, 1))
            else:
                final_obs_dict[k] = v
        else:
            final_obs_dict[k] = v
    
    return final_obs_dict
    


import hydra
@hydra.main(
    config_path="../../cfgs", config_name="launch", version_base=None
)
def main(cfg):
    a = RLBenchEnvFactory()._load_demos(cfg)
    pass

if __name__ == "__main__":
    main()

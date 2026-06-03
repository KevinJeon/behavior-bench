# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from PufferDrive V2.0
# (https://github.com/Emerge-Lab/PufferDrive/)
# Copyright (c) 2026 PufferDrive, licensed under the MIT license.

import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pathlib import Path
from pufferlib.ocean.drive import binding
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class Drive(pufferlib.PufferEnv):
    # Ordered list of reward components written into `self.reward_components`
    # per-step by the C env. Index order must match the RC_* enum in drive.h.
    REWARD_COMPONENT_NAMES = (
        "collision", "offroad", "goal", "jerk_legacy", "velocity", "comfort",
        "l_align", "l_center", "timestep", "reverse", "speed_limit",
    )

    def __init__(
        self,
        data_root=None,
        split="training",
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_speed_limit=-0.1,
        reward_lane_alignment=0.0,
        reward_lane_distance=0.0,
        reward_velocity=0.0,
        reward_comfort=0.0,
        reward_l_align=0.0,
        reward_l_align_vel=0.5,
        reward_l_center=0.0,
        reward_l_center_bias=0.0,
        reward_reverse=0.0,
        reward_jerk_legacy=0.0,
        reward_conditioning=False,
        reward_timestep=0.0,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        goal_behavior=0,
        goal_target_distance=10.0,
        goal_lane_change_prob=0.01,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=None,
        termination_mode=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        idm_others=False,
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        #map_dir="resources/drive/binaries/training",
        use_all_maps=False,
        map_id=-1,
        include_global_state=False,
        placeholder_agents=0,
        max_obs_partners=None,
        ini_file=None,
        collision_shrink=None,  # Read from ini file in C, accepted here to avoid kwarg error
        mix_traffic=False,      # Enable mixed traffic (PPO/IDM/Expert per agent)
        ppo_fraction=1.0,       # Fraction of agents controlled by PPO policy
        idm_fraction=0.0,       # Fraction of agents using IDM controller
        expert_fraction=0.0,    # Fraction of agents following expert replay
        idm_target_velocity=15.0,  # Default IDM target velocity [m/s]
        idm_random_velocity=False,  # If True, sample IDM velocity from {10,15,20,30} per agent
        creward_deterministic=False,  # Eval-only: use fixed ego/traffic creward profiles
        emit_jerk_ego_obs=False,  # Force 10-dim jerk ego obs layout even in classic dynamics
        ego_entity_idx=-1,   # Eval-only: entity index of the ego for creward_ego dispatch.
                             # -1 falls back to active_agent_indices[human_agent_idx].
        creward_ego=None,    # dict with keys delta_goal, alpha_collision, alpha_boundary,
        creward_traffic=None,  # alpha_comfort, alpha_l_align, alpha_vel_align,
                               # alpha_l_center, alpha_center_bias, alpha_reverse.
                               # For traffic: either a dict (single profile) or
                               # a list of dicts/tuples (multiple profiles,
                               # cycled per-agent by entity index).
        pbt_mode="none", # none, reactive, replay, save-population
        ego_ratio=0.0,
        population_path=None,
    ):
        # env
        self.dt = dt
        self.map_id = map_id
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_speed_limit = reward_speed_limit
        self.reward_lane_alignment = reward_lane_alignment
        self.reward_lane_distance = reward_lane_distance
        self.reward_velocity = reward_velocity
        self.reward_comfort = reward_comfort
        self.reward_l_align = reward_l_align
        self.reward_l_align_vel = reward_l_align_vel
        self.reward_l_center = reward_l_center
        self.reward_l_center_bias = reward_l_center_bias
        self.reward_reverse = reward_reverse
        self.reward_jerk_legacy = reward_jerk_legacy
        self.reward_conditioning = int(bool(reward_conditioning)) if not isinstance(reward_conditioning, str) else int(reward_conditioning.lower() in ("true", "1", "yes"))
        self.reward_timestep = reward_timestep
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.goal_lane_change_prob = float(goal_lane_change_prob)
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        self.idm_others = bool(idm_others)
        self.idm_target_velocity = float(idm_target_velocity)
        self.idm_random_velocity = bool(idm_random_velocity)
        self.placeholder_agents = int(placeholder_agents)
        self.max_obs_partners = int(max_obs_partners) if max_obs_partners is not None else binding.MAX_OBS_PARTNERS
        self.ini_file = ini_file or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "config", "ocean", "drive.ini"
        )
        # Traffic mix config
        self.mix_traffic = bool(mix_traffic) if not isinstance(mix_traffic, str) else mix_traffic.lower() in ("true", "1", "yes")
        self.ppo_fraction = float(ppo_fraction)
        self.idm_fraction = float(idm_fraction)
        self.expert_fraction = float(expert_fraction)

        # Deterministic creward (eval): when enabled, C copies creward_ego into the
        # ego agent and cycles through creward_traffic[] by entity index for
        # every other agent, instead of random sampling.
        self.creward_deterministic = bool(creward_deterministic) if not isinstance(creward_deterministic, str) else creward_deterministic.lower() in ("true", "1", "yes")
        self.emit_jerk_ego_obs = bool(emit_jerk_ego_obs) if not isinstance(emit_jerk_ego_obs, str) else emit_jerk_ego_obs.lower() in ("true", "1", "yes")
        self.ego_entity_idx = int(ego_entity_idx)
        self.creward_ego = dict(creward_ego or {})
        # Normalize traffic to a list-of-dicts. Accept: None, single dict,
        # list of dicts, or list of 9-tuples in the documented field order.
        self.creward_traffic = self._normalize_creward_traffic(creward_traffic)

        # TODO: PBT options
        self.pbt_mode = pbt_mode
        self.ego_ratio = float(ego_ratio)
        self.population_path = population_path
        self._pbt_generation = 0
        # When idm_others is enabled, only ego (1 agent per map) is PPO-controlled
        if self.idm_others and max_controlled_agents < 0:
            max_controlled_agents = 1
        self.seed = seed
        # Observation space calculation
        if dynamics_model == "jerk" or self.emit_jerk_ego_obs:
            self.ego_features = binding.EGO_FEATURES_JERK
        else:
            self.ego_features = binding.EGO_FEATURES_CLASSIC

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = self.max_obs_partners
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES

        self.include_global_state = bool(include_global_state)
        self.global_state_features = 7 if self.include_global_state else 0  # x, y, hx, hy, vx, vy, map_id
        self.creward_features = binding.CREWARD_FEATURES if self.reward_conditioning else 0
        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
            + self.creward_features
            + self.global_state_features
        )
        self.single_observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
        
        if data_root == None:
            if 'DRIVE_BINARIES_DATA_ROOT' in os.environ:
                data_root = os.environ['DRIVE_BINARIES_DATA_ROOT']
            else:
                raise ValueError(f"data_root is not specified and environment variable DRIVE_BINARIES_DATA_ROOT is not set")
        # print(f"{data_root} is used as data_root")
        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        #self.map_dir = map_dir

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        elif self.control_mode_str == "control_evaluation":
            self.control_mode = 4
        elif self.control_mode_str == "control_pbt":
            self.control_mode = 5
        else:
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_wosac', 'control_evaluation', 'control_pbt', or 'control_agents'. Got: {self.control_mode_str}"
            )
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        
        self.data_root = data_root

        valid_splits = ["training", "validation", "testing", "pufferhard", "testing_with_connectivity", "validation_interactive",
                        "pufferhard_val_100", "pufferhard_val_1k", "pufferhard_val_10k",
                        "pufferandom_val_100", "pufferandom_val_1k", "pufferandom_val_10k",
                        "pufferinter", "pufferrandom", "nuplan_test14"]
        assert split in valid_splits, f"Passed parameter 'split' must be one of {valid_splits}. Got: {split}"

        self.split = split

        
        if action_type == "discrete":
            if dynamics_model == "classic":
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists and contains .bin files
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory {split_dir} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )
        bin_files = [f for f in os.listdir(split_dir) if f.endswith('.bin')]
        if len(bin_files) == 0:
            raise FileNotFoundError(
                f"No .bin files found in {split_dir}. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        # available_maps = len([name for name in os.listdir("resources/drive/binaries") if name.endswith(".bin")])
        available_maps = len([name for name in os.listdir(os.path.join(data_root, split)) if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )
        self.max_controlled_agents = int(max_controlled_agents)

        # When traffic mix is active, my_shared needs fewer maps because each map
        # produces fewer PPO agents. Scale the target for map packing, but keep
        # num_agents unchanged for PufferLib buffer allocation.
        shared_num_agents = num_agents
        if self.mix_traffic and self.ppo_fraction < 1.0:
            shared_num_agents = max(1, int(num_agents * self.ppo_fraction))

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            #map_dir=map_dir,
            num_agents=shared_num_agents,
            num_maps=num_maps,
            split=split,
            data_root=data_root,
            init_mode=self.init_mode,
            ego_ratio=ego_ratio, #todo: 추가해야함
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            goal_lane_change_prob=self.goal_lane_change_prob,
            use_all_maps=use_all_maps,
            map_id=self.map_id,
            **self._extra_c_kwargs(),
        )
        # agent_offsets[-1] = actual PPO agent count from my_shared.
        # With traffic mix, this is less than num_agents (scaled by ppo_fraction).
        self.num_agents = agent_offsets[-1] if (use_all_maps or self.map_id >= 0 or self.mix_traffic) else num_agents
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        # Provide safe defaults so replay/pbt code paths can attach metadata
        # even when control_mode is not control_pbt.
        self.ego_indices = np.empty(0, dtype=np.int64)
        self.other_indices = np.arange(self.num_agents, dtype=np.int64)
        # TODO: PBT indices sampling
        if self.control_mode_str == "control_pbt":
            self._sample_pbt_indices()
        if self.pbt_mode == "replay":
            if self.population_path is None:
                raise FileNotFoundError("pbt_mode=replay requires pbt.population_path")
            replay_dir = os.path.join(self.population_path, "replay")
            actions_npy = os.path.join(replay_dir, "other_actions.npy")
            offsets_npy = os.path.join(replay_dir, "agent_offsets.npy")
            map_ids_npy = os.path.join(replay_dir, "map_ids.npy")
            replay_npz = os.path.join(replay_dir, "other_actions.npz")

            if all(os.path.exists(p) for p in (actions_npy, offsets_npy, map_ids_npy)):
                self.other_actions = np.load(actions_npy, allow_pickle=True)
                self.actions_agent_offsets = np.load(offsets_npy, allow_pickle=True)
                self.actions_map_id = np.load(map_ids_npy, allow_pickle=True)
            elif os.path.exists(replay_npz):
                npz = np.load(replay_npz, allow_pickle=True)
                self.other_actions = npz["actions"]
                self.actions_agent_offsets = npz["agent_offsets"]
                self.actions_map_id = npz["map_ids"]
                del npz
            else:
                raise FileNotFoundError(
                    f"Replay data not found in {replay_dir}. "
                    "Expected other_actions.npy + agent_offsets.npy + map_ids.npy "
                    "or legacy other_actions.npz"
                )
            self._allocate_replay(self.num_agents, self.map_ids)
        super().__init__(buf=buf)
        # Per-step reward breakdown buffer (one row per agent). C env writes
        # each component's contribution to the slot RC_* (see drive.h);
        # evaluator reads `reward_components[ego_agent_idx]` per step.
        self.reward_components = np.zeros(
            (self.num_agents, len(self.REWARD_COMPONENT_NAMES)), dtype=np.float32
        )
        # Parallel α-less buffer: same shape, but C writes the pre-coefficient
        # "behavior term" (e.g. violation count, seconds reversing, etc.).
        # Lets cross-policy comparison on behavior rather than reward magnitude.
        self.reward_components_raw = np.zeros(
            (self.num_agents, len(self.REWARD_COMPONENT_NAMES)), dtype=np.float32
        )
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            ego_local_indices = []
            if self.control_mode_str == "control_pbt":
                ego_local_indices = (
                    self.ego_indices[
                        (self.ego_indices >= cur) & (self.ego_indices < nxt)
                    ] - cur
                ).astype(np.int32).tolist()
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_speed_limit=self.reward_speed_limit,
                reward_lane_alignment=self.reward_lane_alignment,
                reward_lane_distance=self.reward_lane_distance,
                reward_velocity=self.reward_velocity,
                reward_comfort=self.reward_comfort,
                reward_l_align=self.reward_l_align,
                reward_l_align_vel=self.reward_l_align_vel,
                reward_l_center=self.reward_l_center,
                reward_l_center_bias=self.reward_l_center_bias,
                reward_reverse=self.reward_reverse,
                reward_jerk_legacy=self.reward_jerk_legacy,
                reward_conditioning=self.reward_conditioning,
                reward_timestep=self.reward_timestep,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_lane_change_prob=self.goal_lane_change_prob,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                episode_length=(int(episode_length) if episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                idm_others=int(self.idm_others),
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file,
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,

                data_root=self.data_root,
                split=split,
                include_global_state=int(self.include_global_state),
                placeholder_agents=self.placeholder_agents,
                max_obs_partners=self.max_obs_partners,
                reward_components=self.reward_components[cur:nxt],
                reward_components_raw=self.reward_components_raw[cur:nxt],
                # TODO: PBT
                ego_local_indices=ego_local_indices,
                **self._extra_c_kwargs(),
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        info = {}
        if self.control_mode_str == "control_pbt":
            info["ego_indices"] = self.ego_indices
            info["other_indices"] = self.other_indices

        return self.observations, [info]

    def _sample_pbt_indices(self):
        rng = np.random.default_rng(self.seed + self._pbt_generation)
        ego_indices = []

        for start, end in zip(self.agent_offsets[:-1], self.agent_offsets[1:]):
            num_agents_for_map = end - start
            if num_agents_for_map <= 0:
                continue

            num_ego = int(np.floor(num_agents_for_map * self.ego_ratio + 0.5))
            num_ego = min(num_ego, num_agents_for_map)
            if num_ego <= 0:
                continue

            selected = rng.choice(
                np.arange(start, end, dtype=np.int64),
                size=num_ego,
                replace=False,
            )
            ego_indices.extend(selected.tolist())

        self.ego_indices = np.asarray(sorted(ego_indices), dtype=np.int64)

        other_mask = np.ones(self.num_agents, dtype=bool)
        other_mask[self.ego_indices] = False
        self.other_indices = np.flatnonzero(other_mask).astype(np.int64)

        self._pbt_generation += 1

    def _allocate_replay(self, num_agents, map_ids):
        self.replay_actions = np.zeros((num_agents, self.resample_frequency, 1), dtype=np.int32)
        agent_ind = 0
        for _, map_id in enumerate(map_ids):
            num_rollout = self.other_actions.shape[0]
            sample_ind = np.random.randint(0, num_rollout)
            map_indices = np.where(self.actions_map_id == map_id)[0][0]
            agent_offsets = self.actions_agent_offsets[map_indices:map_indices+2]
            num_agents_for_map = agent_offsets[1] - agent_offsets[0]
            if agent_ind + num_agents_for_map> num_agents:
                num_agents_for_map = num_agents - agent_ind
            self.replay_actions[agent_ind:agent_ind+num_agents_for_map] = self.other_actions[sample_ind, agent_offsets[0]:agent_offsets[0] + num_agents_for_map].copy()
            agent_ind += num_agents_for_map

    def resample_maps(self):
        """Resample environment maps. Closes current envs and creates new ones."""
        self.tick = 0
        binding.vec_close(self.c_envs)
        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            goal_lane_change_prob=self.goal_lane_change_prob,
            goal_speed=self.goal_speed,
            split=self.split,
            data_root=self.data_root,
            use_all_maps=False,
            map_id=self.map_id,
            **self._extra_c_kwargs(),
        )
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        # TODO: pbt
        if self.control_mode_str == "control_pbt":
            self._sample_pbt_indices()
            if self.pbt_mode == "replay":
                self._allocate_replay(self.num_agents, self.map_ids)
        env_ids = []
        seed = np.random.randint(0, 2**32 - 1)
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            ego_local_indices = []
            if self.control_mode_str == "control_pbt":
                ego_local_indices = (
                    self.ego_indices[
                        (self.ego_indices >= cur) & (self.ego_indices < nxt)
                    ] - cur
                ).astype(np.int32).tolist()
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=self.human_agent_idx,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_speed_limit=self.reward_speed_limit,
                reward_lane_alignment=self.reward_lane_alignment,
                reward_lane_distance=self.reward_lane_distance,
                reward_velocity=self.reward_velocity,
                reward_comfort=self.reward_comfort,
                reward_l_align=self.reward_l_align,
                reward_l_align_vel=self.reward_l_align_vel,
                reward_l_center=self.reward_l_center,
                reward_l_center_bias=self.reward_l_center_bias,
                reward_reverse=self.reward_reverse,
                reward_jerk_legacy=self.reward_jerk_legacy,
                reward_conditioning=self.reward_conditioning,
                reward_timestep=self.reward_timestep,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                goal_radius=self.goal_radius,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_lane_change_prob=self.goal_lane_change_prob,
                goal_speed=self.goal_speed,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=self.dt,
                episode_length=(int(self.episode_length) if self.episode_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                idm_others=int(self.idm_others),
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file,
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                data_root=self.data_root,
                split=self.split,
                include_global_state=int(self.include_global_state),
                placeholder_agents=self.placeholder_agents,
                max_obs_partners=self.max_obs_partners,
                reward_components=self.reward_components[cur:nxt],
                reward_components_raw=self.reward_components_raw[cur:nxt],
                # TODO: PBT
                ego_local_indices=ego_local_indices,
                **self._extra_c_kwargs(),
            )
            env_ids.append(env_id)
        self.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.c_envs, seed)
        self.terminals[:] = 1

    def step(self, actions):

        self.terminals[:] = 0
        self.truncations[:] = 0
        self.actions[:] = actions
        if self.pbt_mode == "replay" and hasattr(self, "replay_actions"):
            replay_tick = self.tick % self.replay_actions.shape[1]
            self.actions[self.other_indices] = self.replay_actions[self.other_indices, replay_tick, :]
        # reset environment, if resample_frequency is reached, you do not need to step in this case!
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.resample_maps()
            info = []

        else:
            binding.vec_step(self.c_envs) # truncations and terminals get set in here as well.
            self.tick += 1
            info = []
            if self.tick % self.report_interval == 0:
                log = binding.vec_log(self.c_envs, self.num_agents)
                if log:
                    info.append(log)
                    # print(log)
            if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0: # self.tick just got increased!
                self.truncations[:] = 1.0 # truncations are the ones which are after time out right??
        if self.pbt_mode != "none":
            if len(info) == 0:
                info = [{"agent_offsets": self.agent_offsets, "map_ids": self.map_ids, "num_envs": self.num_envs, "ego_indices": self.ego_indices}]
            else:
                info[0]["agent_offsets"] = self.agent_offsets
                info[0]["map_ids"] = self.map_ids
                info[0]["num_envs"] = self.num_envs
                info[0]["ego_indices"] = self.ego_indices
        if self.pbt_mode == "reactive":
            info[0]["other_indices"] = self.other_indices 

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    # User-facing creward fields. `delta_goal` defaults to 0 (sentinel meaning
    # "use env->goal_radius"); users can override for sweep experiments.
    _CREWARD_FIELDS = (
        "delta_goal",
        "alpha_collision", "alpha_boundary", "alpha_comfort",
        "alpha_l_align", "alpha_vel_align", "alpha_l_center",
        "alpha_center_bias", "alpha_reverse",
        "goal_speed",
    )

    @classmethod
    def _normalize_creward_traffic(cls, value):
        """Turn the user-provided creward_traffic value into a list of dicts.

        Accepts: None, {}, single dict, list/tuple of dicts, or list/tuple of
        tuples in the order listed in _CREWARD_FIELDS. Accepted tuple lengths:
          - full (len == CREWARD_FEATURES): delta_goal, ..., alpha_reverse, goal_speed
          - short (full - 1): delta_goal skipped, so
            (alpha_collision, ..., alpha_reverse, goal_speed). delta_goal=0
            (sentinel -> env->goal_radius) is prepended.
        """
        if value is None:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, (list, tuple)):
            out = []
            full = len(cls._CREWARD_FIELDS)
            short = full - 1  # traffic form without delta_goal
            for item in value:
                if isinstance(item, dict):
                    out.append(dict(item))
                elif isinstance(item, (list, tuple)):
                    if len(item) == full:
                        pairs = zip(cls._CREWARD_FIELDS, item)
                    elif len(item) == short:
                        pairs = zip(cls._CREWARD_FIELDS, (0.0, *item))
                    else:
                        raise ValueError(
                            f"creward_traffic tuple must have {full} or {short} "
                            f"entries (order: {cls._CREWARD_FIELDS}), got {len(item)}")
                    out.append({k: float(v) for k, v in pairs})
                else:
                    raise TypeError(
                        f"creward_traffic entries must be dict or tuple, got {type(item)}")
            return out
        raise TypeError(f"creward_traffic must be None, dict, or list, got {type(value)}")

    def _extra_c_kwargs(self):
        """Return extra kwargs for binding.env_init() / binding.shared()."""
        kwargs = {}
        if self.mix_traffic:
            kwargs['traffic_mix_ppo'] = self.ppo_fraction
            kwargs['traffic_mix_idm'] = self.idm_fraction
            kwargs['traffic_mix_expert'] = self.expert_fraction
            kwargs['idm_random_velocity'] = int(self.idm_random_velocity)
            kwargs['idm_default_velocity'] = self.idm_target_velocity
        if self.emit_jerk_ego_obs:
            kwargs['emit_jerk_ego_obs'] = 1
        if self.creward_deterministic:
            kwargs['creward_deterministic'] = 1
            if self.ego_entity_idx >= 0:
                kwargs['ego_entity_idx'] = self.ego_entity_idx
            for field in self._CREWARD_FIELDS:
                kwargs[f'creward_ego_{field}'] = float(self.creward_ego.get(field, 0.0))
            profiles = self.creward_traffic or [{}]
            kwargs['creward_traffic_count'] = len(profiles)
            for i, profile in enumerate(profiles):
                for field in self._CREWARD_FIELDS:
                    kwargs[f'creward_traffic_{i}_{field}'] = float(profile.get(field, 0.0))
        # dynamics_model + action_type overrides for the C env (the INI is
        # otherwise authoritative). Sent unconditionally so the C env always
        # agrees with the Python-side choice without requiring INI edits.
        kwargs['dynamics_model_override'] = 1 if self.dynamics_model == "jerk" else 0
        kwargs['action_type_override'] = 0 if self._action_type_flag == 0 else 1
        return kwargs

    def get_state(self):
        try:
            return binding.vec_get(self.c_envs)
        except Exception:
            return binding.env_get(self.c_envs)

    def get_episode_log(self):
        """Get episode logs without waiting for num_agents threshold.

        Returns:
            dict with episode metrics, or empty dict if no data available
        """
        log = binding.vec_log(self.c_envs, 1)  # Use threshold of 1 to get logs immediately
        return log if log else {}

    def get_agent_log(self, agent_idx=0):
        """Get per-agent logs for a specific agent.

        Args:
            agent_idx: Index of the agent (default: 0 for ego agent)

        Returns:
            dict with agent-specific metrics
        """
        return binding.vec_get_agent_log(self.c_envs, agent_idx)

    def create_snapshot(self):
        """Create a snapshot of the current simulator state.

        Returns:
            A list of snapshot handles that can be used to restore the state later.
        """
        return binding.vec_create_snapshot(self.c_envs)

    def restore_snapshot(self, snapshot):
        """Restore the simulator to a previously saved state.

        Args:
            snapshot: A list of snapshot handles returned by create_snapshot()
        """
        binding.vec_restore_snapshot(self.c_envs, snapshot)

    def free_snapshot(self, snapshot):
        """Free memory associated with a snapshot.

        Args:
            snapshot: A list of snapshot handles to free
        """
        binding.vec_free_snapshot(snapshot)

    def set_goals(self, agent_indices, goal_xs, goal_ys):
        """Set goal positions for specific agents (world coordinates)."""
        binding.vec_set_goals(self.c_envs, list(agent_indices), list(goal_xs), list(goal_ys))

    def set_positions(self, agent_indices, xs, ys, headings):
        """Set positions and headings for specific agents (world coordinates)."""
        binding.vec_set_positions(self.c_envs, list(agent_indices), list(xs), list(ys), list(headings))

    def get_global_agent_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id', 'length', 'width' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "length": np.zeros(num_agents, dtype=np.float32),
            "width": np.zeros(num_agents, dtype=np.float32),
            "type": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["z"],
            states["heading"],
            states["id"],
            states["length"],
            states["width"],
            states["type"],
        )

        return states

    def get_ground_truth_trajectories(self):
        """Get ground truth trajectories for all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'valid', 'id', 'scenario_id' containing numpy arrays.
        """
        num_agents = self.num_agents

        trajectories = {
            "x": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.int32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "scenario_id": np.zeros(num_agents, dtype=np.int32),
            "is_vehicle": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["scenario_id"],
            trajectories["is_vehicle"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        return trajectories

    def get_road_edge_polylines(self):
        """Get road edge polylines for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'scenario_id' containing numpy arrays.
            x, y are flattened point coordinates; lengths indicates points per polyline.
        """
        num_polylines, total_points = binding.vec_get_road_edge_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        return polylines

    def get_all_road_polylines(self):
        """Get ALL road polylines (lanes, lines, edges) for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'types', 'scenario_id'.
            x, y are flattened point coordinates; lengths indicates points per polyline.
            types: ROAD_LANE=4, ROAD_LINE=5, ROAD_EDGE=6.
        """
        num_polylines, total_points = binding.vec_get_all_road_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "types": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_all_road_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["types"],
            polylines["scenario_id"],
        )

        return polylines

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if area < polyline_reduction_threshold and dist(point1, point3) <= max_segment_length:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file, unique_map_id, trajectory_length=91):
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        # Get metadata
        metadata = map_data.get("metadata", {})
        sdc_track_index = metadata.get("sdc_track_index", -1)  # -1 as default if not found
        tracks_to_predict = metadata.get("tracks_to_predict", [])

        # Write sdc_track_index
        f.write(struct.pack("i", sdc_track_index))

        # Write tracks_to_predict info (indices only)
        f.write(struct.pack("i", len(tracks_to_predict)))
        for track in tracks_to_predict:
            track_index = track.get("track_index", -1)
            f.write(struct.pack("i", track_index))

        # Count total entities
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write unique map id
            f.write(struct.pack("i", unique_map_id))

            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            f.write(struct.pack("i", obj.get("id", 0)))  # id
            f.write(struct.pack("i", trajectory_length))  # array_size
            # Write position arrays
            positions = obj.get("position", [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("x", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("y", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("z", 0.0))))

            # Write velocity arrays
            velocities = obj.get("velocity", [])
            for arr, key in [(velocities, "x"), (velocities, "y"), (velocities, "z")]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {"x": 0.0, "y": 0.0, "z": 0.0}
                    f.write(struct.pack("f", float(vel.get(key, 0.0))))

            # Write heading and valid arrays
            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
                )
            )

            # Write scalar fields
            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

            # Objects have no exit_lanes (write 0 for format consistency)
            f.write(struct.pack("i", 0))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            f.write(struct.pack("i", unique_map_id))

            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)
            # breakpoint()
            if road_type >= 0 and road_type <= 3:
                road_type = 4
            elif road_type >= 5 and road_type <= 13:
                road_type = 5
            elif road_type >= 14 and road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10
            # Write base entity data
            f.write(struct.pack("i", road_type))  # type
            f.write(struct.pack("i", road.get("id", 0)))  # id
            f.write(struct.pack("i", size))  # array_size

            # Write position arrays
            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            # Write scalar fields
            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))

            # Write exit_lanes connectivity (for lane chaining in IDM)
            exit_lanes = road.get("exit_lanes", [])
            f.write(struct.pack("i", len(exit_lanes)))
            for eid in exit_lanes:
                f.write(struct.pack("i", int(eid)))


def load_map(map_name, unique_map_id, binary_output=None, trajectory_length=91):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output, unique_map_id, trajectory_length=trajectory_length)


def _process_single_map(args):
    """Worker function to process a single map file"""
    if len(args) == 4:
        i, map_path, binary_path, trajectory_length = args
    else:
        i, map_path, binary_path = args
        trajectory_length = 91
    try:
        load_map(str(map_path), i, str(binary_path), trajectory_length=trajectory_length)
        return (i, map_path.name, True, None)
    except Exception as e:
        # Remove partially written binary to avoid corrupt files
        import os
        if os.path.exists(str(binary_path)):
            os.remove(str(binary_path))
        return (i, map_path.name, False, str(e))


def process_all_maps(
    json_dir,
    binary_dir,
    max_maps=10_000,
    num_workers=None,
    trajectory_length=91,
):
    """Process all maps and save them as binaries using multiprocessing

    Args:
        data_folder: Path to the folder containing JSON map files
        max_maps: Maximum number of maps to process
        num_workers: Number of parallel workers (defaults to cpu_count())
    """
    from pathlib import Path

    # Create the binaries directory if it doesn't exist
    # binary_dir = Path("resources/drive/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = cpu_count()

    # # Path to the training data
    # data_dir = Path(data_folder)
    # dataset_name = data_dir.name

    # # Create the binaries directory if it doesn't exist
    # binary_dir = Path(f"resources/drive/binaries/{dataset_name}")
    # binary_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the training directory
    # json_files = sorted(data_dir.glob("*.json"))
    json_files = sorted(json_dir.rglob("*.json"))

    print(f"Found {len(json_files)} json files")

    # Prepare arguments for parallel processing
    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:06d}.bin"  # Use zero-padded numbers for consistent sorting
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path, trajectory_length))

    # Process maps in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(_process_single_map, tasks), total=len(tasks), desc="Processing maps", unit="map")
        )

    # Collect statistics and renumber to fill gaps from failed conversions
    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    if failed > 0:
        print(f"\nFailed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")

        # Renumber remaining files to be sequential (no gaps)
        import os as _os
        existing = sorted(binary_dir.glob("map_*.bin"))
        # Move to temp names first to avoid collisions
        for idx, path in enumerate(existing):
            path.rename(binary_dir / f"_tmp_{idx:06d}.bin")
        # Rename to final sequential names
        tmp_files = sorted(binary_dir.glob("_tmp_*.bin"))
        for idx, path in enumerate(tmp_files):
            path.rename(binary_dir / f"map_{idx:06d}.bin")
        print(f"Renumbered {len(tmp_files)} files to map_000000..map_{len(tmp_files)-1:06d}")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(
        num_agents=num_agents,
        num_maps=1,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        init_steps=0,
        episode_length=91,
    )

    env.reset()

    tick = 0
    actions = np.stack(
        [np.random.randint(0, space.n + 1, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")

    env.close()


if __name__ == "__main__":
    data_root = Path(os.environ['DRIVE_DATA_ROOT'])
    bin_root = Path(os.environ['DRIVE_BINARIES_DATA_ROOT'])
    for split_dir in sorted(data_root.iterdir()):
        if split_dir.is_dir():
            cur_bin_dir = bin_root / split_dir.name
            process_all_maps(split_dir, cur_bin_dir, max_maps=10_000_000)

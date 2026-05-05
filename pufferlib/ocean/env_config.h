/*
 * Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
 * SPDX-License-Identifier: AGPL-3.0
 *
 * This source code is derived from PufferDrive V2.0
 * (https://github.com/Emerge-Lab/PufferDrive/)
 * Copyright (c) 2026 PufferDrive, licensed under the MIT license.
 */

#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Config struct for parsing INI files - contains all environment configuration
typedef struct {
    int action_type;
    int dynamics_model;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_speed_limit;
    float reward_lane_alignment;
    float reward_lane_distance;
    float reward_velocity;
    float reward_comfort;
    float reward_l_align;
    float reward_l_align_vel;
    float reward_l_center;
    float reward_l_center_bias;
    float reward_reverse;
    float reward_jerk_legacy;
    float reward_timestep;
    int reward_conditioning;
    float goal_radius;
    float goal_speed;
    int collision_behavior;
    int offroad_behavior;
    int spawn_immunity_timer;
    float dt;
    int goal_behavior;
    float goal_target_distance;
    int episode_length;
    int termination_mode;
    int init_steps;
    int init_mode;
    int control_mode;
    float collision_shrink;
    //char map_dir[256];
    char split[256];
} env_init_config;

// INI file parser handler - parses all environment configuration from drive.ini
static int handler(void *config, const char *section, const char *name, const char *value) {
    env_init_config *env_config = (env_init_config *)config;
#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        if (strcmp(value, "\"discrete\"") == 0 || strcmp(value, "discrete") == 0) {
            env_config->action_type = 0; // DISCRETE
        } else if (strcmp(value, "\"continuous\"") == 0 || strcmp(value, "continuous") == 0) {
            env_config->action_type = 1; // CONTINUOUS
        } else {
            printf("Warning: Unknown action_type value '%s', defaulting to DISCRETE\n", value);
            env_config->action_type = 0; // Default to DISCRETE
        }
    } else if (MATCH("env", "dynamics_model")) {
        if (strcmp(value, "\"classic\"") == 0 || strcmp(value, "classic") == 0) {
            env_config->dynamics_model = 0; // CLASSIC
        } else if (strcmp(value, "\"jerk\"") == 0 || strcmp(value, "jerk") == 0) {
            env_config->dynamics_model = 1; // JERK
        } else {
            printf("Warning: Unknown dynamics_model value '%s', defaulting to JERK\n", value);
            env_config->dynamics_model = 1; // Default to JERK
        }
    } else if (MATCH("env", "goal_behavior")) {
        env_config->goal_behavior = atoi(value);
    } else if (MATCH("env", "goal_target_distance")) {
        env_config->goal_target_distance = atof(value);
    } else if (MATCH("env", "reward_vehicle_collision")) {
        env_config->reward_vehicle_collision = atof(value);
    } else if (MATCH("env", "reward_offroad_collision")) {
        env_config->reward_offroad_collision = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
    } else if (MATCH("env", "reward_goal_post_respawn")) {
        env_config->reward_goal_post_respawn = atof(value);
    } else if (MATCH("env", "reward_vehicle_collision_post_respawn")) {
        env_config->reward_vehicle_collision_post_respawn = atof(value);
    } else if (MATCH("env", "reward_speed_limit")) {
        env_config->reward_speed_limit = atof(value);
    } else if (MATCH("env", "reward_lane_alignment")) {
        env_config->reward_lane_alignment = atof(value);
    } else if (MATCH("env", "reward_lane_distance")) {
        env_config->reward_lane_distance = atof(value);
    } else if (MATCH("env", "reward_velocity")) {
        env_config->reward_velocity = atof(value);
    } else if (MATCH("env", "reward_comfort")) {
        env_config->reward_comfort = atof(value);
    } else if (MATCH("env", "reward_l_align")) {
        env_config->reward_l_align = atof(value);
    } else if (MATCH("env", "reward_l_align_vel")) {
        env_config->reward_l_align_vel = atof(value);
    } else if (MATCH("env", "reward_l_center")) {
        env_config->reward_l_center = atof(value);
    } else if (MATCH("env", "reward_l_center_bias")) {
        env_config->reward_l_center_bias = atof(value);
    } else if (MATCH("env", "reward_reverse")) {
        env_config->reward_reverse = atof(value);
    } else if (MATCH("env", "reward_jerk_legacy")) {
        env_config->reward_jerk_legacy = atof(value);
    } else if (MATCH("env", "reward_conditioning")) {
        if (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "True") == 0) {
            env_config->reward_conditioning = 1;
        } else {
            env_config->reward_conditioning = 0;
        }
    } else if (MATCH("env", "reward_timestep")) {
        env_config->reward_timestep = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "goal_speed")) {
        env_config->goal_speed = atof(value);
    } else if (MATCH("env", "collision_behavior")) {
        env_config->collision_behavior = atoi(value);
    } else if (MATCH("env", "offroad_behavior")) {
        env_config->offroad_behavior = atoi(value);
    } else if (MATCH("env", "spawn_immunity_timer")) {
        env_config->spawn_immunity_timer = atoi(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "episode_length")) {
        env_config->episode_length = atoi(value);
    } else if (MATCH("env", "termination_mode")) {
        env_config->termination_mode = atoi(value);
    } else if (MATCH("env", "init_steps")) {
        env_config->init_steps = atoi(value);
    } else if (MATCH("env", "init_mode")) {
        env_config->init_mode = atoi(value);
    } else if (MATCH("env", "control_mode")) {
        env_config->control_mode = atoi(value);
    } else if (MATCH("env", "collision_shrink")) {
        env_config->collision_shrink = atof(value);
    } else if (MATCH("env", "split")) {
        if (sscanf(value, "\"%255[^\"]\"", env_config->split) != 1) {
            strncpy(env_config->split, value, sizeof(env_config->split) - 1);
            env_config->split[sizeof(env_config->split) - 1] = '\0';
        }
        //printf("Parsed map_dir: '%s'\n", env_config->map_dir);


    // } else if (MATCH("env", "map_dir")) {
    //     if (sscanf(value, "\"%255[^\"]\"", env_config->map_dir) != 1) {
    //         strncpy(env_config->map_dir, value, sizeof(env_config->map_dir) - 1);
    //         env_config->map_dir[sizeof(env_config->map_dir) - 1] = '\0';
    //     }
    //     //printf("Parsed map_dir: '%s'\n", env_config->map_dir);

    
    } else if (MATCH("env", "traffic_mix") ||
               MATCH("env", "mix_traffic") ||
               MATCH("env", "ppo_fraction") ||
               MATCH("env", "idm_fraction") ||
               MATCH("env", "expert_fraction") ||
               MATCH("env", "idm_target_velocity") ||
               MATCH("env", "idm_random_velocity") ||
               MATCH("env", "idm_others") ||
               MATCH("env", "max_controlled_agents") ||
               MATCH("env", "max_obs_partners") ||
               MATCH("env", "num_agents") ||
               MATCH("env", "num_maps") ||
               MATCH("env", "resample_frequency") ||
               MATCH("env", "use_all_maps") ||
               MATCH("env", "placeholder_agents")) {
        // Python-only params — ignored by C parser
    } else if (strcmp(section, "env") != 0) {
        // Non-env sections (train, eval, etc.) — ignored
    } else {
        return 0;  // Unknown env key
    }

#undef MATCH
    return 1;
}

#endif // ENV_CONFIG_H

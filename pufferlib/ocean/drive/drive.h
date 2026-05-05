/*
 * Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
 * SPDX-License-Identifier: AGPL-3.0
 *
 * This source code is derived from PufferDrive V2.0
 * (https://github.com/Emerge-Lab/PufferDrive/)
 * Copyright (c) 2026 PufferDrive, licensed under the MIT license.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <float.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include "error.h"

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

// Movement modes for controlled agents
#define MOVEMENT_DYNAMICS 0
#define MOVEMENT_IDM 1
#define MOVEMENT_EXPERT 2

#define INVALID_POSITION -10000.0f

// Lane connectivity / route constants
#define MAX_EXIT_LANES 8
#define MAX_ROUTE_POINTS 500

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3
#define CONTROL_EVALUATION 4

// Minimum distance to goal position
#define MIN_DISTANCE_TO_GOAL 2.0f

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define JERK 1

// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Metrics array indices
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACHED_GOAL_IDX 2
#define LANE_ALIGNED_IDX 3
#define LANE_DISTANCE_IDX 4

// Grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL                                                                                          \
    30 // Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in
       // gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2

// Observation constants
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128

// Maximum number of agents per scene
#ifndef MAX_AGENTS
#define MAX_AGENTS 128
#endif
#ifndef MAX_OBS_PARTNERS
#define MAX_OBS_PARTNERS 31
#endif
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define ROAD_FEATURES 7
#define ROAD_FEATURES_ONEHOT 13
#define PARTNER_FEATURES 8

// Ego features depend on dynamics model
#define EGO_FEATURES_CLASSIC 7
#define EGO_FEATURES_JERK 10

// Reward-conditioning features (Gigaflow paper S. 14). Appended when
// env->reward_conditioning = 1. 9 = δ_goal + 8 α-coefficients.
#define CREWARD_FEATURES 10

// Per-step reward breakdown written to env->reward_components when the
// buffer pointer is non-NULL. Index order must stay in sync with
// Drive.REWARD_COMPONENT_NAMES in drive.py.
#define REWARD_COMPONENT_COUNT 11
enum {
    RC_COLLISION = 0,
    RC_OFFROAD,
    RC_GOAL,
    RC_JERK_LEGACY,
    RC_VELOCITY,
    RC_COMFORT,
    RC_L_ALIGN,
    RC_L_CENTER,
    RC_TIMESTEP,
    RC_REVERSE,
    RC_SPEED_LIMIT,
};

// Global state features appended to observations when include_global_state=1
// x, y, heading_x, heading_y, vx, vy, map_id
#define GLOBAL_STATE_FEATURES 7

// Observation normalization constants
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MIN_REL_GOAL_COORD -1000.0f
#define MAX_REL_GOAL_COORD 1000.0f
#define MIN_REL_AGENT_POS -1000.0f
#define MAX_REL_AGENT_POS 1000.0f
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000.0f
#define MAX_RG_COORD 1000.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

// Goal behavior
#define GOAL_RESPAWN 0
#define GOAL_GENERATE_NEW 1
#define GOAL_STOP 2
#define GOAL_REMOVE 3
#define GOAL_CONTINUE 4
#define GOAL_SAMPLE_LANE_AHEAD 5

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[13] = {-1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
                                          0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f};

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, // Second row
    {-2, 0},  {-1, 0},  {0, 0},  {1, 0},  {2, 0},  // Middle row (including center)
    {-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},  // Fourth row
    {-2, 2},  {-1, 2},  {0, 2},  {1, 2},  {2, 2}   // Bottom row
};

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};
const Color LIGHTYELLOW = (Color){255, 255, 152, 255};
const Color SOFT_YELLOW = (Color){245, 245, 220, 255};

struct timespec ts;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;

struct Log {
    float episode_return;
    float episode_length;
    float score;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    float offroad_rate;
    float collision_rate;
    float completion_rate;
    float offroad_per_agent;
    float collisions_per_agent;
    float dnf_rate;
    float n;
    float lane_alignment_rate;
    float lane_aligned_steps;   // running count of steps with lane_aligned=1 (for score_l_align)
    float speed_limit_rate;
    float lane_distance_avg;
    float lane_distance_count;
    float velocity_reward_total;
    float comfort_violations;
    float speed_at_goal;
    float active_agent_count;
    float expert_static_agent_count;
    float static_agent_count;
};

typedef struct Entity Entity;
struct Entity {
    int scenario_id;
    int type;
    int id;
    int array_size;
    float *traj_x;
    float *traj_y;
    float *traj_z;
    float *traj_vx;
    float *traj_vy;
    float *traj_vz;
    float *traj_heading;
    int *traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    float init_goal_x;
    float init_goal_y;
    int mark_as_expert;
    int collision_state;
    float metrics_array[5]; // metrics_array: [collision, offroad, reached_goal, lane_aligned
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int current_lane_idx;
    float lane_heading_diff;   // theta_f: heading diff to nearest lane [-pi, pi]
    float lane_lateral_dist;   // x_f: lateral distance to nearest lane center (m)

    // Reward-conditioning (Creward) — sampled per-agent per-episode when
    // env->reward_conditioning is enabled. See sample_agent_creward().
    float creward_delta_goal;       // U(2, 12)
    float creward_alpha_collision;  // U(0, 3)
    float creward_alpha_boundary;   // U(0, 3)
    float creward_alpha_comfort;    // U(0, 0.1)
    float creward_alpha_l_align;    // U(2.5e-4, 2.5e-2)
    float creward_alpha_vel_align;  // U(0, 1)
    float creward_alpha_l_center;   // U(2.5e-4, 7.5e-3)
    float creward_alpha_center_bias;// U(-0.5, 0.5)
    float creward_alpha_reverse;    // U(2.5e-4, 7.5e-3)
    float creward_goal_speed;       // U(3, 30) m/s

    int valid;
    int respawn_timestep;
    int respawn_count;
    int collided_before_goal;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    int current_goal_reached;
    int active_agent;
    int stopped;
    int removed;

    // Collision snapshot — saved at collision detection time, before position clearing
    int collided_with_index;          // entity index of collision partner (-1 = none)
    float collision_x, collision_y;   // ego position when AABB overlap detected
    float collision_other_x, collision_other_y; // other's position at collision time

    // Jerk dynamics
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;

    // Movement mode: MOVEMENT_DYNAMICS (default) or MOVEMENT_IDM
    int movement_mode;

    // IDM (Waymax-style): target velocity from expert initial speed
    float idm_target_velocity;         // desired velocity [m/s], default 15.0
    float idm_lateral_offset;          // lateral offset from lane center [m], default 0.0

    // Lane connectivity (exit_lanes from enriched binary)
    int exit_lanes[MAX_EXIT_LANES];
    int exit_lane_count;

    // Lane route for IDM (built from lane-center polylines)
    float *route_x;
    float *route_y;
    float *route_heading;
    int route_size;
    int route_progress;  // current index on route (monotonically increasing)
};

void free_entity(Entity *entity) {
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
    // free lane route arrays
    free(entity->route_x);
    free(entity->route_y);
    free(entity->route_heading);
}

// Utility functions
float relative_distance(float a, float b) {
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx * dx + dy * dy);
    return distance;
}

float clip(float value, float min, float max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

typedef struct GridMapEntity GridMapEntity;
struct GridMapEntity {
    int entity_idx;
    int geometry_idx;
};

typedef struct GridMap GridMap;
struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int cell_size_x;
    int cell_size_y;
    int *cell_entities_count; // number of entities in each cell of the GridMap
    GridMapEntity **cells;    // list of gridEntities in each cell of the GridMap
    // Extras/Optimizations
    int vision_range;
    int *neighbor_cache_count;               // number of entities in each cells neighbor cache
    GridMapEntity **neighbor_cache_entities; // preallocated array to hold neighbor entities
};

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    // Decomposed rewards for CEM planning
    float *collision_rewards;
    float *offroad_rewards;
    float *goal_rewards;
    float *goal_distances;  // Current distance to goal for each agent
    float *jerk_rewards;    // Jerk penalty for each agent
    float *lane_distances;  // Distance to closest lane center for each agent
    float *lane_alignments; // |lane_heading_diff| in [0, π] for each agent
    float *reward_components;  // Optional: [num_active_agents][REWARD_COMPONENT_COUNT]
                               // per-step reward breakdown; NULL = no-op writes
    float *reward_components_raw;  // Optional: same shape; stores the pre-α
                                   // formula body so cross-policy behavior
                                   // can be compared independent of conditioning.
    unsigned char *terminals;
    unsigned char *truncations;
    Log log;
    Log *logs;
    int num_agents;
    int active_agent_count;
    int *active_agent_indices;
    int action_type;
    int human_agent_idx;
    Entity *entities;
    int num_entities;
    int num_actors;
    int num_objects;
    int num_roads;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    GridMap *grid_map;
    int *neighbor_offsets;
    int episode_length;
    int termination_mode;
    float reward_vehicle_collision;
    float reward_offroad_collision;
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
    int reward_conditioning;  // 0: use global env->reward_*, 1: use per-agent creward_* fields
    // Deterministic creward override (for eval): if 1, sample_agent_creward copies
    // creward_ego[] into the ego agent (active-index == human_agent_idx) and
    // cycles through creward_traffic[0..count-1] by agent_idx for every other
    // agent, instead of random sampling.
    // Field order: [delta_goal, alpha_collision, alpha_boundary, alpha_comfort,
    //               alpha_l_align, alpha_vel_align, alpha_l_center,
    //               alpha_center_bias, alpha_reverse].
    int creward_deterministic;
    int emit_jerk_ego_obs;
    // Entity index (not position) of the agent that should receive creward_ego
    // in deterministic mode. -1 = fall back to active_agent_indices[human_agent_idx]
    // (which works only when the evaluator passes a position, not an entity id).
    int ego_entity_idx;
    float creward_ego[CREWARD_FEATURES];
    #define MAX_TRAFFIC_PROFILES 16
    int creward_traffic_count;
    float creward_traffic[MAX_TRAFFIC_PROFILES][CREWARD_FEATURES];
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    float reward_goal;
    float reward_goal_post_respawn;
    float goal_radius;
    float goal_speed;
    int max_controlled_agents;
    int logs_capacity;
    int use_goal_generation;
    int scenario_length;
    int control_non_vehicles;
    float *map_corners;
    int goal_behavior;
    float goal_target_distance;
    float goal_lane_change_prob;  // for GOAL_SAMPLE_LANE_AHEAD: p of parallel-lane jump
    char *ini_file;
    char *scenario_id;
    int collision_behavior;
    int offroad_behavior;
    int sdc_track_index;
    int num_tracks_to_predict;
    int *tracks_to_predict_indices;
    int init_mode;
    int control_mode;
    int idm_others;  // If 1, static agents use IDM instead of expert replay
    int include_global_state;  // If 1, append absolute (x, y, heading_x, heading_y, vx, vy, map_id) to observations
    int map_id;  // Map ID for this environment instance
    int placeholder_agents;  // Number of placeholder entities for user-added agents
    float collision_shrink;   // Bounding box shrink factor for collision detection (e.g. 0.7)
    float idm_min_gap;        // IDM: minimum gap to lead vehicle [m]
    float idm_headway_time;   // IDM: desired time headway [s]
    float idm_accel_max;      // IDM: maximum acceleration [m/s²]
    float idm_decel_max;      // IDM: maximum deceleration [m/s²]
    int max_obs_partners;     // Max partners in observation (default: MAX_OBS_PARTNERS)
    // Traffic mix: fraction of controlled agents assigned to each type (sum to 1.0; ppo=0 = disabled)
    float traffic_mix_ppo;       // e.g. 0.4
    float traffic_mix_idm;       // e.g. 0.4
    float traffic_mix_expert;    // e.g. 0.2
    int   idm_random_velocity;   // 1 = sample from {10,15,20,30}, 0 = use fixed
    float idm_default_velocity;  // fixed IDM velocity when not random (default 15.0)
};

// Forward declarations for functions defined later
void build_lane_routes(Drive* env);

void add_log(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity *e = &env->entities[env->active_agent_indices[i]];

        env->log.goals_reached_this_episode += e->goals_reached_this_episode;
        env->log.goals_sampled_this_episode += e->goals_sampled_this_episode;

        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        float offroad_per_agent = env->logs[i].offroad_per_agent;
        env->log.offroad_per_agent += offroad_per_agent;
        float collisions_per_agent = env->logs[i].collisions_per_agent;
        env->log.collisions_per_agent += collisions_per_agent;

        float frac_goal_reached = e->goals_reached_this_episode / e->goals_sampled_this_episode;

        // Update score, which is an aggregate measure whether the agent fully solved its task
        // Note: When resampling goals, performance is relative to the number of goals sampled
        float threshold = 0.99f; // Default threshold for 1 goal
        if (e->goals_sampled_this_episode == 2.0f) {
            threshold = 0.5f; // Require ≥50% completion for 2 goals
        } else if (e->goals_sampled_this_episode < 5.0f) {
            threshold = 0.8f; // Require ≥80% completion for 3-4 goals
        } else {
            threshold = 0.9f; // Require ≥90% completion for 5+ goals
        }

        int collision_occurred =
            (env->goal_behavior == GOAL_RESPAWN) ? e->collided_before_goal : env->logs[i].collision_rate;
        if (frac_goal_reached > threshold && !collision_occurred) {
            env->log.score += 1.0f;
        }
        if (!offroad && !collided && frac_goal_reached < 1.0f) {
            env->log.dnf_rate += 1.0f;
        }
        int lane_aligned = env->logs[i].lane_alignment_rate;
        env->log.lane_alignment_rate += lane_aligned;
        env->log.lane_aligned_steps += env->logs[i].lane_aligned_steps;
        env->log.speed_limit_rate += env->logs[i].speed_limit_rate;
        env->log.lane_distance_avg += env->logs[i].lane_distance_avg;
        env->log.lane_distance_count += env->logs[i].lane_distance_count;
        env->log.velocity_reward_total += env->logs[i].velocity_reward_total;
        env->log.comfort_violations += env->logs[i].comfort_violations;
        env->log.speed_at_goal += env->logs[i].speed_at_goal;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        // Log composition counts per agent so vec_log averaging recovers the per-env value
        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_agent_count += env->expert_static_agent_count;
        env->log.static_agent_count += env->static_agent_count;
        env->log.n += 1;
    }
}

Entity *load_map_binary(const char *filename, Drive *env) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;

    // Read sdc_track_index
    fread(&env->sdc_track_index, sizeof(int), 1, file);

    // Read tracks_to_predict
    fread(&env->num_tracks_to_predict, sizeof(int), 1, file);
    if (env->num_tracks_to_predict > 0) {
        env->tracks_to_predict_indices = (int *)malloc(env->num_tracks_to_predict * sizeof(int));
        for (int i = 0; i < env->num_tracks_to_predict; i++) {
            fread(&env->tracks_to_predict_indices[i], sizeof(int), 1, file);
        }
    } else {
        env->tracks_to_predict_indices = NULL;
    }

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity *entities = (Entity *)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].scenario_id, sizeof(int), 1, file);
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].id, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float *)malloc(size * sizeof(float));
        entities[i].traj_y = (float *)malloc(size * sizeof(float));
        entities[i].traj_z = (float *)malloc(size * sizeof(float));
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float *)malloc(size * sizeof(float));
            entities[i].traj_vy = (float *)malloc(size * sizeof(float));
            entities[i].traj_vz = (float *)malloc(size * sizeof(float));
            entities[i].traj_heading = (float *)malloc(size * sizeof(float));
            entities[i].traj_valid = (int *)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);

        // Read exit_lanes connectivity (present in v3 binaries)
        int num_exit;
        fread(&num_exit, sizeof(int), 1, file);
        entities[i].exit_lane_count = (num_exit > MAX_EXIT_LANES) ? MAX_EXIT_LANES : num_exit;
        for (int j = 0; j < entities[i].exit_lane_count; j++) {
            fread(&entities[i].exit_lanes[j], sizeof(int), 1, file);
        }
        // Skip remaining exit_lanes if more than MAX_EXIT_LANES
        for (int j = entities[i].exit_lane_count; j < num_exit; j++) {
            int dummy;
            fread(&dummy, sizeof(int), 1, file);
        }

        // Initialize route (built later in build_lane_routes)
        entities[i].route_x = NULL;
        entities[i].route_y = NULL;
        entities[i].route_heading = NULL;
        entities[i].route_size = 0;
        entities[i].route_progress = 0;

        // Initialize non-serialized fields
        entities[i].movement_mode = MOVEMENT_DYNAMICS;

        // Mark trajectory timesteps as invalid where position or velocity == INVALID_POSITION
        // (agents that disappear from the scene are padded with -10000)
        if (entities[i].traj_valid != NULL) {
            for (int t = 0; t < entities[i].array_size; t++) {
                if (entities[i].traj_x[t] == INVALID_POSITION ||
                    entities[i].traj_y[t] == INVALID_POSITION ||
                    entities[i].traj_vx[t] == INVALID_POSITION ||
                    entities[i].traj_vy[t] == INVALID_POSITION) {
                    entities[i].traj_valid[t] = 0;
                }
            }
        }

        // Set IDM target velocity — default 15 m/s
        entities[i].idm_target_velocity = 15.0f;
        entities[i].idm_lateral_offset = 0.0f;
    }

    fclose(file);
    return entities;
}

void add_placeholder_entities(Drive* env, int count) {
    if (count <= 0) return;

    int old_num_entities = env->num_entities;
    int new_num_entities = old_num_entities + count;
    int insert_at = env->num_objects;  // Insert before road entities

    // Reallocate entities array
    env->entities = (Entity *)realloc(env->entities, new_num_entities * sizeof(Entity));

    // Shift road entities (from insert_at..old_num_entities-1) to the right by count
    int num_road_entities = old_num_entities - insert_at;
    if (num_road_entities > 0) {
        memmove(&env->entities[insert_at + count], &env->entities[insert_at],
                num_road_entities * sizeof(Entity));
    }

    int array_size = env->episode_length + 1;

    for (int i = 0; i < count; i++) {
        int idx = insert_at + i;
        Entity* e = &env->entities[idx];
        memset(e, 0, sizeof(Entity));

        e->type = 1;  // VEHICLE
        e->id = 90000 + i;
        e->scenario_id = env->entities[0].scenario_id;
        e->length = 4.5f;
        e->width = 2.0f;
        e->height = 1.5f;
        e->mark_as_expert = 0;
        e->movement_mode = MOVEMENT_DYNAMICS;
        e->goal_position_x = -9000.0f;
        e->goal_position_y = -9000.0f;
        e->goal_position_z = 0.0f;
        e->init_goal_x = -9000.0f;
        e->init_goal_y = -9000.0f;
        e->idm_target_velocity = 15.0f;
        e->idm_lateral_offset = 0.0f;

        e->array_size = array_size;

        // Allocate trajectory arrays (position at INVALID_POSITION)
        e->traj_x = (float *)calloc(array_size, sizeof(float));
        e->traj_y = (float *)calloc(array_size, sizeof(float));
        e->traj_z = (float *)calloc(array_size, sizeof(float));
        e->traj_vx = (float *)calloc(array_size, sizeof(float));
        e->traj_vy = (float *)calloc(array_size, sizeof(float));
        e->traj_vz = (float *)calloc(array_size, sizeof(float));
        e->traj_heading = (float *)calloc(array_size, sizeof(float));
        e->traj_valid = (int *)calloc(array_size, sizeof(int));

        for (int t = 0; t < array_size; t++) {
            e->traj_x[t] = INVALID_POSITION;
            e->traj_y[t] = INVALID_POSITION;
            e->traj_valid[t] = 1;  // Valid so set_active_agents picks them up
        }

        // Route (initialized later in build_lane_routes)
        e->route_x = NULL;
        e->route_y = NULL;
        e->route_heading = NULL;
        e->route_size = 0;
        e->route_progress = 0;
    }

    env->num_objects += count;
    env->num_entities += count;
}

void set_start_position(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Entity *e = &env->entities[i];

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->array_size)
            step = e->array_size - 1;
        if (step < 0)
            step = 0;

        e->x = e->traj_x[step];
        e->y = e->traj_y[step];
        e->z = e->traj_z[step];
        if (e->type > CYCLIST || e->type == 0) {
            continue;
        }
        e->valid = e->traj_valid[env->init_steps];

        // If initial position is invalid (agent not yet in scene), mark removed
        if (!e->valid || e->x == INVALID_POSITION || e->y == INVALID_POSITION) {
            e->x = INVALID_POSITION;
            e->y = INVALID_POSITION;
            e->valid = 0;
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            if (is_active) {
                e->removed = 1;
            }
            e->collided_before_goal = 0;
            continue;
        }

        if (is_active == 0) {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->collided_before_goal = 0;
        } else {
            e->vx = e->traj_vx[env->init_steps];
            e->vy = e->traj_vy[env->init_steps];
            e->vz = e->traj_vz[env->init_steps];
        }
        e->heading = e->traj_heading[env->init_steps];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->collision_state = 0;
        e->collided_with_index = -1;
        e->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->metrics_array[LANE_DISTANCE_IDX] = 0.0f; // lane distance
        e->respawn_timestep = -1;
        e->stopped = 0;
        e->removed = 0;
        e->respawn_count = 0;

        // Dynamics
        e->a_long = 0.0f;
        e->a_lat = 0.0f;
        e->jerk_long = 0.0f;
        e->jerk_lat = 0.0f;
        e->steering_angle = 0.0f;
        e->wheelbase = 0.6f * e->length;

        // Reset IDM route progress so route following restarts from beginning
        e->route_progress = 0;
    }
}

int getGridIndex(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x ||
        env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1; // Invalid grid coordinates
    }

    float relativeX = x1 - env->grid_map->top_left_x;     // Distance from left
    float relativeY = y1 - env->grid_map->bottom_right_y; // Distance from bottom
    int gridX = (int)(relativeX / GRID_CELL_SIZE);        // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);        // Row index
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1; // Return -1 for out of bounds
    }
    int index = (gridY * env->grid_map->grid_cols) + gridX;
    return index;
}

void add_entity_to_grid(Drive *env, int grid_index, int entity_idx, int geometry_idx, int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
               "%d\n",
               grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

void init_grid_map(Drive *env) {
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap *)malloc(sizeof(GridMap));

    // Find top left and bottom right points of the map
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int first_valid_point = 0;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            // Check all points in the trajectory for road elements
            Entity *e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++) {
                if (e->traj_x[j] == INVALID_POSITION)
                    continue;
                if (e->traj_y[j] == INVALID_POSITION)
                    continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if (e->traj_x[j] < top_left_x)
                    top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x)
                    bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] > top_left_y)
                    top_left_y = e->traj_y[j];
                if (e->traj_y[j] < bottom_right_y)
                    bottom_right_y = e->traj_y[j];
            }
        }
    }

    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;
    env->grid_map->cell_size_x = GRID_CELL_SIZE;
    env->grid_map->cell_size_y = GRID_CELL_SIZE;

    env->map_corners = (float*)calloc(4, sizeof(float));
    env->map_corners[0] = top_left_x;
    env->map_corners[1] = top_left_y;
    env->map_corners[2] = bottom_right_x;
    env->map_corners[3] = bottom_right_y;


    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity **)calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                env->grid_map->cell_entities_count[grid_index]++;
            }
        }
    }
    int cell_entities_insert_index[grid_cell_count]; // Helper array for insertion index
    memset(cell_entities_insert_index, 0, grid_cell_count * sizeof(int));

    // Initialize grid cells
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        env->grid_map->cells[grid_index] =
            (GridMapEntity *)calloc(env->grid_map->cell_entities_count[grid_index], sizeof(GridMapEntity));
    }
    for (int i = 0; i < grid_cell_count; i++) {
        if (cell_entities_insert_index[i] != 0) {
            printf("Error: cell_entities_insert_index[%d] not zero during initialization.\n", i);
            cell_entities_insert_index[i] = 0;
        }
    }

    // Populate grid cells
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 &&
            env->entities[i].type < 7) { // NOTE: Only Road Edges, Lines, and Lanes in grid map
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            }
        }
    }
}

void init_neighbor_offsets(Drive *env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int *)calloc(env->grid_map->vision_range * env->grid_map->vision_range * 2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;                  // Current x offset
    int y = 0;                  // Current y offset
    int dir = 0;                // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1;      // Number of steps in current direction
    int steps_taken = 0;        // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0;              // Total offsets added
    int max_offsets = env->grid_map->vision_range * env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0; // x offset
    env->neighbor_offsets[curr_idx++] = 0; // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range / 2 && abs(y) <= env->grid_map->vision_range / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;     // Reset steps taken
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **)calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *)calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i] =
            (GridMapEntity *)calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];

            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }

            int src_idx = grid_index;
            int dst_idx = base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->grid_map->neighbor_cache_entities[i][dst_idx], env->grid_map->cells[src_idx],
                   grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

int get_neighbor_cache_entities(Drive *env, int cell_idx, GridMapEntity *entities, int max_entities) {
    GridMap *grid_map = env->grid_map;
    if (cell_idx < 0 || cell_idx >= (grid_map->grid_cols * grid_map->grid_rows)) {
        return 0; // Invalid cell index
    }

    int count = grid_map->neighbor_cache_count[cell_idx];
    // Limit to available space
    if (count > max_entities) {
        count = max_entities;
    }
    memcpy(entities, grid_map->neighbor_cache_entities[cell_idx], count * sizeof(GridMapEntity));
    return count;
}

void set_means(Drive *env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all entities (vehicles and roads)
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                // Assume a validity flag exists (e.g., valid[j]); adjust if not available
                if (env->entities[i].traj_valid[j]) { // Add validity check if applicable
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        } else if (env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST || env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_x[j] == INVALID_POSITION)
                    continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    if (agent->traj_valid && agent->traj_valid[t] == 0) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0)
        return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive *env, float x, float y, GridMapEntity *entity_list, int max_size,
                   const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1)
        return 0; // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            int entityId = env->grid_map->cells[neighborIndex][j].entity_idx;
            int geometry_idx = env->grid_map->cells[neighborIndex][j].geometry_idx;
            entity_list[entity_list_count].entity_idx = entityId;
            entity_list[entity_list_count].geometry_idx = geometry_idx;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity *car1, Entity *car2) {
    // Get car corners in world space
    float cos1 = car1->heading_x;
    float sin1 = car1->heading_y;
    float cos2 = car2->heading_x;
    float sin2 = car2->heading_y;

    // Calculate half dimensions
    float half_len1 = car1->length * 0.5f;
    float half_width1 = car1->width * 0.5f;
    float half_len2 = car2->length * 0.5f;
    float half_width2 = car2->width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->x + (half_len1 * cos1 - half_width1 * sin1), car1->y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (half_len1 * cos1 + half_width1 * sin1), car1->y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 - half_width1 * sin1), car1->y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}};

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},  // Car1's length axis
        {-sin1, cos1}, // Car1's width axis
        {cos2, sin2},  // Car2's length axis
        {-sin2, cos2}  // Car2's width axis
    };

    // Check each axis
    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if (max1 < min2 || min1 > max2) {
            return 0; // No collision
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1; // Collision
}

int collision_check(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];

    if (agent->x == INVALID_POSITION)
        return -1;

    int car_collided_with_index = -1;

    if (agent->respawn_timestep != -1)
        return car_collided_with_index; // Skip respawning entities

    for (int i = 0; i < MAX_AGENTS; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->num_actors) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;
        Entity *entity = &env->entities[index];
        if (entity->respawn_timestep != -1)
            continue; // Skip respawning entities
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = ((x1 - agent->x) * (x1 - agent->x) + (y1 - agent->y) * (y1 - agent->y));
        if (dist > 225.0f)
            continue;
        if (check_aabb_collision(agent, entity)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

int check_lane_aligned(Entity *car, Entity *lane, int geometry_idx) {
    // Validate lane geometry length
    if (!lane || lane->array_size < 2)
        return 0;

    // Clamp geometry index to valid segment range [0, array_size-2]
    if (geometry_idx < 0)
        geometry_idx = 0;
    if (geometry_idx >= lane->array_size - 1)
        geometry_idx = lane->array_size - 2;

    // Compute local lane segment heading
    float heading_x1, heading_y1;
    if (geometry_idx > 0) {
        heading_x1 = lane->traj_x[geometry_idx] - lane->traj_x[geometry_idx - 1];
        heading_y1 = lane->traj_y[geometry_idx] - lane->traj_y[geometry_idx - 1];
    } else {
        // For first segment, just use the forward direction
        heading_x1 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
        heading_y1 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];
    }

    float heading_x2 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
    float heading_y2 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];

    float heading_1 = atan2f(heading_y1, heading_x1);
    float heading_2 = atan2f(heading_y2, heading_x2);
    float heading = (heading_1 + heading_2) / 2.0f;

    // Normalize to [-pi, pi]
    if (heading > M_PI)
        heading -= 2.0f * M_PI;
    if (heading < -M_PI)
        heading += 2.0f * M_PI;

    // Compute heading difference
    float car_heading = car->heading; // radians
    float heading_diff = fabsf(car_heading - heading);

    if (heading_diff > M_PI)
        heading_diff = 2.0f * M_PI - heading_diff;

    // within 15 degrees
    return (heading_diff < (M_PI / 12.0f)) ? 1 : 0;
}

void reset_agent_metrics(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->metrics_array[LANE_DISTANCE_IDX] = 0.0f; // lane distance
    agent->collision_state = 0;
    agent->collided_with_index = -1;
}

float point_to_segment_distance_2d(float px, float py, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;

    if (dx == 0 && dy == 0) {
        // The segment is a point
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }

    // Calculate the t that minimizes the distance
    float t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // Clamp t to the segment
    if (t < 0)
        t = 0;
    else if (t > 1)
        t = 1;

    // Find the closest point on the segment
    float closestX = x1 + t * dx;
    float closestY = y1 + t * dy;

    // Return the distance from p to the closest point
    return sqrtf((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
}

void compute_agent_metrics(Drive *env, int agent_idx, int env_idx) {
    Entity *agent = &env->entities[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if (agent->x == INVALID_POSITION)
        return; // invalid agent position (e.g. if removed before.)

    int collided = 0;
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float min_distance = (float)INT16_MAX;
    float best_signed_lateral = 0.0f;

    int closest_lane_entity_idx = -1;
    int closest_lane_geometry_idx = -1;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25]; // Array big enough for all neighboring cells
    int list_size =
        checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        if (entity_list[i].entity_idx == agent_idx)
            continue;
        Entity *entity;
        entity = &env->entities[entity_list[i].entity_idx];

        // Check for offroad collision with road edges
        if (entity->type == ROAD_EDGE) {
            int geometry_idx = entity_list[i].geometry_idx;
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided = OFFROAD;
                    break;
                }
            }
        }

        if (collided == OFFROAD)
            break;

        // Find closest point on the road centerline to the agent (paper: current lane = lane containing the agent).
        if (entity->type == ROAD_LANE) {
            int entity_idx = entity_list[i].entity_idx;
            int geometry_idx = entity_list[i].geometry_idx;

            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};

            float dist = point_to_segment_distance_2d(agent->x, agent->y, start[0], start[1], end[0], end[1]);

            if (dist < min_distance) {
                // Signed lateral distance via 2D cross product with lane direction.
                float lx = end[0] - start[0];
                float ly = end[1] - start[1];
                float ax = agent->x - start[0];
                float ay = agent->y - start[1];
                float cross = ax * ly - ay * lx;
                min_distance = dist;
                best_signed_lateral = (cross >= 0.0f) ? dist : -dist;
                closest_lane_entity_idx = entity_idx;
                closest_lane_geometry_idx = geometry_idx;
            }
        }
    }

    // check if aligned with closest lane and set current lane
    // 4.0m threshold: agents more than 4 meters from any lane are considered off-road
    if (min_distance > 4.0f || closest_lane_entity_idx == -1) {
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        agent->current_lane_idx = -1;
        agent->lane_heading_diff = M_PI;  // max misalignment
        agent->lane_lateral_dist = (min_distance < (float)INT16_MAX) ? min_distance : 10.0f;
    } else {
        agent->current_lane_idx = closest_lane_entity_idx;
        int lane_aligned =
            check_lane_aligned(agent, &env->entities[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned;

        // Compute theta_f: heading difference to lane segment
        Entity *lane = &env->entities[closest_lane_entity_idx];
        int g = closest_lane_geometry_idx;
        if (g >= lane->array_size - 1) g = lane->array_size - 2;
        if (g < 0) g = 0;
        float lx = lane->traj_x[g + 1] - lane->traj_x[g];
        float ly = lane->traj_y[g + 1] - lane->traj_y[g];
        float lane_h = atan2f(ly, lx);
        float theta_f = agent->heading - lane_h;
        if (theta_f > M_PI) theta_f -= 2.0f * M_PI;
        if (theta_f < -M_PI) theta_f += 2.0f * M_PI;
        agent->lane_heading_diff = theta_f;
        agent->lane_lateral_dist = best_signed_lateral;
    }

    // Store lane distance in metrics_array for reward computation
    agent->metrics_array[LANE_DISTANCE_IDX] = (min_distance < (float)INT16_MAX) ? min_distance : 10.0f;

    // Store lane distance for CEM (if allocated)
    if (env->lane_distances) {
        env->lane_distances[env_idx] = agent->metrics_array[LANE_DISTANCE_IDX];
    }

    // Store lane alignment (|heading_diff| in [0, π]) for CEM (if allocated)
    if (env->lane_alignments) {
        env->lane_alignments[env_idx] = fabsf(agent->lane_heading_diff);
    }

    // Check for vehicle collisions
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1)
        collided = VEHICLE_COLLISION;

    agent->collision_state = collided;
    agent->collided_with_index = -1;

    if (collided == VEHICLE_COLLISION && car_collided_with_index != -1) {
        Entity *other_ent = &env->entities[car_collided_with_index];
        agent->collided_with_index = car_collided_with_index;
        agent->collision_x = agent->x;
        agent->collision_y = agent->y;
        agent->collision_other_x = other_ent->x;
        agent->collision_other_y = other_ent->y;
    }

    if (collided == VEHICLE_COLLISION) {
        if (env->collision_behavior == STOP_AGENT && !agent->stopped) {
            // printf("Agent %d collided with agent %d\n", agent_idx, car_collided_with_index);
            agent->stopped = 1;
            // env->terminals[env_idx] = 1;
            // agent->vx = agent->vy = 0.0f;
        } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
            // printf("Agent %d collided with agent %d and is being removed\n", agent_idx, car_collided_with_index);
            // Entity *agent_collided = &env->entities[car_collided_with_index];
            agent->removed = 1;
            // env->terminals[env_idx] = 1;
            // agent_collided->removed = 1; -> should get removed/collided information itself later!
            // agent->x = agent->y = -10000.0f;
            // agent_collided->x = agent_collided->y = -10000.0f;
        } else if (env->collision_behavior > REMOVE_AGENT) {
            printf("Unknown setting for collision_behavior!");
        }
    }
    if (collided == OFFROAD) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
            // printf("Agent %d went offroad\n", agent_idx);
            agent->stopped = 1;
            // env->terminals[env_idx] = 1;
            // agent->vx = agent->vy = 0.0f;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            // printf("Agent %d went offroad and is being removed\n", agent_idx);
            agent->removed = 1;
            // env->terminals[env_idx] = 1;
            // agent->x = agent->y = -10000.0f;
        } else if (env->offroad_behavior > REMOVE_AGENT) {
            printf("Unknown setting for offroad_behavior!");
        }
    }

    return;
}

bool should_control_agent(Drive *env, int agent_idx, bool skip_capacity_check) {

    // Check if we have room for more agents or are already at capacity
    if (!skip_capacity_check && env->active_agent_count >= env->num_agents) {
        return false;
    }
    // Respect max_controlled_agents limit (e.g., 1 for ego-only PPO training)
    if (!skip_capacity_check && env->max_controlled_agents > 0 && env->active_agent_count >= env->max_controlled_agents) {
        return false;
    }

    Entity *entity = &env->entities[agent_idx];

    // Shrink bounding box for collision detection
    if (env->collision_shrink > 0.0f && env->collision_shrink < 1.0f) {
        entity->width *= env->collision_shrink;
        entity->length *= env->collision_shrink;
    }

    if (env->control_mode == CONTROL_SDC_ONLY) {
        return agent_idx == env->sdc_track_index;
    }

    bool is_vehicle = (entity->type == VEHICLE);
    bool is_ped_or_bike = (entity->type == PEDESTRIAN || entity->type == CYCLIST);
    bool type_is_valid = false;

    switch (env->control_mode) {
    case CONTROL_WOSAC:
    case CONTROL_EVALUATION:
        // Valid types only, ignore expert flag and goal distance
        return is_vehicle;

    case CONTROL_VEHICLES:
        type_is_valid = is_vehicle;
        break;

    default:
        type_is_valid = is_vehicle;
        break;
    }

    // Filter invalid types (experts are now controllable too)
    if (!type_is_valid) {
        return false;
    }

    // Check distance to goal in agent's local frame
    float cos_heading = cosf(entity->traj_heading[0]);
    float sin_heading = sinf(entity->traj_heading[0]);
    float goal_dx = entity->goal_position_x - entity->traj_x[0];
    float goal_dy = entity->goal_position_y - entity->traj_y[0];

    // Transform to agent's local frame
    float local_goal_x = goal_dx * cos_heading + goal_dy * sin_heading;
    float local_goal_y = -goal_dx * sin_heading + goal_dy * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, local_goal_x, local_goal_y);

    return distance_to_goal >= MIN_DISTANCE_TO_GOAL;
}

void set_active_agents(Drive *env) {

    // Initialize
    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_actors = 0;                // Total agents created

    int active_agent_indices[MAX_AGENTS];
    int static_agent_indices[MAX_AGENTS];
    int expert_static_agent_indices[MAX_AGENTS];

    // Traffic mix counters
    int ppo_count = 0, idm_count = 0, expert_count = 0;
    bool has_traffic_mix = (env->traffic_mix_idm > 0.0f || env->traffic_mix_expert > 0.0f);

    if (env->num_agents == 0) {
        env->num_agents = MAX_AGENTS;
    }

    // Mark all entities as having no agent slot (inactive)
    for (int i = 0; i < env->num_objects; i++) {
        env->entities[i].active_agent = -1;
    }

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_objects && env->num_actors < MAX_AGENTS; i++) {

        Entity *entity = &env->entities[i];

        // Skip if not valid at initialization
        if (entity->traj_valid[env->init_steps] != 1) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (env->init_mode == INIT_ALL_VALID) {
            should_create = true; // All valid entities
        } else if (env->control_mode == CONTROL_VEHICLES) {
            should_create = (entity->type >= VEHICLE && entity->type <= CYCLIST);
        } else { // Control all agents
            should_create = (entity->type >= VEHICLE && entity->type <= CYCLIST);
        }

        if (!should_create) {
            continue;
        }

        // Determine if this agent should be policy-controlled.
        // When traffic mix is active, skip the capacity check so all eligible
        // vehicles enter the sampling block. PPO count is capped there instead.
        bool is_controlled = should_control_agent(env, i, has_traffic_mix);

        if (has_traffic_mix && is_controlled) {
            // Deficit-based interleaving: assign agent to the type with the
            // largest deficit relative to its target fraction, guaranteeing
            // an even mix on every map regardless of agent count.
            int total = ppo_count + idm_count + expert_count + 1;
            float deficit_ppo    = env->traffic_mix_ppo    * total - (float)ppo_count;
            float deficit_idm    = env->traffic_mix_idm    * total - (float)idm_count;
            float deficit_expert = env->traffic_mix_expert * total - (float)expert_count;

            if ((ppo_count == 0 || (deficit_ppo >= deficit_idm && deficit_ppo >= deficit_expert))
                    && ppo_count < env->num_agents) {
                // → PPO (cap at num_agents to not overflow observation buffer)
                ppo_count++;
            } else if (deficit_idm >= deficit_expert) {
                // → IDM
                is_controlled = false;
                entity->movement_mode = MOVEMENT_IDM;
                if (env->idm_random_velocity) {
                    float grid[] = {10.0f, 15.0f, 20.0f, 30.0f};
                    entity->idm_target_velocity = grid[((unsigned int)(entity->id * 7 + entity->scenario_id * 13)) % 4];
                } else {
                    entity->idm_target_velocity = env->idm_default_velocity;
                }
                idm_count++;
            } else {
                // → Expert replay
                is_controlled = false;
                entity->movement_mode = MOVEMENT_EXPERT;
                expert_count++;
            }
        }

        if (is_controlled) {
            env->num_actors++;
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            entity->active_agent = 1;
        } else if (env->control_mode == CONTROL_EVALUATION) {
            // Evaluation: VRUs follow expert trajectories, other non-controlled agents are ignored
            if (entity->type == PEDESTRIAN || entity->type == CYCLIST) {
                env->num_actors++;
                static_agent_indices[env->static_agent_count] = i;
                env->static_agent_count++;
                entity->active_agent = 0;
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                entity->mark_as_expert = 1;
            }
            continue;
        } else if (env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            // Non-controlled vehicles follow expert trajectories
            env->num_actors++;
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            entity->active_agent = 0;
            expert_static_agent_indices[env->expert_static_agent_count] = i;
            env->expert_static_agent_count++;
            entity->mark_as_expert = 1;
        } else {
            continue;
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int *)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *)malloc(env->expert_static_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++) {
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for (int i = 0; i < env->static_agent_count; i++) {
        env->static_agent_indices[i] = static_agent_indices[i];
    }
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        env->expert_static_agent_indices[i] = expert_static_agent_indices[i];
    }

    return;
}

void remove_bad_trajectories(Drive *env) {

    if (env->control_mode != CONTROL_WOSAC) {
        return; // Leave all trajectories in WOSAC control mode
    }

    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // move experts through trajectories to check for collisions and remove as illegal agents
    for (int t = 0; t < env->episode_length; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->entities[expert_idx].x == INVALID_POSITION)
                continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = 0;
            int collided_with_index = collision_check(env, agent_idx);
            if ((collided_with_index >= 0) && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1)
            continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_agent_idx = env->static_agent_indices[j];
            if (static_agent_idx != collided_with_indices[i])
                continue;
            env->entities[static_agent_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_agent_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

void init_goal_positions(Drive *env) {
    for (int x = 0; x < env->active_agent_count; x++) {
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].init_goal_x = env->entities[agent_idx].goal_position_x;
        env->entities[agent_idx].init_goal_y = env->entities[agent_idx].goal_position_y;
    }
}

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    if (env->max_obs_partners <= 0) env->max_obs_partners = MAX_OBS_PARTNERS;
    env->idm_min_gap = 1.0f;
    env->idm_headway_time = 1.5f;
    env->idm_accel_max = 1.0f;
    env->idm_decel_max = 2.0f;
    env->entities = load_map_binary(env->map_name, env);
    if (env->placeholder_agents > 0) {
        add_placeholder_entities(env, env->placeholder_agents);
    }
    set_means(env);
    init_grid_map(env);
    env->grid_map->vision_range = 21; // TODO: Why is this hardcoded?
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    env->logs_capacity = 0;
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    remove_bad_trajectories(env);
    set_start_position(env);
    if (env->idm_others || env->traffic_mix_idm > 0.0f) {
        build_lane_routes(env);
    }
    init_goal_positions(env);
    env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
}

void c_close(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    // GridMap cleanup
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(env->grid_map->cells[grid_index]);
    }
    free(env->grid_map->cells);
    free(env->grid_map->cell_entities_count);
    free(env->neighbor_offsets);

    for (int i = 0; i < grid_cell_count; i++) {
        free(env->grid_map->neighbor_cache_entities[i]);
    }
    free(env->grid_map->neighbor_cache_entities);
    free(env->grid_map->neighbor_cache_count);
    free(env->grid_map);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->ini_file);
}

void allocate(Drive *env) {
    init(env);
    int ego_dim = (env->dynamics_model == JERK || env->emit_jerk_ego_obs) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int extra = env->include_global_state ? GLOBAL_STATE_FEATURES : 0;
    int creward_dim = env->reward_conditioning ? CREWARD_FEATURES : 0;
    int max_obs = ego_dim + PARTNER_FEATURES * env->max_obs_partners + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS + creward_dim + extra;
    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (float *)calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
    env->truncations = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    c_close(env);
}

float clipSpeed(float speed) {
    const float maxSpeed = MAX_SPEED;
    if (speed > maxSpeed)
        return maxSpeed;
    if (speed < -maxSpeed)
        return -maxSpeed;
    return speed;
}

float normalize_heading(float heading) {
    if (heading > M_PI)
        heading -= 2 * M_PI;
    if (heading < -M_PI)
        heading += 2 * M_PI;
    return heading;
}

float normalize_value(float value, float min, float max) { return (value - min) / (max - min); }

void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed)
        return;

    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        return;
    }

    if (env->dynamics_model == CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];

            acceleration *= ACCELERATION_VALUES[6];
            steering *= STEERING_VALUES[12];
        } else { // discrete
            // Interpret action as a single integer: a = accel_idx * num_steer + steer_idx
            int *action_array = (int *)env->actions;
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int action_val = action_array[action_idx];
            int acceleration_index = action_val / num_steer;
            int steering_index = action_val % num_steer;
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }

        // Current state
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;

        // Calculate current speed (signed based on direction relative to heading)
        float speed_magnitude = sqrtf(vx * vx + vy * vy);
        float v_dot_heading = vx * agent->heading_x + vy * agent->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Update speed with acceleration
        signed_speed = signed_speed + acceleration * env->dt;
        signed_speed = clipSpeed(signed_speed);
        // Compute yaw rate
        float beta = tanh(.5 * tanf(steering));

        // New heading
        float yaw_rate = (signed_speed * cosf(beta) * tanf(steering)) / agent->length;

        // New velocity
        float new_vx = signed_speed * cosf(heading + beta);
        float new_vy = signed_speed * sinf(heading + beta);

        // Update position
        x = x + (new_vx * env->dt);
        y = y + (new_vy * env->dt);
        heading = heading + yaw_rate * env->dt;

        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
        agent->steering_angle = steering;
    } else {
        // JERK dynamics model
        // Extract action components
        float a_long, a_lat;
        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;

            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float a_long_action = action_array_f[action_idx][0]; // [-1, 1]
            if (a_long_action < 0) {
                a_long = a_long_action * (-JERK_LONG[0]); // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                a_long = a_long_action * JERK_LONG[3]; // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            a_lat = action_array_f[action_idx][1] * JERK_LAT[2];
        } else { // discrete
            // Interpret action as a single integer: a = long_idx * num_lat + lat_idx
            int *action_array = (int *)env->actions;
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int action_val = action_array[action_idx];
            int a_long_idx = action_val / num_lat;
            int a_lat_idx = action_val % num_lat;
            a_long = JERK_LONG[a_long_idx];
            a_lat = JERK_LAT[a_lat_idx];
        }

        // Calculate new acceleration
        float a_long_new = agent->a_long + a_long * env->dt;
        float a_lat_new = agent->a_lat + a_lat * env->dt;

        // Make it easy to stop with 0 accel
        if (agent->a_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, -5.0f, 2.5f);
        }

        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, -4.0f, 4.0f);
        }

        // Calculate new velocity
        float v_dot_heading = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
        float signed_v = copysignf(sqrtf(agent->vx * agent->vx + agent->vy * agent->vy), v_dot_heading);
        float v_new = signed_v + 0.5f * (a_long_new + agent->a_long) * env->dt;

        // Make it easy to stop with 0 vel
        if (signed_v * v_new < 0) {
            v_new = 0.0f;
        } else {
            v_new = clip(v_new, -2.0f, 20.0f);
        }

        // Calculate new steering angle
        float signed_curvature = a_lat_new / fmaxf(v_new * v_new, 1e-5f);
        signed_curvature = copysignf(fmaxf(fabsf(signed_curvature), 1e-5f), signed_curvature);
        float steering_angle = atanf(signed_curvature * agent->wheelbase);
        float delta_steer = clip(steering_angle - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        float new_steering_angle = clip(agent->steering_angle + delta_steer, -0.55f, 0.55f);

        // Update curvature and accel to account for limited steering
        signed_curvature = tanf(new_steering_angle) / agent->wheelbase;
        a_lat_new = v_new * v_new * signed_curvature;

        // Calculate resulting movement using bicycle dynamics
        float d = 0.5f * (v_new + signed_v) * env->dt;
        float theta = d * signed_curvature;
        float dx_local, dy_local;

        if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
            dx_local = d;
            dy_local = 0.0f;
        } else {
            dx_local = sinf(theta) / signed_curvature;
            dy_local = (1.0f - cosf(theta)) / signed_curvature;
        }

        float dx = dx_local * agent->heading_x - dy_local * agent->heading_y;
        float dy = dx_local * agent->heading_y + dy_local * agent->heading_x;

        // Update everything
        agent->x += dx;
        agent->y += dy;
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->heading = normalize_heading(agent->heading + theta);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
        agent->vx = v_new * agent->heading_x;
        agent->vy = v_new * agent->heading_y;
        agent->steering_angle = new_steering_angle;
    }

    return;
}

static inline int get_track_id_or_placeholder(Drive *env, int agent_idx) {
    if (env->tracks_to_predict_indices == NULL || env->num_tracks_to_predict == 0) {
        return -1;
    }
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict_indices[k] == agent_idx) {
            return env->tracks_to_predict_indices[k];
        }
    }
    return -1;
}

void c_get_global_agent_state(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out, int *id_out,
                              float *length_out, float *width_out, int *type_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];

        if (agent->removed) {
            x_out[i] = -10000.0f;
            y_out[i] = -10000.0f;
            z_out[i] = 0.0f;
            heading_out[i] = 0.0f;
            id_out[i] = get_track_id_or_placeholder(env, agent_idx);
            length_out[i] = 0.0f;
            width_out[i] = 0.0f;
            type_out[i] = 0;
            continue;
        }

        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->x + env->world_mean_x;
        y_out[i] = agent->y + env->world_mean_y;
        z_out[i] = agent->z;
        heading_out[i] = agent->heading;
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        length_out[i] = agent->length;
        width_out[i] = agent->width;
        type_out[i] = agent->type;
    }
}

void c_get_global_ground_truth_trajectories(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out,
                                            int *valid_out, int *id_out, int *scenario_id_out, int *is_vehicle_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        scenario_id_out[i] = agent->scenario_id;
        is_vehicle_out[i] = (agent->type == VEHICLE) ? 1 : 0;

        for (int t = env->init_steps; t < agent->array_size; t++) {
            int out_idx = i * (agent->array_size - env->init_steps) + (t - env->init_steps);
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->traj_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->traj_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->traj_z[t];
            heading_out[out_idx] = agent->traj_heading[t];
            valid_out[out_idx] = agent->traj_valid[t];
        }
    }
}

void c_get_road_edge_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type == ROAD_EDGE) {
            count++;
            points += env->entities[i].array_size;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

void c_get_road_edge_polylines(Drive *env, float *x_out, float *y_out, int *lengths_out, int *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type == ROAD_EDGE) {
            lengths_out[poly_idx] = e->array_size;
            scenario_ids_out[poly_idx] = e->scenario_id;
            for (int j = 0; j < e->array_size; j++) {
                x_out[pt_idx] = e->traj_x[j] + env->world_mean_x;
                y_out[pt_idx] = e->traj_y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

// Returns counts for ALL road entities (lanes, lines, edges, driveways, etc.)
void c_get_all_road_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        int t = env->entities[i].type;
        if (t >= ROAD_LANE && t <= ROAD_EDGE) {  // types 4, 5, 6
            count++;
            points += env->entities[i].array_size;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

// Returns polyline data for ALL road entities with type info
void c_get_all_road_polylines(Drive *env, float *x_out, float *y_out,
                              int *lengths_out, int *types_out, int *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        int t = e->type;
        if (t >= ROAD_LANE && t <= ROAD_EDGE) {
            lengths_out[poly_idx] = e->array_size;
            types_out[poly_idx] = t;  // ROAD_LANE=4, ROAD_LINE=5, ROAD_EDGE=6
            scenario_ids_out[poly_idx] = e->scenario_id;
            for (int j = 0; j < e->array_size; j++) {
                x_out[pt_idx] = e->traj_x[j] + env->world_mean_x;
                y_out[pt_idx] = e->traj_y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

void compute_observations(Drive *env) {
    int ego_dim = (env->dynamics_model == JERK || env->emit_jerk_ego_obs) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int extra = env->include_global_state ? GLOBAL_STATE_FEATURES : 0;
    int creward_dim = env->reward_conditioning ? CREWARD_FEATURES : 0;
    int max_obs = ego_dim + PARTNER_FEATURES * env->max_obs_partners + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS + creward_dim + extra;
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        Entity *ego_entity = &env->entities[env->active_agent_indices[i]];
        if (ego_entity->type > 3)
            break;

        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float speed_magnitude = sqrtf(ego_entity->vx * ego_entity->vx + ego_entity->vy * ego_entity->vy);
        float v_dot_heading = ego_entity->vx * ego_entity->heading_x + ego_entity->vy * ego_entity->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;

        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
        float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;

        obs[0] = rel_goal_x * 0.005f;
        obs[1] = rel_goal_y * 0.005f;
        obs[2] = signed_speed / MAX_SPEED;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;

        if (env->dynamics_model == JERK || env->emit_jerk_ego_obs) {
            obs[6] = ego_entity->steering_angle / M_PI;
            // Asymmetric normalization for a_long to match action space
            obs[7] =
                (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[8] = ego_entity->a_lat / JERK_LAT[2];
            obs[9] = (ego_entity->respawn_timestep != -1) ? 1 : 0;
        } else {
            obs[6] = (ego_entity->respawn_timestep != -1) ? 1 : 0;
        }

        // Collect candidate partners with distances, then sort by distance
        int obs_idx = ego_dim;
        int num_candidates = 0;
        struct { int index; float dist_sq; } candidates[MAX_AGENTS];

        if (ego_entity->respawn_timestep == -1) {
            for (int j = 0; j < env->num_actors; j++) {
                int index = -1;
                if (j < env->active_agent_count) {
                    index = env->active_agent_indices[j];
                } else {
                    index = env->static_agent_indices[j - env->active_agent_count];
                }
                if (index == -1) continue;
                if (env->entities[index].type > 3) break;
                if (index == env->active_agent_indices[i]) continue;
                Entity *other_entity = &env->entities[index];
                if (other_entity->x == INVALID_POSITION || other_entity->removed) continue;
                if (other_entity->respawn_timestep != -1) continue;
                float dx = other_entity->x - ego_entity->x;
                float dy = other_entity->y - ego_entity->y;
                float dist_sq = dx * dx + dy * dy;
                if (!(dist_sq <= 2500.0f)) continue;
                candidates[num_candidates].index = index;
                candidates[num_candidates].dist_sq = dist_sq;
                num_candidates++;
            }
        }

        // Simple insertion sort by distance (small N)
        for (int a = 1; a < num_candidates; a++) {
            int tmp_idx = candidates[a].index;
            float tmp_dist = candidates[a].dist_sq;
            int b = a - 1;
            while (b >= 0 && candidates[b].dist_sq > tmp_dist) {
                candidates[b + 1] = candidates[b];
                b--;
            }
            candidates[b + 1].index = tmp_idx;
            candidates[b + 1].dist_sq = tmp_dist;
        }

        // Write the closest max_obs_partners into observation
        int cars_seen = 0;
        int limit = (num_candidates < env->max_obs_partners) ? num_candidates : env->max_obs_partners;
        for (int c = 0; c < limit; c++) {
            Entity *other_entity = &env->entities[candidates[c].index];
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float rel_x = dx * cos_heading + dy * sin_heading;
            float rel_y = -dx * sin_heading + dy * cos_heading;
            obs[obs_idx] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            float rel_heading_x =
                other_entity->heading_x * ego_entity->heading_x +
                other_entity->heading_y * ego_entity->heading_y;
            float rel_heading_y =
                other_entity->heading_y * ego_entity->heading_x -
                other_entity->heading_x * ego_entity->heading_y;
            obs[obs_idx + 4] = rel_heading_x;
            obs[obs_idx + 5] = rel_heading_y;
            float other_speed_magnitude =
                sqrtf(other_entity->vx * other_entity->vx + other_entity->vy * other_entity->vy);
            float other_v_dot_heading =
                other_entity->vx * other_entity->heading_x + other_entity->vy * other_entity->heading_y;
            float other_signed_speed = copysignf(other_speed_magnitude, other_v_dot_heading);
            obs[obs_idx + 6] = other_signed_speed / MAX_SPEED;
            obs[obs_idx + 7] = (float)other_entity->type;
            cars_seen++;
            obs_idx += PARTNER_FEATURES;
        }
        int remaining_partner_obs = (env->max_obs_partners - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);

        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Validate entity_idx before accessing
            if (entity_idx < 0 || entity_idx >= env->num_entities) {
                printf("ERROR: Invalid entity_idx %d (max: %d)\n", entity_idx, env->num_entities - 1);
                continue;
            }

            Entity *entity = &env->entities[entity_idx];

            // Validate geometry_idx before accessing
            if (geometry_idx < 0 || geometry_idx >= entity->array_size) {
                printf("ERROR: Invalid geometry_idx %d for entity %d (max: %d)\n", geometry_idx, entity_idx,
                       entity->array_size - 1);
                continue;
            }
            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego_entity->x;
            float rel_y = mid_y - ego_entity->y;
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
            float width = 0.1;
            // Calculate angle from ego to midpoint (vector from ego to midpoint)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float dx_norm = dx;
            float dy_norm = dy;
            float hypot = sqrtf(dx * dx + dy * dy);
            if (hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }
            // Compute sin and cos of relative angle directly without atan2f
            float cos_angle = dx_norm * cos_heading + dy_norm * sin_heading;
            float sin_angle = -dx_norm * sin_heading + dy_norm * cos_heading;
            obs[obs_idx] = x_obs * 0.02f;
            obs[obs_idx + 1] = y_obs * 0.02f;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = width / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - 4.0f;
            obs_idx += 7;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * 7;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
        obs_idx += remaining_obs;

        // Creward conditioning features (Gigaflow paper S. 14), normalized to [-1, 1].
        if (env->reward_conditioning) {
            obs[obs_idx++] = 2.0f * (ego_entity->creward_delta_goal        -  2.0f)   / (12.0f    -  2.0f)    - 1.0f;
            obs[obs_idx++] = 2.0f *  ego_entity->creward_alpha_collision   /  3.0f                           - 1.0f;
            obs[obs_idx++] = 2.0f *  ego_entity->creward_alpha_boundary    /  3.0f                           - 1.0f;
            obs[obs_idx++] = 2.0f *  ego_entity->creward_alpha_comfort     /  0.1f                           - 1.0f;
            obs[obs_idx++] = 2.0f * (ego_entity->creward_alpha_l_align     -  2.5e-4f) / (2.5e-2f - 2.5e-4f) - 1.0f;
            obs[obs_idx++] = 2.0f *  ego_entity->creward_alpha_vel_align   /  1.0f                           - 1.0f;
            obs[obs_idx++] = 2.0f * (ego_entity->creward_alpha_l_center    -  2.5e-4f) / (7.5e-3f - 2.5e-4f) - 1.0f;
            obs[obs_idx++] = 2.0f *  ego_entity->creward_alpha_center_bias;                                    // [-0.5,0.5] -> [-1,1]
            obs[obs_idx++] = 2.0f * (ego_entity->creward_alpha_reverse     -  2.5e-4f) / (7.5e-3f - 2.5e-4f) - 1.0f;
            obs[obs_idx++] = 2.0f * (ego_entity->creward_goal_speed        -  3.0f)    / (30.0f   -  3.0f)   - 1.0f;
        }

        // Append absolute global state for SMART reference model (SPACeR KL)
        if (env->include_global_state) {
            obs[obs_idx]     = ego_entity->x;
            obs[obs_idx + 1] = ego_entity->y;
            obs[obs_idx + 2] = ego_entity->heading_x;
            obs[obs_idx + 3] = ego_entity->heading_y;
            obs[obs_idx + 4] = ego_entity->vx;
            obs[obs_idx + 5] = ego_entity->vy;
            obs[obs_idx + 6] = (float)env->map_id;
        }
    }
}

void sample_new_goal(Drive *env, int agent_idx) {
    // Samples a new goal position based on the existing road lane points
    Entity *agent = &env->entities[agent_idx];
    float best_x = agent->x;
    float best_y = agent->y;
    float best_distance_error = 1e30f;

    // Sample points from all road lanes
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type != ROAD_LANE)
            continue;

        Entity *lane = &env->entities[i];

        // Check every point in the lane
        for (int j = 0; j < lane->array_size; j++) {
            float point_x = lane->traj_x[j];
            float point_y = lane->traj_y[j];

            // Calculate vector from agent to point
            float to_point_x = point_x - agent->x;
            float to_point_y = point_y - agent->y;

            // Check if point is ahead of agent
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            if (dot <= 0.0f)
                continue;

            // Calculate distance to point
            float distance = sqrtf(to_point_x * to_point_x + to_point_y * to_point_y);

            // Find point closest to target distance
            float distance_error = fabsf(distance - env->goal_target_distance);
            if (distance_error < best_distance_error) {
                best_distance_error = distance_error;
                best_x = point_x;
                best_y = point_y;
            }
        }
    }

    // If no valid goal found, use another agent's initial goal
    if (best_distance_error >= 1e30f && env->active_agent_count > 1) {
        int other_idx = env->active_agent_indices[(agent_idx + 1) % env->active_agent_count];
        best_x = env->entities[other_idx].init_goal_x;
        best_y = env->entities[other_idx].init_goal_y;
    }

    agent->goal_position_x = best_x;
    agent->goal_position_y = best_y;
    agent->goals_sampled_this_episode += 1;
}

static inline float frand01(void);  // forward decl (defined later)

// Walk `lane` from its closest point to the agent in the direction of the
// agent's heading, accumulate arc-length up to env->goal_target_distance.
// Guarantees the returned (gx, gy) is strictly ahead of the agent
// (dot with heading > 0); returns 0 if the lane has no such forward point.
static int walk_lane_ahead(Drive *env, Entity *agent, Entity *lane,
                           float *gx, float *gy) {
    if (lane->array_size < 2) return 0;

    int j0 = 0;
    float best_d2 = 1e30f;
    for (int j = 0; j < lane->array_size; j++) {
        float dx = lane->traj_x[j] - agent->x;
        float dy = lane->traj_y[j] - agent->y;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; j0 = j; }
    }

    // Direction along the polyline that moves us forward (ahead of agent).
    // Try both neighbors of j0; pick whichever is more forward.
    float ahead_plus = -1e30f, ahead_minus = -1e30f;
    if (j0 + 1 < lane->array_size) {
        float dx = lane->traj_x[j0 + 1] - agent->x;
        float dy = lane->traj_y[j0 + 1] - agent->y;
        ahead_plus = dx * agent->heading_x + dy * agent->heading_y;
    }
    if (j0 - 1 >= 0) {
        float dx = lane->traj_x[j0 - 1] - agent->x;
        float dy = lane->traj_y[j0 - 1] - agent->y;
        ahead_minus = dx * agent->heading_x + dy * agent->heading_y;
    }
    if (ahead_plus <= 0.0f && ahead_minus <= 0.0f) return 0;  // nothing forward
    int step = (ahead_plus >= ahead_minus) ? +1 : -1;

    float target = env->goal_target_distance;
    float cum = 0.0f;
    float x = lane->traj_x[j0], y = lane->traj_y[j0];
    int found_forward = 0;
    int j = j0;
    while (1) {
        int jn = j + step;
        if (jn < 0 || jn >= lane->array_size) break;
        float dx = lane->traj_x[jn] - lane->traj_x[j];
        float dy = lane->traj_y[jn] - lane->traj_y[j];
        cum += sqrtf(dx * dx + dy * dy);
        float cand_x = lane->traj_x[jn], cand_y = lane->traj_y[jn];
        // Only accept candidates that are strictly ahead of agent.
        float fdx = cand_x - agent->x, fdy = cand_y - agent->y;
        if (fdx * agent->heading_x + fdy * agent->heading_y > 0.0f) {
            x = cand_x; y = cand_y;
            found_forward = 1;
            if (cum >= target) break;
        }
        j = jn;
    }
    if (!found_forward) return 0;
    *gx = x; *gy = y;
    return 1;
}

// Sample new goal on the current lane, or — with probability
// env->goal_lane_change_prob — on a parallel lane (same heading, ~lane-width
// offset). Falls back to sample_new_goal when nothing suitable is found.
void sample_new_goal_on_lane(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int cur_idx = agent->current_lane_idx;

    int want_change = (env->goal_lane_change_prob > 0.0f
                       && frand01() < env->goal_lane_change_prob);

    if (want_change && cur_idx >= 0) {
        // One-lane hop only: typical lane width ~3.5 m, so a lane 2 over
        // sits ~7 m away.  MAX=4.5 m admits exactly the adjacent lane.
        const float LANE_CHANGE_MIN_M       = 2.0f;
        const float LANE_CHANGE_MAX_M       = 4.5f;
        const float LANE_CHANGE_HEADING_DOT = 0.8f;
        const int   MAX_CANDIDATES          = 8;
        int candidates[MAX_CANDIDATES];
        int n_cand = 0;

        for (int li = env->num_objects; li < env->num_entities; li++) {
            if (li == cur_idx) continue;
            if (env->entities[li].type != ROAD_LANE) continue;
            Entity *lane = &env->entities[li];
            if (lane->array_size < 2) continue;

            int j0 = 0;
            float best_d2 = 1e30f;
            for (int j = 0; j < lane->array_size; j++) {
                float dx = lane->traj_x[j] - agent->x;
                float dy = lane->traj_y[j] - agent->y;
                float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) { best_d2 = d2; j0 = j; }
            }
            float d = sqrtf(best_d2);
            if (d < LANE_CHANGE_MIN_M || d > LANE_CHANGE_MAX_M) continue;

            int jn = (j0 + 1 < lane->array_size) ? j0 + 1 : j0;
            int jp = (j0 - 1 >= 0) ? j0 - 1 : j0;
            float sx = lane->traj_x[jn] - lane->traj_x[jp];
            float sy = lane->traj_y[jn] - lane->traj_y[jp];
            float slen = sqrtf(sx * sx + sy * sy);
            if (slen < 1e-4f) continue;
            float heading_dot = (sx * agent->heading_x + sy * agent->heading_y) / slen;
            if (heading_dot < LANE_CHANGE_HEADING_DOT) continue;  // same-direction only

            float fdx = lane->traj_x[j0] - agent->x;
            float fdy = lane->traj_y[j0] - agent->y;
            if (fdx * agent->heading_x + fdy * agent->heading_y < -LANE_CHANGE_MIN_M) continue;

            candidates[n_cand++] = li;
            if (n_cand == MAX_CANDIDATES) break;
        }
        if (n_cand > 0) {
            int pick = candidates[(int)(frand01() * n_cand) % n_cand];
            float gx, gy;
            if (walk_lane_ahead(env, agent, &env->entities[pick], &gx, &gy)) {
                agent->goal_position_x = gx;
                agent->goal_position_y = gy;
                agent->goals_sampled_this_episode += 1;
                return;
            }
        }
    }

    if (cur_idx >= 0 && cur_idx < env->num_entities
            && env->entities[cur_idx].type == ROAD_LANE) {
        float gx, gy;
        if (walk_lane_ahead(env, agent, &env->entities[cur_idx], &gx, &gy)) {
            agent->goal_position_x = gx;
            agent->goal_position_y = gy;
            agent->goals_sampled_this_episode += 1;
            return;
        }
    }

    // Fallback: project straight ahead of the agent by goal_target_distance.
    // This keeps the goal in front (never multi-lane-jumping) when no valid
    // lane-following continuation exists.
    agent->goal_position_x = agent->x + agent->heading_x * env->goal_target_distance;
    agent->goal_position_y = agent->y + agent->heading_y * env->goal_target_distance;
    agent->goals_sampled_this_episode += 1;
}

// ============================================================================
// SNAPSHOT FUNCTIONALITY - Save and restore simulator state
// ============================================================================

typedef struct EntitySnapshot EntitySnapshot;
struct EntitySnapshot {
    float x, y, z;
    float vx, vy, vz;
    float heading, heading_x, heading_y;
    int collision_state;
    int valid;
    int respawn_timestep;
    int respawn_count;
    int collided_before_goal;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    int current_goal_reached;
    int stopped;
    int removed;
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float metrics_array[5];
    float goal_position_x;
    float goal_position_y;
    int movement_mode;
    int current_lane_idx;

    float idm_target_velocity;
    float idm_lateral_offset;

    // Route snapshot (deep-copied)
    float *route_x;
    float *route_y;
    float *route_heading;
    int route_size;
    int route_progress;

    // Creward (per-agent reward conditioning) — only populated if reward_conditioning enabled
    float creward_delta_goal;
    float creward_alpha_collision;
    float creward_alpha_boundary;
    float creward_alpha_comfort;
    float creward_alpha_l_align;
    float creward_alpha_vel_align;
    float creward_alpha_l_center;
    float creward_alpha_center_bias;
    float creward_alpha_reverse;
    float creward_goal_speed;
};

typedef struct DriveSnapshot DriveSnapshot;
struct DriveSnapshot {
    int timestep;
    int num_entities;
    int active_agent_count;
    EntitySnapshot* entity_snapshots;
    Log* logs;
    float* observations;
    float* rewards;
    unsigned char* terminals;
    int obs_size;
    int num_agents;
};

DriveSnapshot* create_snapshot(Drive* env) {
    DriveSnapshot* snapshot = (DriveSnapshot*)calloc(1, sizeof(DriveSnapshot));
    if (!snapshot) return NULL;

    snapshot->timestep = env->timestep;
    snapshot->num_entities = env->num_entities;
    snapshot->active_agent_count = env->active_agent_count;
    snapshot->num_agents = env->num_agents;

    // Snapshot all entities
    snapshot->entity_snapshots = (EntitySnapshot*)calloc(env->num_entities, sizeof(EntitySnapshot));
    if (!snapshot->entity_snapshots) {
        free(snapshot);
        return NULL;
    }

    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        EntitySnapshot* es = &snapshot->entity_snapshots[i];

        es->x = e->x;
        es->y = e->y;
        es->z = e->z;
        es->vx = e->vx;
        es->vy = e->vy;
        es->vz = e->vz;
        es->heading = e->heading;
        es->heading_x = e->heading_x;
        es->heading_y = e->heading_y;
        es->collision_state = e->collision_state;
        es->valid = e->valid;
        es->respawn_timestep = e->respawn_timestep;
        es->respawn_count = e->respawn_count;
        es->collided_before_goal = e->collided_before_goal;
        es->goals_reached_this_episode = e->goals_reached_this_episode;
        es->goals_sampled_this_episode = e->goals_sampled_this_episode;
        es->current_goal_reached = e->current_goal_reached;
        es->stopped = e->stopped;
        es->removed = e->removed;
        es->a_long = e->a_long;
        es->a_lat = e->a_lat;
        es->jerk_long = e->jerk_long;
        es->jerk_lat = e->jerk_lat;
        es->steering_angle = e->steering_angle;
        es->creward_delta_goal         = e->creward_delta_goal;
        es->creward_alpha_collision    = e->creward_alpha_collision;
        es->creward_alpha_boundary     = e->creward_alpha_boundary;
        es->creward_alpha_comfort      = e->creward_alpha_comfort;
        es->creward_alpha_l_align      = e->creward_alpha_l_align;
        es->creward_alpha_vel_align    = e->creward_alpha_vel_align;
        es->creward_alpha_l_center     = e->creward_alpha_l_center;
        es->creward_alpha_center_bias  = e->creward_alpha_center_bias;
        es->creward_alpha_reverse      = e->creward_alpha_reverse;
        es->creward_goal_speed         = e->creward_goal_speed;
        es->goal_position_x = e->goal_position_x;
        es->goal_position_y = e->goal_position_y;
        es->movement_mode = e->movement_mode;
        es->current_lane_idx = e->current_lane_idx;
        es->idm_target_velocity = e->idm_target_velocity;
        es->idm_lateral_offset = e->idm_lateral_offset;

        // Deep-copy route arrays
        es->route_size = e->route_size;
        es->route_progress = e->route_progress;
        if (e->route_size > 0 && e->route_x) {
            es->route_x = (float*)malloc(e->route_size * sizeof(float));
            es->route_y = (float*)malloc(e->route_size * sizeof(float));
            es->route_heading = (float*)malloc(e->route_size * sizeof(float));
            memcpy(es->route_x, e->route_x, e->route_size * sizeof(float));
            memcpy(es->route_y, e->route_y, e->route_size * sizeof(float));
            memcpy(es->route_heading, e->route_heading, e->route_size * sizeof(float));
        } else {
            es->route_x = NULL;
            es->route_y = NULL;
            es->route_heading = NULL;
        }

        for (int j = 0; j < 5; j++) {
            es->metrics_array[j] = e->metrics_array[j];
        }
    }

    // Snapshot logs
    snapshot->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
    if (snapshot->logs) {
        memcpy(snapshot->logs, env->logs, env->active_agent_count * sizeof(Log));
    }

    // Snapshot observations (matches compute_observations sizing)
    int ego_dim = (env->dynamics_model == JERK || env->emit_jerk_ego_obs) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int gs_extra = env->include_global_state ? GLOBAL_STATE_FEATURES : 0;
    int creward_dim = env->reward_conditioning ? CREWARD_FEATURES : 0;
    int obs_per_agent = ego_dim + PARTNER_FEATURES * env->max_obs_partners
        + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS + creward_dim + gs_extra;
    snapshot->obs_size = env->active_agent_count * obs_per_agent;
    snapshot->observations = (float*)calloc(snapshot->obs_size, sizeof(float));
    if (snapshot->observations) {
        memcpy(snapshot->observations, env->observations, snapshot->obs_size * sizeof(float));
    }

    // Snapshot rewards and terminals
    snapshot->rewards = (float*)calloc(env->active_agent_count, sizeof(float));
    if (snapshot->rewards) {
        memcpy(snapshot->rewards, env->rewards, env->active_agent_count * sizeof(float));
    }

    snapshot->terminals = (unsigned char*)calloc(env->active_agent_count, sizeof(unsigned char));
    if (snapshot->terminals) {
        memcpy(snapshot->terminals, env->terminals, env->active_agent_count * sizeof(unsigned char));
    }

    return snapshot;
}

void restore_snapshot(Drive* env, DriveSnapshot* snapshot) {
    if (!snapshot) return;

    env->timestep = snapshot->timestep;

    // Restore all entities
    for (int i = 0; i < env->num_entities && i < snapshot->num_entities; i++) {
        Entity* e = &env->entities[i];
        EntitySnapshot* es = &snapshot->entity_snapshots[i];

        e->x = es->x;
        e->y = es->y;
        e->z = es->z;
        e->vx = es->vx;
        e->vy = es->vy;
        e->vz = es->vz;
        e->heading = es->heading;
        e->heading_x = es->heading_x;
        e->heading_y = es->heading_y;
        e->collision_state = es->collision_state;
        e->valid = es->valid;
        e->respawn_timestep = es->respawn_timestep;
        e->respawn_count = es->respawn_count;
        e->collided_before_goal = es->collided_before_goal;
        e->goals_reached_this_episode = es->goals_reached_this_episode;
        e->goals_sampled_this_episode = es->goals_sampled_this_episode;
        e->current_goal_reached = es->current_goal_reached;
        e->stopped = es->stopped;
        e->removed = es->removed;
        e->a_long = es->a_long;
        e->a_lat = es->a_lat;
        e->jerk_long = es->jerk_long;
        e->jerk_lat = es->jerk_lat;
        e->steering_angle = es->steering_angle;
        e->creward_delta_goal         = es->creward_delta_goal;
        e->creward_alpha_collision    = es->creward_alpha_collision;
        e->creward_alpha_boundary     = es->creward_alpha_boundary;
        e->creward_alpha_comfort      = es->creward_alpha_comfort;
        e->creward_alpha_l_align      = es->creward_alpha_l_align;
        e->creward_alpha_vel_align    = es->creward_alpha_vel_align;
        e->creward_alpha_l_center     = es->creward_alpha_l_center;
        e->creward_alpha_center_bias  = es->creward_alpha_center_bias;
        e->creward_alpha_reverse      = es->creward_alpha_reverse;
        e->creward_goal_speed         = es->creward_goal_speed;
        e->goal_position_x = es->goal_position_x;
        e->goal_position_y = es->goal_position_y;
        e->movement_mode = es->movement_mode;
        e->current_lane_idx = es->current_lane_idx;
        e->idm_target_velocity = es->idm_target_velocity;
        e->idm_lateral_offset = es->idm_lateral_offset;

        // Restore route arrays (deep-copy from snapshot)
        if (e->route_x) { free(e->route_x); e->route_x = NULL; }
        if (e->route_y) { free(e->route_y); e->route_y = NULL; }
        if (e->route_heading) { free(e->route_heading); e->route_heading = NULL; }
        e->route_size = es->route_size;
        e->route_progress = es->route_progress;
        if (es->route_size > 0 && es->route_x) {
            e->route_x = (float*)malloc(es->route_size * sizeof(float));
            e->route_y = (float*)malloc(es->route_size * sizeof(float));
            e->route_heading = (float*)malloc(es->route_size * sizeof(float));
            memcpy(e->route_x, es->route_x, es->route_size * sizeof(float));
            memcpy(e->route_y, es->route_y, es->route_size * sizeof(float));
            memcpy(e->route_heading, es->route_heading, es->route_size * sizeof(float));
        }

        for (int j = 0; j < 5; j++) {
            e->metrics_array[j] = es->metrics_array[j];
        }
    }

    // Restore logs
    if (snapshot->logs && env->logs) {
        memcpy(env->logs, snapshot->logs, env->active_agent_count * sizeof(Log));
    }

    // Restore observations
    if (snapshot->observations && env->observations) {
        memcpy(env->observations, snapshot->observations, snapshot->obs_size * sizeof(float));
    }

    // Restore rewards
    if (snapshot->rewards && env->rewards) {
        memcpy(env->rewards, snapshot->rewards, env->active_agent_count * sizeof(float));
    }

    // Restore terminals
    if (snapshot->terminals && env->terminals) {
        memcpy(env->terminals, snapshot->terminals, env->active_agent_count * sizeof(unsigned char));
    }
}

void free_snapshot(DriveSnapshot* snapshot) {
    if (!snapshot) return;

    // Free deep-copied route arrays in entity snapshots
    if (snapshot->entity_snapshots) {
        for (int i = 0; i < snapshot->num_entities; i++) {
            EntitySnapshot* es = &snapshot->entity_snapshots[i];
            if (es->route_x) free(es->route_x);
            if (es->route_y) free(es->route_y);
            if (es->route_heading) free(es->route_heading);
        }
        free(snapshot->entity_snapshots);
    }
    if (snapshot->logs) free(snapshot->logs);
    if (snapshot->observations) free(snapshot->observations);
    if (snapshot->rewards) free(snapshot->rewards);
    if (snapshot->terminals) free(snapshot->terminals);
    free(snapshot);
}

// ============================================================================

static inline float frand01(void) {
    return (float)rand() / (float)RAND_MAX;
}

// Sample per-agent reward-conditioning α values from the paper-specified U(a,b)
// ranges (Gigaflow Table A2). Called at c_reset and respawn_agent when
// env->reward_conditioning is enabled. If env->creward_deterministic is set,
// copies the fixed ego/traffic profiles from env->creward_ego / creward_traffic
// instead of random sampling.
static inline void sample_agent_creward(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (env->creward_deterministic) {
        // Prefer explicit ego_entity_idx when set (direct entity index, no
        // interpretation ambiguity). Otherwise fall back to treating
        // human_agent_idx as a position in active_agent_indices.
        int ego_entity_idx = env->ego_entity_idx;
        if (ego_entity_idx < 0) {
            ego_entity_idx = (env->active_agent_indices && env->active_agent_count > 0)
                ? env->active_agent_indices[env->human_agent_idx]
                : -1;
        }
        const float *src;
        if (agent_idx == ego_entity_idx) {
            src = env->creward_ego;
        } else {
            int count = env->creward_traffic_count > 0 ? env->creward_traffic_count : 1;
            if (count > MAX_TRAFFIC_PROFILES) count = MAX_TRAFFIC_PROFILES;
            // Deterministic per-entity dispatch so the same agent always gets
            // the same profile across resets/respawns within a scene.
            int p = ((agent_idx % count) + count) % count;
            src = env->creward_traffic[p];
        }
        // delta_goal defaults to env->goal_radius so the conditioning signal
        // naturally matches the actual goal in the scene. A user who passes
        // creward_ego_delta_goal > 0 (sweep experiments, extrapolation) gets
        // their value instead. Training (creward_deterministic=0) unaffected.
        agent->creward_delta_goal        = (src[0] > 0.0f) ? src[0] : env->goal_radius;
        agent->creward_alpha_collision   = src[1];
        agent->creward_alpha_boundary    = src[2];
        agent->creward_alpha_comfort     = src[3];
        agent->creward_alpha_l_align     = src[4];
        agent->creward_alpha_vel_align   = src[5];
        agent->creward_alpha_l_center    = src[6];
        agent->creward_alpha_center_bias = src[7];
        agent->creward_alpha_reverse     = src[8];
        agent->creward_goal_speed        = src[9];
        return;
    }
    agent->creward_delta_goal        =  2.0f    + frand01() * (12.0f    -  2.0f);
    agent->creward_alpha_collision   =           frand01() *  3.0f;
    agent->creward_alpha_boundary    =           frand01() *  3.0f;
    agent->creward_alpha_comfort     =           frand01() *  0.1f;
    agent->creward_alpha_l_align     =  2.5e-4f + frand01() * (2.5e-2f - 2.5e-4f);
    agent->creward_alpha_vel_align   =           frand01() *  1.0f;
    agent->creward_alpha_l_center    =  2.5e-4f + frand01() * (7.5e-3f - 2.5e-4f);
    agent->creward_alpha_center_bias = -0.5f    + frand01() *  1.0f;
    agent->creward_alpha_reverse     =  2.5e-4f + frand01() * (7.5e-3f - 2.5e-4f);
    agent->creward_goal_speed        =  3.0f    + frand01() * (30.0f   -  3.0f);
}

void c_reset(Drive *env) {
    env->timestep = env->init_steps;
    // TODO: check if this is correct here
    // memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));

    set_start_position(env);
    if (env->idm_others || env->traffic_mix_idm > 0.0f) {
        build_lane_routes(env);
    }
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].respawn_count = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].goals_reached_this_episode = 0.0f;
        // Initialize to 1 because there is one goal in the data file
        env->entities[agent_idx].goals_sampled_this_episode = 1.0f;
        env->entities[agent_idx].current_goal_reached = 0;
        env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        env->entities[agent_idx].stopped = 0;
        env->entities[agent_idx].removed = 0;

        if (env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_SAMPLE_LANE_AHEAD) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].init_goal_x;
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].init_goal_y;
        }

        // For GOAL_SAMPLE_LANE_AHEAD: agents whose initial goal is within 2 m
        // of their start position are "parked" scenarios; leave them stopped
        // instead of chasing a meaningless goal or immediately respawning.
        if (env->goal_behavior == GOAL_SAMPLE_LANE_AHEAD) {
            Entity *e = &env->entities[agent_idx];
            float dgx = e->init_goal_x - e->x;
            float dgy = e->init_goal_y - e->y;
            if (dgx * dgx + dgy * dgy < 4.0f) {  // 2 m radius
                e->stopped = 1;
                e->vx = 0.0f;
                e->vy = 0.0f;
            }
        }

        if (env->reward_conditioning) {
            sample_agent_creward(env, agent_idx);
        }

        compute_agent_metrics(env, agent_idx, x);
    }
    compute_observations(env);
}

void respawn_agent(Drive *env, int agent_idx) {
    env->entities[agent_idx].x = env->entities[agent_idx].traj_x[0];
    env->entities[agent_idx].y = env->entities[agent_idx].traj_y[0];
    env->entities[agent_idx].heading = env->entities[agent_idx].traj_heading[0];
    env->entities[agent_idx].heading_x = cosf(env->entities[agent_idx].heading);
    env->entities[agent_idx].heading_y = sinf(env->entities[agent_idx].heading);
    env->entities[agent_idx].vx = env->entities[agent_idx].traj_vx[0];
    env->entities[agent_idx].vy = env->entities[agent_idx].traj_vy[0];
    env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;

    env->entities[agent_idx].respawn_timestep = env->timestep;
    env->entities[agent_idx].collided_before_goal = 0;
    env->entities[agent_idx].stopped = 0;
    env->entities[agent_idx].removed = 0;
    env->entities[agent_idx].a_long = 0.0f;
    env->entities[agent_idx].a_lat = 0.0f;
    env->entities[agent_idx].jerk_long = 0.0f;
    env->entities[agent_idx].jerk_lat = 0.0f;
    env->entities[agent_idx].steering_angle = 0.0f;
    if (env->reward_conditioning) {
        sample_agent_creward(env, agent_idx);
    }
}

// ============================================================================
// IDM (Intelligent Driver Model) functions — V-Max style
// Lane-center route + Bezier blending + Bicycle dynamics
// ============================================================================

static float wrap_angle(float a) {
    while (a > M_PI) a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

// --- Phase 2: Lane Route Builder ---

// Find the entity index of a lane with the given id (linear scan)
static int find_lane_entity_by_id(Drive* env, int lane_id) {
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type == ROAD_LANE && env->entities[i].id == lane_id) {
            return i;
        }
    }
    return -1;
}

// Build lane route for a single agent (V-Max get_sdc_lane + lane chaining)
// Resample a polyline to uniform spacing (returns new count, modifies arrays in-place)
static int resample_route_uniform(float* x, float* y, int count, float spacing, int max_pts) {
    if (count < 2 || spacing <= 0.0f) return count;

    // Compute cumulative arc lengths
    float* arc = (float*)malloc(count * sizeof(float));
    arc[0] = 0.0f;
    for (int i = 1; i < count; i++) {
        float dx = x[i] - x[i-1];
        float dy = y[i] - y[i-1];
        arc[i] = arc[i-1] + sqrtf(dx * dx + dy * dy);
    }
    float total_len = arc[count - 1];
    if (total_len < spacing) { free(arc); return count; }

    // Number of output points
    int n_out = (int)(total_len / spacing) + 1;
    if (n_out > max_pts) n_out = max_pts;
    if (n_out < 2) { free(arc); return count; }

    float* ox = (float*)malloc(n_out * sizeof(float));
    float* oy = (float*)malloc(n_out * sizeof(float));

    int seg = 0;  // current segment index
    for (int i = 0; i < n_out; i++) {
        float target = i * spacing;
        if (target >= total_len) target = total_len - 0.001f;

        // Advance segment pointer
        while (seg < count - 2 && arc[seg + 1] < target) seg++;

        float seg_len = arc[seg + 1] - arc[seg];
        float t = (seg_len > 0.001f) ? (target - arc[seg]) / seg_len : 0.0f;
        ox[i] = x[seg] + t * (x[seg + 1] - x[seg]);
        oy[i] = y[seg] + t * (y[seg + 1] - y[seg]);
    }

    // Copy back
    for (int i = 0; i < n_out; i++) { x[i] = ox[i]; y[i] = oy[i]; }

    free(arc);
    free(ox);
    free(oy);
    return n_out;
}

static void build_route_for_agent(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];

    // Skip invalid agents
    if (agent->removed || agent->x == INVALID_POSITION) return;

    // Free any existing route
    if (agent->route_x) { free(agent->route_x); agent->route_x = NULL; }
    if (agent->route_y) { free(agent->route_y); agent->route_y = NULL; }
    if (agent->route_heading) { free(agent->route_heading); agent->route_heading = NULL; }
    agent->route_size = 0;

    float ax = agent->x;
    float ay = agent->y;
    float ah = agent->heading;

    // --- a) Find the closest lane matching agent heading ---
    // Use combined score: distance + heading penalty (Bezier blend handles misalignment)
    int best_lane_idx = -1;
    int best_point_idx = -1;
    float best_score = INFINITY;
    const float heading_threshold = 1.0f;   // hard cutoff at ~57° (reject opposite lanes)
    const float heading_weight = 5.0f;      // 1 rad heading error = 5m distance penalty
    const float max_lane_dist_sq = 25.0f;   // 5m max perpendicular distance to lane

    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity* lane = &env->entities[i];
        if (lane->type != ROAD_LANE) continue;
        if (lane->array_size < 2) continue;

        for (int p = 0; p < lane->array_size - 1; p++) {
            // Compute lane segment direction
            float seg_dx = lane->traj_x[p + 1] - lane->traj_x[p];
            float seg_dy = lane->traj_y[p + 1] - lane->traj_y[p];
            float seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy;
            if (seg_len_sq < 0.0001f) continue;
            float lane_heading = atan2f(seg_dy, seg_dx);

            // Heading filter (hard cutoff to reject opposite-direction lanes)
            float hdiff = fabsf(wrap_angle(lane_heading - ah));
            if (hdiff > heading_threshold) continue;

            // Perpendicular distance from agent to lane segment
            float rel_x = ax - lane->traj_x[p];
            float rel_y = ay - lane->traj_y[p];
            float t = (rel_x * seg_dx + rel_y * seg_dy) / seg_len_sq;
            t = fmaxf(0.0f, fminf(1.0f, t));
            float proj_x = lane->traj_x[p] + t * seg_dx;
            float proj_y = lane->traj_y[p] + t * seg_dy;
            float dx = ax - proj_x;
            float dy = ay - proj_y;
            float d2 = dx * dx + dy * dy;

            // Skip lanes too far from agent (prevents matching parallel roads)
            if (d2 > max_lane_dist_sq) continue;

            // Combined score: spatial distance + heading penalty
            float score = d2 + heading_weight * heading_weight * hdiff * hdiff;
            if (score < best_score) {
                best_score = score;
                best_lane_idx = i;
                // Use the point index at the projection (start of segment or next)
                best_point_idx = (t < 0.5f) ? p : p + 1;
            }
        }
    }

    // No matching lane found — agent will use expert-trajectory fallback in move_idm
    if (best_lane_idx < 0) return;

    // --- b) Allocate route and copy lane points from start_point onward ---
    float* tmp_x = (float*)malloc(MAX_ROUTE_POINTS * sizeof(float));
    float* tmp_y = (float*)malloc(MAX_ROUTE_POINTS * sizeof(float));
    int count = 0;

    Entity* lane = &env->entities[best_lane_idx];
    for (int p = best_point_idx; p < lane->array_size && count < MAX_ROUTE_POINTS; p++) {
        tmp_x[count] = lane->traj_x[p];
        tmp_y[count] = lane->traj_y[p];
        count++;
    }

    // --- c) Chain lanes via exit_lanes (stop after 120m or 20 iterations) ---
    const float max_route_length = 120.0f;  // ~12s at 10 m/s
    float route_length = 0.0f;
    // Compute initial route length from first lane segment
    for (int p = 1; p < count; p++) {
        float dx = tmp_x[p] - tmp_x[p-1];
        float dy = tmp_y[p] - tmp_y[p-1];
        route_length += sqrtf(dx * dx + dy * dy);
    }
    int current_lane_idx = best_lane_idx;
    for (int chain = 0; chain < 20 && count < MAX_ROUTE_POINTS && route_length < max_route_length; chain++) {
        Entity* cur = &env->entities[current_lane_idx];

        if (cur->exit_lane_count == 0) {
            // Spatial fallback: find lane whose first point is ≤5m from last route point
            // and heading within ±60°
            float end_x = tmp_x[count - 1];
            float end_y = tmp_y[count - 1];
            // Compute heading from second-to-last to last route point
            float end_heading = ah;  // fallback to agent heading when count < 2
            if (count >= 2) {
                end_heading = atan2f(tmp_y[count-1] - tmp_y[count-2],
                                     tmp_x[count-1] - tmp_x[count-2]);
            }

            int fallback_idx = -1;
            float fallback_dist = INFINITY;
            for (int i = env->num_objects; i < env->num_entities; i++) {
                if (i == current_lane_idx) continue;
                Entity* cand = &env->entities[i];
                if (cand->type != ROAD_LANE || cand->array_size < 2) continue;
                float dx = cand->traj_x[0] - end_x;
                float dy = cand->traj_y[0] - end_y;
                float d2 = dx * dx + dy * dy;
                if (d2 > 25.0f) continue;  // > 5m
                float cand_heading = atan2f(cand->traj_y[1] - cand->traj_y[0],
                                            cand->traj_x[1] - cand->traj_x[0]);
                if (fabsf(wrap_angle(cand_heading - end_heading)) > 0.5f) continue;  // > ~30°
                if (d2 < fallback_dist) {
                    fallback_dist = d2;
                    fallback_idx = i;
                }
            }

            if (fallback_idx < 0) break;  // No continuation found — end route
            current_lane_idx = fallback_idx;
        } else if (cur->exit_lane_count == 1) {
            // Single exit lane — take it directly
            int next_idx = find_lane_entity_by_id(env, cur->exit_lanes[0]);
            if (next_idx < 0) break;
            current_lane_idx = next_idx;
        } else {
            // Multiple exit lanes — pick the one whose start is closest to
            // the current route end (lateral continuity, no lane jumps)
            float end_x = tmp_x[count - 1];
            float end_y = tmp_y[count - 1];
            float end_heading = ah;  // fallback to agent heading when count < 2
            if (count >= 2) {
                end_heading = atan2f(tmp_y[count-1] - tmp_y[count-2],
                                     tmp_x[count-1] - tmp_x[count-2]);
            }
            int best_exit_idx = -1;
            float best_exit_dist = INFINITY;
            for (int e = 0; e < cur->exit_lane_count; e++) {
                int cand_idx = find_lane_entity_by_id(env, cur->exit_lanes[e]);
                if (cand_idx < 0) continue;
                Entity* cand = &env->entities[cand_idx];
                if (cand->array_size < 2) continue;
                // Heading continuity check
                float cand_heading = atan2f(cand->traj_y[1] - cand->traj_y[0],
                                            cand->traj_x[1] - cand->traj_x[0]);
                if (fabsf(wrap_angle(cand_heading - end_heading)) > 0.5f) continue;  // > ~30°
                // Distance from route end to candidate start
                float dx = cand->traj_x[0] - end_x;
                float dy = cand->traj_y[0] - end_y;
                float d2 = dx * dx + dy * dy;
                if (d2 < best_exit_dist) {
                    best_exit_dist = d2;
                    best_exit_idx = cand_idx;
                }
            }
            if (best_exit_idx < 0) break;
            current_lane_idx = best_exit_idx;
        }

        // Append points from the next lane (track cumulative length)
        Entity* next_lane = &env->entities[current_lane_idx];
        for (int p = 0; p < next_lane->array_size && count < MAX_ROUTE_POINTS && route_length < max_route_length; p++) {
            tmp_x[count] = next_lane->traj_x[p];
            tmp_y[count] = next_lane->traj_y[p];
            if (count > 0) {
                float dx = tmp_x[count] - tmp_x[count-1];
                float dy = tmp_y[count] - tmp_y[count-1];
                route_length += sqrtf(dx * dx + dy * dy);
            }
            count++;
        }
    }

    // Extend route linearly past last lane point to ensure it reaches the goal
    if (count >= 2) {
        float dx = tmp_x[count-1] - tmp_x[count-2];
        float dy = tmp_y[count-1] - tmp_y[count-2];
        float seg_len = sqrtf(dx * dx + dy * dy);
        if (seg_len > 0.001f) {
            float ux = dx / seg_len;
            float uy = dy / seg_len;
            int extend_limit = count + 30;  // add up to 30m extra
            if (extend_limit > MAX_ROUTE_POINTS) extend_limit = MAX_ROUTE_POINTS;
            while (count < extend_limit) {
                tmp_x[count] = tmp_x[count-1] + ux * 1.0f;
                tmp_y[count] = tmp_y[count-1] + uy * 1.0f;
                count++;
            }
        }
    }

    if (count < 2) {
        free(tmp_x);
        free(tmp_y);
        return;
    }

    // --- d) Resample to uniform 1m spacing ---
    count = resample_route_uniform(tmp_x, tmp_y, count, 1.0f, MAX_ROUTE_POINTS);

    if (count < 2) {
        free(tmp_x);
        free(tmp_y);
        return;
    }

    // --- e) Compute route headings ---
    agent->route_x = (float*)realloc(tmp_x, count * sizeof(float));
    agent->route_y = (float*)realloc(tmp_y, count * sizeof(float));
    agent->route_heading = (float*)malloc(count * sizeof(float));
    agent->route_size = count;
    agent->route_progress = 0;

    for (int k = 0; k < count - 1; k++) {
        agent->route_heading[k] = atan2f(agent->route_y[k+1] - agent->route_y[k],
                                          agent->route_x[k+1] - agent->route_x[k]);
    }
    agent->route_heading[count - 1] = agent->route_heading[count - 2];
}

// Build lane routes for all active agents — called after set_start_position in c_reset
void build_lane_routes(Drive* env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        build_route_for_agent(env, env->active_agent_indices[i]);
    }
    // Also build for static agents that might use IDM
    for (int i = 0; i < env->num_objects; i++) {
        if (env->entities[i].route_size == 0) {
            build_route_for_agent(env, i);
        }
    }
}

// --- Phase 3: Bezier Trajectory Generation (V-Max) ---

// Find the closest point on the route to position (px, py), starting from start_idx
static int find_closest_route_point(Entity* agent, float px, float py, int start_idx) {
    int best = start_idx;
    float best_d2 = INFINITY;
    for (int k = start_idx; k < agent->route_size; k++) {
        float dx = agent->route_x[k] - px;
        float dy = agent->route_y[k] - py;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = k;
        }
    }
    return best;
}

// Find route point at arc-length distance d from route_start_idx
static int find_route_point_at_distance(Entity* agent, int route_start_idx, float d) {
    float cumulative = 0.0f;
    for (int k = route_start_idx; k < agent->route_size - 1; k++) {
        float dx = agent->route_x[k+1] - agent->route_x[k];
        float dy = agent->route_y[k+1] - agent->route_y[k];
        cumulative += sqrtf(dx * dx + dy * dy);
        if (cumulative >= d) return k + 1;
    }
    return agent->route_size - 1;
}

// Compute max curvature in a window of route points
static float compute_max_curvature(Entity* agent, int start_idx, int end_idx) {
    float max_curv = 0.0f;
    for (int k = start_idx + 1; k < end_idx && k < agent->route_size - 1; k++) {
        float h0 = agent->route_heading[k - 1];
        float h1 = agent->route_heading[k];
        float dx = agent->route_x[k] - agent->route_x[k - 1];
        float dy = agent->route_y[k] - agent->route_y[k - 1];
        float seg_len = sqrtf(dx * dx + dy * dy);
        if (seg_len < 0.01f) continue;
        float curv = fabsf(wrap_angle(h1 - h0)) / seg_len;
        if (curv > max_curv) max_curv = curv;
    }
    return max_curv;
}

// ============================================================================
// V-Max style IDM implementation
// ============================================================================

// --- OBB overlap check using Separating Axis Theorem (2D) ---
static int obb_overlap(float x1, float y1, float len1, float wid1, float yaw1,
                       float x2, float y2, float len2, float wid2, float yaw2) {
    float c1 = cosf(yaw1), s1 = sinf(yaw1);
    float c2 = cosf(yaw2), s2 = sinf(yaw2);
    float hl1 = len1 * 0.5f, hw1 = wid1 * 0.5f;
    float hl2 = len2 * 0.5f, hw2 = wid2 * 0.5f;
    float dx = x2 - x1, dy = y2 - y1;

    // 4 separating axes: 2 edge normals per box
    float axes_x[4] = {c1, -s1, c2, -s2};
    float axes_y[4] = {s1,  c1, s2,  c2};
    for (int a = 0; a < 4; a++) {
        float ax = axes_x[a], ay = axes_y[a];
        float proj1 = hl1 * fabsf(ax * c1 + ay * s1) + hw1 * fabsf(-ax * s1 + ay * c1);
        float proj2 = hl2 * fabsf(ax * c2 + ay * s2) + hw2 * fabsf(-ax * s2 + ay * c2);
        float sep = fabsf(dx * ax + dy * ay);
        if (sep > proj1 + proj2) return 0;  // separating axis found
    }
    return 1;
}

// --- Find leading vehicle via OBB overlap along route (V-Max style) ---
// Places the agent bbox at each route point and checks for overlap with all
// other vehicles. Returns arc-length distance to the first collision point.
static float find_lead_on_route(Drive* env, int agent_idx,
                                float* out_lead_speed, int* out_has_leader) {
    Entity* agent = &env->entities[agent_idx];
    *out_has_leader = 0;
    *out_lead_speed = 0.0f;
    if (agent->route_size < 2) return 0.0f;

    float cum_dist = 0.0f;
    int start = agent->route_progress;

    for (int k = start; k < agent->route_size; k++) {
        if (k > start) {
            float dx = agent->route_x[k] - agent->route_x[k - 1];
            float dy = agent->route_y[k] - agent->route_y[k - 1];
            cum_dist += sqrtf(dx * dx + dy * dy);
        }
        if (cum_dist > 50.0f) break;

        float px = agent->route_x[k];
        float py = agent->route_y[k];
        float yaw = agent->route_heading[k];

        for (int ii = 0; ii < MAX_AGENTS; ii++) {
            int i = -1;
            if (ii < env->active_agent_count)
                i = env->active_agent_indices[ii];
            else if (ii < env->num_actors)
                i = env->static_agent_indices[ii - env->active_agent_count];
            if (i == -1) continue;
            if (i == agent_idx) continue;
            Entity* o = &env->entities[i];
            if (o->type < VEHICLE || o->type > CYCLIST) continue;
            if (o->x == INVALID_POSITION || o->removed) continue;

            // Quick distance filter
            float odx = o->x - px, ody = o->y - py;
            if (odx * odx + ody * ody > 400.0f) continue;

            if (obb_overlap(px, py, agent->length, agent->width, yaw,
                            o->x, o->y, o->length, o->width, o->heading)) {
                *out_has_leader = 1;
                *out_lead_speed = sqrtf(o->vx * o->vx + o->vy * o->vy);
                return cum_dist;
            }
        }
    }
    return 0.0f;
}

// --- V-Max Bezier trajectory + arc-length interpolation ---
// Generates a smooth path from agent position to the route via cubic Bezier,
// then interpolates at new_speed * dt to get the next position.
static void generate_vmax_trajectory(Entity* agent, float new_speed, float dt,
                                     float* out_x, float* out_y) {
    int cp = agent->route_progress;
    if (cp >= agent->route_size - 1) cp = agent->route_size - 2;
    if (cp < 0) cp = 0;

    // Velocity (use heading as fallback when nearly stationary)
    float spd = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    float vx = (spd > 0.01f) ? agent->vx : agent->heading_x * 0.01f;
    float vy = (spd > 0.01f) ? agent->vy : agent->heading_y * 0.01f;
    float vel_norm = sqrtf(vx * vx + vy * vy);
    float vel_dir_x = vx / fmaxf(vel_norm, 0.001f);
    float vel_dir_y = vy / fmaxf(vel_norm, 0.001f);

    // Route direction at closest point
    float route_dir_x = cosf(agent->route_heading[cp]);
    float route_dir_y = sinf(agent->route_heading[cp]);

    // Alignment & curvature (V-Max formulas)
    float alignment = vel_dir_x * route_dir_x + vel_dir_y * route_dir_y;
    float alignment_scaling = 1.0f - 0.5f * alignment;

    int window_end = (cp + 30 < agent->route_size) ? cp + 30 : agent->route_size;
    float max_curvature = compute_max_curvature(agent, cp, window_end);

    float d_merge = (5.0f + 0.5f * new_speed - 0.5f * max_curvature) * alignment_scaling;
    float h_merge = (new_speed > 0.1f) ? d_merge / new_speed : 1.0f;
    if (h_merge < 0.3f) h_merge = 0.3f;
    if (h_merge > 4.0f) h_merge = 4.0f;
    d_merge = h_merge * new_speed;

    // Find blending point on route at distance d_merge
    int blend_idx = find_route_point_at_distance(agent, cp, d_merge);
    float lat_off = agent->idm_lateral_offset;  // perpendicular shift from lane center

    // Apply lateral offset: shift perpendicular to route heading (left = positive)
    float bh = agent->route_heading[blend_idx];
    float blend_x = agent->route_x[blend_idx] + lat_off * (-sinf(bh));
    float blend_y = agent->route_y[blend_idx] + lat_off * cosf(bh);

    // Cubic Bezier control points (V-Max: P1 uses actual velocity, not normalized)
    float P0x = agent->x,                          P0y = agent->y;
    float P1x = P0x + vx * h_merge / 3.0f,         P1y = P0y + vy * h_merge / 3.0f;
    float P3x = blend_x,                            P3y = blend_y;
    float P2x = (2.0f * P3x + P1x) / 3.0f,         P2y = (2.0f * P3y + P1y) / 3.0f;

    // Sample Bezier (20 pts) + remaining route after blend point
    #define VMAX_BEZ_SAMPLES 20
    #define VMAX_MAX_PATH 200
    float path_x[VMAX_MAX_PATH], path_y[VMAX_MAX_PATH];
    int n = 0;

    for (int i = 0; i < VMAX_BEZ_SAMPLES && n < VMAX_MAX_PATH; i++) {
        float a = (float)i / (float)(VMAX_BEZ_SAMPLES - 1);
        float b = 1.0f - a;
        path_x[n] = b*b*b*P0x + 3*b*b*a*P1x + 3*b*a*a*P2x + a*a*a*P3x;
        path_y[n] = b*b*b*P0y + 3*b*b*a*P1y + 3*b*a*a*P2y + a*a*a*P3y;
        n++;
    }
    for (int k = blend_idx; k < agent->route_size && n < VMAX_MAX_PATH; k++) {
        // Apply lateral offset to remaining route points too
        float rh = agent->route_heading[k];
        path_x[n] = agent->route_x[k] + lat_off * (-sinf(rh));
        path_y[n] = agent->route_y[k] + lat_off * cosf(rh);
        n++;
    }

    if (n < 2) { *out_x = agent->x; *out_y = agent->y; return; }

    // Arc-length interpolation at distance = new_speed * dt (V-Max style)
    float target_dist = new_speed * dt;
    if (target_dist <= 0.0f) { *out_x = agent->x; *out_y = agent->y; return; }

    float cum = 0.0f;
    for (int i = 1; i < n; i++) {
        float dx = path_x[i] - path_x[i - 1];
        float dy = path_y[i] - path_y[i - 1];
        float seg = sqrtf(dx * dx + dy * dy);
        if (cum + seg >= target_dist) {
            float frac = (seg > 0.0001f) ? (target_dist - cum) / seg : 0.0f;
            *out_x = path_x[i - 1] + frac * dx;
            *out_y = path_y[i - 1] + frac * dy;
            return;
        }
        cum += seg;
    }
    // target_dist beyond path → last point
    *out_x = path_x[n - 1];
    *out_y = path_y[n - 1];
}

// --- Heading-based leader detection (fallback when no route) ---
Entity* find_leading_vehicle(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    Entity* closest_leader = NULL;
    float min_distance = 100.0f;

    for (int ii = 0; ii < MAX_AGENTS; ii++) {
        int i = -1;
        if (ii < env->active_agent_count)
            i = env->active_agent_indices[ii];
        else if (ii < env->num_actors)
            i = env->static_agent_indices[ii - env->active_agent_count];
        if (i == -1) continue;
        if (i == agent_idx) continue;
        Entity* other = &env->entities[i];
        if (other->type < VEHICLE || other->type > CYCLIST) continue;
        if (other->x == INVALID_POSITION) continue;

        float dx = other->x - agent->x;
        float dy = other->y - agent->y;
        if (dx * dx + dy * dy > 900.0f) continue;  // > 30m

        float dot = dx * agent->heading_x + dy * agent->heading_y;
        if (dot <= 0.0f) continue;

        float lat = fabsf(dx * agent->heading_y - dy * agent->heading_x);
        if (lat < 1.5f && dot < min_distance) {
            min_distance = dot;
            closest_leader = other;
        }
    }
    return closest_leader;
}

// ============================================================================
// move_idm — V-Max faithful IDM with nuPlan parameters
//
// Flow:
//   1. Prevent reverse driving
//   2. Find leading vehicle via OBB overlap along route
//   3. IDM unified acceleration formula (nuPlan parameters)
//   4. Speed update with backward-prevention trick
//   5. V-Max Bezier trajectory + arc-length interpolation → next position
//   6. Update position & heading from displacement (smooth via Bezier)
// ============================================================================
void move_idm(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    if (agent->removed) return;
    if (agent->stopped) { agent->vx = 0; agent->vy = 0; return; }

    // === 1) Current speed — prevent reverse (V-Max) ===
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    float speed_along_heading = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
    if (speed_along_heading < 0.0f) speed = 0.0f;

    // === 2) Find leading vehicle ===
    float lead_speed = 0.0f;
    int has_leader = 0;
    float lead_dist = 0.0f;

    if (agent->route_size >= 2) {
        lead_dist = find_lead_on_route(env, agent_idx, &lead_speed, &has_leader);
    } else {
        // Fallback: heading-based detection
        Entity* leader = find_leading_vehicle(env, agent_idx);
        if (leader) {
            has_leader = 1;
            float dx = leader->x - agent->x;
            float dy = leader->y - agent->y;
            lead_dist = sqrtf(dx * dx + dy * dy)
                      - agent->length * 0.5f - leader->length * 0.5f;
            if (lead_dist < 0.1f) lead_dist = 0.1f;
            lead_speed = sqrtf(leader->vx * leader->vx + leader->vy * leader->vy);
        }
    }

    // === 3) IDM acceleration — unified formula, nuPlan parameters ===
    float desired_velocity = agent->idm_target_velocity;
    const float min_spacing      = env->idm_min_gap;
    const float time_headway     = env->idm_headway_time;
    const float max_accel        = env->idm_accel_max;
    const float max_decel        = env->idm_decel_max;
    const float idm_delta        = 4.0f;
    const float accel_clip       = 6.0f;   // V-Max: MAX_ACCEL_BICYCLE
    const float min_lead_dist    = 0.1f;   // V-Max: _MINIMUM_LEAD_DISTANCE

    float s_star = min_spacing + fmaxf(0.0f,
        speed * time_headway
        + speed * (speed - lead_speed) / (2.0f * sqrtf(max_accel * max_decel)));
    s_star *= (float)has_leader;  // 0 when no leader

    float eff_lead_dist = lead_dist * (float)has_leader;
    if (eff_lead_dist == 0.0f) eff_lead_dist = min_lead_dist;

    float acceleration = max_accel * (
        1.0f
        - powf(speed / fmaxf(desired_velocity, 0.01f), idm_delta)
        - powf(s_star / eff_lead_dist, 2.0f));

    if (acceleration >  accel_clip) acceleration =  accel_clip;
    if (acceleration < -accel_clip) acceleration = -accel_clip;

    // Emergency stop: if a leader was detected but we can't brake in time
    // at comfortable decel_max, bypass the IDM formula and snap speed to 0
    // this step. d_stop = v_rel^2 / (2*decel_max) is the minimum distance
    // required to close the relative speed; if (lead_dist - min_gap) < d_stop
    // we've already passed the point of no return.
    // if (has_leader && speed > 0.1f) {
    //     float v_rel = speed - lead_speed;
    //     if (v_rel > 0.0f) {
    //         float d_stop = (v_rel * v_rel) / (2.0f * max_decel);
    //         float d_avail = lead_dist - min_spacing;
    //         if (d_avail < d_stop) {
    //             acceleration = -speed / env->dt;  // overrides accel_clip intentionally
    //         }
    //     }
    // }

    // === 4) Speed update + backward-prevention trick (V-Max) ===
    float new_speed = speed + acceleration * env->dt;
    if (new_speed < 0.0f) new_speed = 0.0f;
    new_speed = clipSpeed(new_speed);
    acceleration = (new_speed - speed) / env->dt;  // recompute after clamp

    // === 5) Trajectory → next position ===
    if (agent->route_size < 2) {
        // No-route fallback: expert trajectory projection
        float dist_travel = (new_speed + speed) * 0.5f * env->dt;
        float nx = agent->x + dist_travel * agent->heading_x;
        float ny = agent->y + dist_travel * agent->heading_y;

        int ci = -1;
        float md2 = INFINITY;
        for (int t = env->init_steps; t < agent->array_size; t++) {
            if (!agent->traj_valid[t]) continue;
            float dx = agent->traj_x[t] - nx, dy = agent->traj_y[t] - ny;
            float d2 = dx * dx + dy * dy;
            if (d2 < md2) { md2 = d2; ci = t; }
        }
        if (ci < 0 || md2 > 100.0f) { agent->vx = 0; agent->vy = 0; return; }

        int lv = env->init_steps;
        for (int t = agent->array_size - 1; t >= env->init_steps; t--) {
            if (agent->traj_valid[t]) { lv = t; break; }
        }
        if (lv <= env->init_steps + 1) { agent->vx = 0; agent->vy = 0; return; }

        float dxe = agent->traj_x[lv] - nx, dye = agent->traj_y[lv] - ny;
        if (ci >= lv || dxe * dxe + dye * dye < 4.0f) {
            agent->vx = 0; agent->vy = 0; return;
        }
        float h = agent->traj_heading[ci];
        float cx = cosf(h), cy = sinf(h);
        float rx = nx - agent->traj_x[ci], ry = ny - agent->traj_y[ci];
        float p = rx * cx + ry * cy;
        agent->x = agent->traj_x[ci] + p * cx;
        agent->y = agent->traj_y[ci] + p * cy;
        agent->heading = h; agent->heading_x = cx; agent->heading_y = cy;
        agent->vx = new_speed * cx; agent->vy = new_speed * cy;
        return;
    }

    // Update route progress
    int cp = find_closest_route_point(agent, agent->x, agent->y, agent->route_progress);
    if (cp > agent->route_progress) agent->route_progress = cp;

    // End-of-route → stop
    float dx_re = agent->route_x[agent->route_size - 1] - agent->x;
    float dy_re = agent->route_y[agent->route_size - 1] - agent->y;
    if (dx_re * dx_re + dy_re * dy_re < 4.0f) {
        agent->vx = 0; agent->vy = 0; return;
    }

    // Too far from route → freeze
    float dx_cp = agent->route_x[cp] - agent->x;
    float dy_cp = agent->route_y[cp] - agent->y;
    if (dx_cp * dx_cp + dy_cp * dy_cp > 100.0f) {
        agent->vx = 0; agent->vy = 0; return;
    }

    // V-Max Bezier + arc-length interpolation
    float next_x, next_y;
    generate_vmax_trajectory(agent, new_speed, env->dt, &next_x, &next_y);

    // === 6) Update position & heading from displacement (smooth via Bezier) ===
    float dx_mv = next_x - agent->x;
    float dy_mv = next_y - agent->y;
    float mv_dist = sqrtf(dx_mv * dx_mv + dy_mv * dy_mv);

    // NaN guard: if Bezier produced invalid position, freeze agent
    if (isnan(next_x) || isnan(next_y)) {
        agent->vx = 0; agent->vy = 0; return;
    }

    agent->x = next_x;
    agent->y = next_y;

    if (mv_dist > 0.001f) {
        // Heading from trajectory displacement (smooth because Bezier is smooth)
        agent->heading = atan2f(dy_mv, dx_mv);
    } else {
        // Nearly stationary — use route heading to prevent jitter
        int hi = find_closest_route_point(agent, next_x, next_y, agent->route_progress);
        agent->heading = agent->route_heading[hi];
    }
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
    agent->vx = new_speed * agent->heading_x;
    agent->vy = new_speed * agent->heading_y;
}

// ============================================================================

void c_step(Drive *env) {
    /*
    This function generally consists of 5 components:
    1. reset environment, if episode length is reached or termination_mode is set to 1 and all agents got already either respawned, stopped or removed
    2. move expert and active cars + jerk reward
    3. compute rewards for each agent -> does not reset individual agents to not disturb reward calculation for the following agents in the iteration
    4. reset individual agents, if reset behavior is set for collision, offroad or goal behavior.
    5. calculate observation (watch out, if there was termination of the agents, they got reset!)
    */
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));
    // Reset decomposed rewards if allocated (only for cloned CEM envs)
    if (env->collision_rewards) memset(env->collision_rewards, 0, env->active_agent_count * sizeof(float));
    if (env->offroad_rewards) memset(env->offroad_rewards, 0, env->active_agent_count * sizeof(float));
    if (env->goal_rewards) memset(env->goal_rewards, 0, env->active_agent_count * sizeof(float));
    if (env->goal_distances) memset(env->goal_distances, 0, env->active_agent_count * sizeof(float));
    if (env->jerk_rewards) memset(env->jerk_rewards, 0, env->active_agent_count * sizeof(float));
    if (env->lane_distances) memset(env->lane_distances, 0, env->active_agent_count * sizeof(float));
    if (env->lane_alignments) memset(env->lane_alignments, 0, env->active_agent_count * sizeof(float));
    if (env->reward_components) memset(env->reward_components, 0,
        env->active_agent_count * REWARD_COMPONENT_COUNT * sizeof(float));
    if (env->reward_components_raw) memset(env->reward_components_raw, 0,
        env->active_agent_count * REWARD_COMPONENT_COUNT * sizeof(float));
    env->timestep++;

    int originals_remaining = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        // Keep flag true if there is at least one agent that has not been respawned yet
        if ((env->entities[agent_idx].respawn_count + env->entities[agent_idx].stopped + env->entities[agent_idx].removed)  == 0) {
            originals_remaining = 1;
            break;
        }
    }

    if (env->timestep == env->episode_length || (!originals_remaining && env->termination_mode == 1)) {
        add_log(env);
        c_reset(env);
        return;
    }

    // Move static agents (expert replay or IDM, based on per-entity movement_mode or idm_others flag)
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->entities[expert_idx].x == INVALID_POSITION)
            continue;
        if (env->entities[expert_idx].movement_mode == MOVEMENT_IDM || env->idm_others) {
            move_idm(env, expert_idx);
        } else {
            move_expert(env, env->actions, expert_idx);
        }
    }
    // Process actions for all active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        // this only handles truncations at the end of the episode length and not if truncations are caused when all agents are dead.
        if (env->timestep == (env->episode_length - 1)) {
            // this is the last step, afterwards the above if clausal will be triggered
            env->truncations[i] = 1;
        }
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        float prev_vx = env->entities[agent_idx].vx;
        float prev_vy = env->entities[agent_idx].vy;

        if (env->entities[agent_idx].movement_mode == MOVEMENT_IDM) {
            move_idm(env, agent_idx);
        } else if (env->entities[agent_idx].movement_mode == MOVEMENT_EXPERT) {
            move_expert(env, env->actions, agent_idx);
        } else {
            move_dynamics(env, i, agent_idx);
        }

        // Compute acceleration and jerk for classic dynamics (needed for comfort reward)
        if (env->dynamics_model == CLASSIC) {
            float delta_vx = env->entities[agent_idx].vx - prev_vx;
            float delta_vy = env->entities[agent_idx].vy - prev_vy;

            // Decompose into longitudinal/lateral using heading
            float hx = env->entities[agent_idx].heading_x;
            float hy = env->entities[agent_idx].heading_y;
            float prev_a_long = env->entities[agent_idx].a_long;
            float prev_a_lat = env->entities[agent_idx].a_lat;
            float new_a_long = (delta_vx * hx + delta_vy * hy) / env->dt;
            float new_a_lat = (-delta_vx * hy + delta_vy * hx) / env->dt;
            env->entities[agent_idx].jerk_long = (new_a_long - prev_a_long) / env->dt;
            env->entities[agent_idx].jerk_lat = (new_a_lat - prev_a_lat) / env->dt;
            env->entities[agent_idx].a_long = new_a_long;
            env->entities[agent_idx].a_lat = new_a_lat;

            // Tiny jerk penalty for smoothness (legacy). Gate on config flag.
            if (env->reward_jerk_legacy != 0.0f) {
                float jerk_penalty = env->reward_jerk_legacy * sqrtf(delta_vx * delta_vx + delta_vy * delta_vy) / env->dt;
                env->rewards[i] += jerk_penalty;
                env->logs[i].episode_return += jerk_penalty;
                // Decomposed reward for CEM
                if (env->jerk_rewards) env->jerk_rewards[i] = jerk_penalty;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_JERK_LEGACY] += jerk_penalty;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_JERK_LEGACY] += sqrtf(delta_vx * delta_vx + delta_vy * delta_vy) / env->dt;
            }
        }
    }

    // Compute rewards and only rewards!! Do not replace agents here, to not break the correct reward calculation for the following agents in the iteration.
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        if (env->entities[agent_idx].removed == 1 || env->entities[agent_idx].stopped == 1) { // if the agents got removed/stopped in a step before, they should not get any reward in the upfollowing step, continue here.
            env->terminals[i] = 1;
            continue;
        }
        compute_agent_metrics(env, agent_idx, i); // sets collision_state/stopped/removed flags for current agent for collision and offroad, not for goal_behavior!
        int collision_state = env->entities[agent_idx].collision_state;

        float current_speed = sqrtf(env->entities[agent_idx].vx * env->entities[agent_idx].vx +
                                    env->entities[agent_idx].vy * env->entities[agent_idx].vy);

        // Resolve reward coefficients — per-agent when reward_conditioning is on, else global.
        // Stored positive; applied with sign at the reward formula.
        Entity *cur_agent = &env->entities[agent_idx];
        float alpha_collision = env->reward_conditioning ? cur_agent->creward_alpha_collision
                                                         : -env->reward_vehicle_collision;
        float alpha_boundary  = env->reward_conditioning ? cur_agent->creward_alpha_boundary
                                                         : -env->reward_offroad_collision;

        // is this fair, if the agents get stopped at vehicle collision or going offroad? Do they get negative reward for each timestep afterwards??
        if (collision_state > 0) {
            if (collision_state == VEHICLE_COLLISION) {
                // Paper: R_collision = -(alpha_collision + 0.1*|v|) * 1_collision
                float crash_penalty = -alpha_collision - 0.1f * current_speed;
                env->rewards[i] += crash_penalty;
                env->logs[i].episode_return += crash_penalty;
                env->logs[i].collision_rate = 1.0f;
                env->logs[i].collisions_per_agent += 1.0f;
                if (env->collision_rewards) env->collision_rewards[i] = crash_penalty;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_COLLISION] += crash_penalty;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_COLLISION] += 1.0f;
            } else if (collision_state == OFFROAD) {
                float offroad_penalty = -alpha_boundary;
                env->rewards[i] += offroad_penalty;
                env->logs[i].episode_return += offroad_penalty;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].offroad_per_agent += 1.0f;
                // Decomposed reward for CEM
                if (env->offroad_rewards) env->offroad_rewards[i] = offroad_penalty;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_OFFROAD] += offroad_penalty;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_OFFROAD] += 1.0f;
            }

            env->entities[agent_idx].collided_before_goal = 1;
        }

        float distance_to_goal =
            relative_distance_2d(env->entities[agent_idx].x, env->entities[agent_idx].y,
                                 env->entities[agent_idx].goal_position_x, env->entities[agent_idx].goal_position_y);

        // Store goal distance for CEM (if allocated)
        if (env->goal_distances) env->goal_distances[i] = distance_to_goal;

        // Speed limit penalty (hardcoded 15 m/s)
        if (current_speed > 15.0f) {
            env->logs[i].speed_limit_rate = 1.0f;
            // Raw counts speeding steps regardless of whether reward_speed_limit
            // is set, so behavior is always observable.
            if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_SPEED_LIMIT] += 1.0f;
            if (env->reward_speed_limit != 0.0f) {
                env->rewards[i] += env->reward_speed_limit;
                env->logs[i].episode_return += env->reward_speed_limit;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_SPEED_LIMIT] += env->reward_speed_limit;
            }
        }

        // Reward agent if it is within X meters of goal and speed is below threshold
        // Goal-reach threshold. In training (creward_deterministic=0) it
        // tracks the per-agent sampled creward_delta_goal so the policy is
        // supervised consistently with its conditioning. In deterministic
        // eval (=1) we decouple: creward_delta_goal still drives the
        // conditioning obs, but the actual reach check uses env->goal_radius
        // so sweeping the conditioning doesn't change scenario difficulty.
        float eff_goal_radius = (env->reward_conditioning && !env->creward_deterministic)
            ? cur_agent->creward_delta_goal
            : env->goal_radius;
        bool within_distance = distance_to_goal < eff_goal_radius;
        // During training (creward_deterministic=0) the per-agent sampled
        // creward_goal_speed drives the goal-reach speed cap so the policy is
        // supervised consistently with its conditioning. In deterministic eval
        // (=1) we decouple: creward_goal_speed still feeds the conditioning
        // obs (line 2403) but the actual reach check uses env->goal_speed so
        // scenario difficulty stays the same across agents/conditionings.
        // Mirrors the eff_goal_radius logic above.
        float eff_goal_speed = (env->reward_conditioning && !env->creward_deterministic)
            ? cur_agent->creward_goal_speed
            : env->goal_speed;
        bool within_speed = current_speed <= eff_goal_speed;

        if (within_distance && within_speed && !env->entities[agent_idx].current_goal_reached) {
            if (env->goal_behavior == GOAL_RESPAWN){
                if (env->entities[agent_idx].respawn_timestep != -1) {
                    // if there already was a respawn, there is a discounted goal reward
                    env->rewards[i] += env->reward_goal_post_respawn;
                    env->logs[i].episode_return += env->reward_goal_post_respawn;
                    if (env->goal_rewards) env->goal_rewards[i] = env->reward_goal_post_respawn;
                    if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] += env->reward_goal_post_respawn;
                    if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                    env->entities[agent_idx].current_goal_reached = 1; // does this mean, this can only happen once?
                    env->entities[agent_idx].goals_reached_this_episode += 1.0f;
                } else {
                    // general respawn reward
                    env->rewards[i] += env->reward_goal;
                    env->logs[i].episode_return += env->reward_goal;
                    if (env->goal_rewards) env->goal_rewards[i] = env->reward_goal;
                    if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] += env->reward_goal;
                    if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                    env->entities[agent_idx].goals_reached_this_episode += 1.0f;
                }
            // if (env->goal_behavior == GOAL_RESPAWN && env->entities[agent_idx].respawn_timestep != -1) {
            //     env->rewards[i] += env->reward_goal_post_respawn;
            //     env->logs[i].episode_return += env->reward_goal_post_respawn;
            //     env->entities[agent_idx].current_goal_reached = 1;
            } else if (env->goal_behavior == GOAL_GENERATE_NEW) { // always same reward for each reached goal -> is this intended? rather also the discounted reward?
                env->rewards[i] += env->reward_goal;
                env->logs[i].episode_return += env->reward_goal;
                if (env->goal_rewards) env->goal_rewards[i] = env->reward_goal;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] += env->reward_goal;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                sample_new_goal(env, agent_idx);
                env->entities[agent_idx].current_goal_reached = 0;
                env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            } else if (env->goal_behavior == GOAL_SAMPLE_LANE_AHEAD) {
                float r_goal = (env->entities[agent_idx].goals_reached_this_episode > 0.0f)
                    ? env->reward_goal_post_respawn
                    : env->reward_goal;
                env->rewards[i] += r_goal;
                env->logs[i].episode_return += r_goal;
                if (env->goal_rewards) env->goal_rewards[i] = r_goal;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] += r_goal;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                sample_new_goal_on_lane(env, agent_idx);
                env->entities[agent_idx].current_goal_reached = 0;
                env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            } else if (env->goal_behavior == GOAL_REMOVE) {
                if (env->entities[agent_idx].removed != 1) { // if removed through collision or offroad just before.
                    // printf("Agent %d reached goal and will be removed from the environment.\n", agent_idx);
                    env->rewards[i] += env->reward_goal;
                    env->logs[i].episode_return += env->reward_goal;
                    if (env->goal_rewards) env->goal_rewards[i] = env->reward_goal;
                    if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] += env->reward_goal;
                    if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                    env->entities[agent_idx].removed = 1;
                    // env->terminals[i] = 1;
                    // env->entities[agent_idx].x = env->entities[agent_idx].y = -10000.0f;
                    env->entities[agent_idx].goals_reached_this_episode += 1.0f;
                }
            } else if (env->goal_behavior == GOAL_STOP) {
                if (env->entities[agent_idx].stopped != 1) {
                    // printf("Agent %d reached goal and will stop moving.\n", agent_idx);
                    env->rewards[i] = env->reward_goal;
                    env->logs[i].episode_return = env->reward_goal;
                    if (env->goal_rewards) env->goal_rewards[i] = env->reward_goal;
                    // GOAL_STOP uses `=` (overwrite); mirror by clearing this
                    // agent's row then setting RC_GOAL, so sum(components) still
                    // equals env->rewards[i] for this step.
                    if (env->reward_components) {
                        memset(&env->reward_components[i * REWARD_COMPONENT_COUNT], 0,
                               REWARD_COMPONENT_COUNT * sizeof(float));
                        env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] = env->reward_goal;
                    }
                    // Raw: GOAL_STOP overwrites reward; raw is a pure behavior
                    // count so it just increments (no reset).
                    if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                    env->entities[agent_idx].stopped = 1;
                    // env->terminals[i] = 1;
                    // env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
                    env->entities[agent_idx].goals_reached_this_episode += 1.0f;
                }

            } else if (env->goal_behavior == GOAL_CONTINUE) {
                // Reward for reaching goal but keep driving (no stop/remove/respawn)
                env->rewards[i] += env->reward_goal;
                env->logs[i].episode_return += env->reward_goal;
                if (env->goal_rewards) env->goal_rewards[i] = env->reward_goal;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_GOAL] += env->reward_goal;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_GOAL] += 1.0f;
                env->entities[agent_idx].current_goal_reached = 1;
                env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            } else {
                printf("Unknown goal behavior.\n");
                // printf("Something really goes wrong here, goal behavior of agent %d is %d.\n", agent_idx, env->goal_behavior);
                // env->rewards[i] = env->reward_goal;
                // env->logs[i].episode_return = env->reward_goal;
                // env->entities[agent_idx].stopped = 1;
                // env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
                // env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            }

            // stopped nur, wenn wirklich stopped
            // respawn_timestep == -1 -> nochmal anderes if drumrum

            env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
            env->logs[i].speed_at_goal = current_speed;
        }

        int lane_aligned = env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX];
        env->logs[i].lane_alignment_rate = lane_aligned;
        env->logs[i].lane_aligned_steps += (float)lane_aligned;

        // Lane alignment reward (per-step bonus when aligned with lane heading)
        if (env->reward_lane_alignment != 0.0f && lane_aligned) {
            env->rewards[i] += env->reward_lane_alignment;
            env->logs[i].episode_return += env->reward_lane_alignment;
            // Legacy term — bucket under RC_L_ALIGN alongside the paper's R_l-align.
            if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_L_ALIGN] += env->reward_lane_alignment;
            // Raw: 1 per aligned step, only when the legacy term is enabled
            // (keeps the standard eval path where reward_lane_alignment=0 undisturbed).
            if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_L_ALIGN] += 1.0f;
        }

        // Lane distance: always track metric, optionally apply reward
        float lane_dist = env->entities[agent_idx].metrics_array[LANE_DISTANCE_IDX];
        env->logs[i].lane_distance_avg += lane_dist;
        env->logs[i].lane_distance_count += 1.0f;

        if (env->reward_lane_distance != 0.0f) {
            float lane_dist_penalty = env->reward_lane_distance * lane_dist;
            env->rewards[i] += lane_dist_penalty;
            env->logs[i].episode_return += lane_dist_penalty;
            // Legacy term — bucket under RC_L_CENTER alongside the paper's R_l-center.
            if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_L_CENTER] += lane_dist_penalty;
            // Raw: bare lane_dist (meters) per step, only when legacy term is enabled.
            if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_L_CENTER] += lane_dist;
        }

        // Velocity reward: α_velocity * Δt * max(cos(θ_f), 0) * 1_{|v|>2.5}
        // θ_f = angle between agent heading and closest lane direction
        if (env->reward_velocity != 0.0f && current_speed > 2.5f) {
            int lane_idx = env->entities[agent_idx].current_lane_idx;
            if (lane_idx >= 0) {
                Entity *lane = &env->entities[lane_idx];
                // Find closest geometry segment for heading
                float best_d = FLT_MAX;
                int best_g = 0;
                for (int g = 0; g < lane->array_size - 1; g++) {
                    float dx = env->entities[agent_idx].x - lane->traj_x[g];
                    float dy = env->entities[agent_idx].y - lane->traj_y[g];
                    float d = dx*dx + dy*dy;
                    if (d < best_d) { best_d = d; best_g = g; }
                }
                float lx = lane->traj_x[best_g + 1] - lane->traj_x[best_g];
                float ly = lane->traj_y[best_g + 1] - lane->traj_y[best_g];
                float lane_heading = atan2f(ly, lx);
                float theta_f = env->entities[agent_idx].heading - lane_heading;
                // Normalize to [-pi, pi]
                if (theta_f > M_PI) theta_f -= 2.0f * M_PI;
                if (theta_f < -M_PI) theta_f += 2.0f * M_PI;
                float cos_theta = cosf(theta_f);
                if (cos_theta < 0.0f) cos_theta = 0.0f;
                float vel_reward = env->reward_velocity * env->dt * cos_theta;
                env->rewards[i] += vel_reward;
                env->logs[i].episode_return += vel_reward;
                env->logs[i].velocity_reward_total += vel_reward;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_VELOCITY] += vel_reward;
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_VELOCITY] += env->dt * cos_theta;
            }
        }

        // Resolve the remaining per-agent α coefficients (conditioning or global).
        float alpha_comfort     = env->reward_conditioning ? cur_agent->creward_alpha_comfort
                                                           : env->reward_comfort;
        float alpha_l_align     = env->reward_conditioning ? cur_agent->creward_alpha_l_align
                                                           : env->reward_l_align;
        float alpha_vel_align   = env->reward_conditioning ? cur_agent->creward_alpha_vel_align
                                                           : env->reward_l_align_vel;
        float alpha_l_center    = env->reward_conditioning ? cur_agent->creward_alpha_l_center
                                                           : env->reward_l_center;
        float alpha_center_bias = env->reward_conditioning ? cur_agent->creward_alpha_center_bias
                                                           : env->reward_l_center_bias;
        float alpha_reverse     = env->reward_conditioning ? cur_agent->creward_alpha_reverse
                                                           : env->reward_reverse;

        // R_comfort = -alpha * (1_{|a_long|>3} + 1_{|a_lat|>3} + 1_{|jerk_long|>5 or |jerk_lat|>5})
        {
            float violations = 0.0f;
            if (fabsf(env->entities[agent_idx].a_long) > 3.0f) violations += 1.0f;
            if (fabsf(env->entities[agent_idx].a_lat) > 3.0f) violations += 1.0f;
            if (fabsf(env->entities[agent_idx].jerk_long) > 5.0f || fabsf(env->entities[agent_idx].jerk_lat) > 5.0f)
                violations += 1.0f;
            env->logs[i].comfort_violations += violations;
            // Raw: violation count regardless of α, so behavior is always visible.
            if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_COMFORT] += violations;
            if (alpha_comfort != 0.0f && violations > 0.0f) {
                float comfort_penalty = -alpha_comfort * violations;
                env->rewards[i] += comfort_penalty;
                env->logs[i].episode_return += comfort_penalty;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_COMFORT] += comfort_penalty;
            }
        }

        float theta_f = env->entities[agent_idx].lane_heading_diff;
        float cos_tf = cosf(theta_f);
        float x_f = env->entities[agent_idx].lane_lateral_dist;

        // R_l-align = alpha * dt * (min(cos(theta_f),0) + alpha_vel*min(cos(theta_f)*v,0) + 0.0025*(1-|theta_f|/(pi/2)))
        // Raw: drop BOTH α_l_align and α_vel_align so the metric is purely
        // behavior-driven (signed lane-alignment integral over time).
        {
            float term1 = cos_tf < 0.0f ? cos_tf : 0.0f;
            float term2_raw = (cos_tf * current_speed) < 0.0f ? (cos_tf * current_speed) : 0.0f;
            float term3 = 0.0025f * (1.0f - fabsf(theta_f) / (M_PI / 2.0f));
            if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_L_ALIGN] += env->dt * (term1 + term2_raw + term3);
            if (alpha_l_align != 0.0f) {
                float term2 = alpha_vel_align * term2_raw;
                float align_reward = alpha_l_align * env->dt * (term1 + term2 + term3);
                env->rewards[i] += align_reward;
                env->logs[i].episode_return += align_reward;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_L_ALIGN] += align_reward;
            }
        }

        // R_l-center = -alpha * dt * (1_{cos(theta_f)>0.5} * |x_f - bias| - 0.05/exp(|x_f - bias|-0.5))
        if (env->entities[agent_idx].current_lane_idx >= 0) {
            float dx = fabsf(x_f - alpha_center_bias);
            float center_term = (cos_tf > 0.5f) ? dx : 0.0f;
            float attract = 0.05f / expf(dx - 0.5f);
            // Raw: α-less formula body, always written (behavior visible even when α=0).
            if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_L_CENTER] += env->dt * (center_term - attract);
            if (alpha_l_center != 0.0f) {
                float center_penalty = -alpha_l_center * env->dt * (center_term - attract);
                env->rewards[i] += center_penalty;
                env->logs[i].episode_return += center_penalty;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_L_CENTER] += center_penalty;
            }
        }

        // R_timestep = -(alpha * dt) * 1_{|v|>0 or |a|>0}
        {
            float a_mag = fabsf(env->entities[agent_idx].a_long) + fabsf(env->entities[agent_idx].a_lat);
            int active = (current_speed > 0.0f || a_mag > 0.0f);
            if (active && env->reward_components_raw) {
                env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_TIMESTEP] += env->dt;
            }
            if (env->reward_timestep != 0.0f && active) {
                float ts_penalty = -(env->reward_timestep * env->dt);
                env->rewards[i] += ts_penalty;
                env->logs[i].episode_return += ts_penalty;
                if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_TIMESTEP] += ts_penalty;
            }
        }

        // R_reverse = -alpha * dt * 1_{v_long<0}
        {
            float v_long = env->entities[agent_idx].vx * env->entities[agent_idx].heading_x
                         + env->entities[agent_idx].vy * env->entities[agent_idx].heading_y;
            if (v_long < 0.0f) {
                if (env->reward_components_raw) env->reward_components_raw[i * REWARD_COMPONENT_COUNT + RC_REVERSE] += env->dt;
                if (alpha_reverse != 0.0f) {
                    float rev_penalty = -alpha_reverse * env->dt;
                    env->rewards[i] += rev_penalty;
                    env->logs[i].episode_return += rev_penalty;
                    if (env->reward_components) env->reward_components[i * REWARD_COMPONENT_COUNT + RC_REVERSE] += rev_penalty;
                }
            }
        }
    }

    // reset agents if respawned, reset velocity or position if stopped/removed.
    if (env->goal_behavior == GOAL_RESPAWN) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                respawn_agent(env, agent_idx);
                env->entities[agent_idx].respawn_count++;
                env->terminals[i] = 1; // terminated in this context, since it got respawned afterwards and values are not connected anymore!
            }
        }


    }
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->entities[agent_idx].stopped == 1){
            env->terminals[i] = 1;
            env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
        } else if (env->entities[agent_idx].removed == 1){
            env->terminals[i] = 1;
            env->entities[agent_idx].x = env->entities[agent_idx].y = -10000.0f;
        }
    }
    // termination_mode==1: truncate scene in the step afterwards, if all agents either got respawned, removed or stopped
    if (env->termination_mode == 1){
        int all_agents_respawned_stopped_removed = 1;
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if ((env->entities[agent_idx].respawn_count + env->entities[agent_idx].stopped + env->entities[agent_idx].removed) == 0) {
                all_agents_respawned_stopped_removed = 0;
                break;
            }
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            if (all_agents_respawned_stopped_removed == 1) {
                env->truncations[i] = 1;
            }
        }
    }


    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    Model cyclist;
    Model pedestrian;
    ModelAnimation *cycle_anim;
    int car_assignments[MAX_AGENTS]; // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client *make_client(Drive *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferDrive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");
    int animCountCyc = 0;
    client->cycle_anim = LoadModelAnimations("resources/drive/cyclist.glb", &animCountCyc);
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }
    // Get initial target position from first active agent
    Vector3 target_pos = {
        0,
        0, // Y is up
        1  // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,      // Same X as target
        120.0f, // 20 units above target
        175.0f  // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client *client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
                         -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {client->camera.position.x - client->camera.target.x,
                             client->camera.position.y - client->camera.target.y,
                             client->camera.position.z - client->camera.target.z};

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive *env, int agent_index, int mode, int obs_only, int lasers) {
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid
    if (mode == 0) {
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
        return;
    }

    int ego_dim = (env->dynamics_model == JERK || env->emit_jerk_ego_obs) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int extra = env->include_global_state ? GLOBAL_STATE_FEATURES : 0;
    int creward_dim = env->reward_conditioning ? CREWARD_FEATURES : 0;
    int max_obs = ego_dim + PARTNER_FEATURES * env->max_obs_partners + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS + creward_dim + extra;
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;
    if (mode == 0) {
        DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x, goal_y, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }

    if (mode == 1) {
        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < env->max_obs_partners; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += PARTNER_FEATURES; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, 1}, (Vector3){partner_x, partner_y, 1}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), 1},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), 1},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), 1},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), 1},
        };

        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], ORANGE);
            }
        }

        if (mode == 1) {
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 2.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, 0.0}, (Vector3){arrow_x, arrow_y, 0.0}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, 1}, (Vector3){arrow_x_world, arrow_y_world, 1}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.3f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx * arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy * arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx * arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy * arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end1, arrow_y_end1, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end2, arrow_y_end2, 0.0}, PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, 0.0}, PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES; // Move to next agent observation
    }
    // Then draw map observations
    int map_start_idx = ego_dim + PARTNER_FEATURES * env->max_obs_partners; // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {          // Loop through potential map entities
        int entity_idx = map_start_idx + k * 7;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE) {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, 1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb
    float road_z = 0.0f;      // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {end_x - start_x, end_y - start_y, 0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {direction.x / length, direction.y / length, 0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, road_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, road_z};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {

    if (show_grid) {
        float grid_start_x = env->grid_map->top_left_x;
        float grid_start_y = env->grid_map->bottom_right_y;
        for (int i = 0; i < env->grid_map->grid_cols; i++) {
            for (int j = 0; j < env->grid_map->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 0.0f}, GRID_CELL_SIZE,
                              GRID_CELL_SIZE, 0.1f, Fade(PUFF_BACKGROUND2, 0.3f));
            }
        }
    }

    // Draw a grid to help with orientation
    for (int i = 0; i < env->num_entities; i++) {
        // Draw objects
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }
            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if ((!is_active_agent && !is_static_agent) || env->entities[i].respawn_timestep != -1) {
                continue;
            }
            Vector3 position;
            float heading;
            position = (Vector3){env->entities[i].x, env->entities[i].y, 1.1};
            heading = env->entities[i].heading;
            // Create size vector
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);

            // Save current transform
            if (mode == 1) {
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){position.x + (half_len * cos_heading - half_width * sin_heading),
                              position.y + (half_len * sin_heading + half_width * cos_heading), position.z},
                    (Vector3){position.x + (half_len * cos_heading + half_width * sin_heading),
                              position.y + (half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading + half_width * sin_heading),
                              position.y + (-half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading - half_width * sin_heading),
                              position.y + (-half_len * sin_heading + half_width * cos_heading), position.z},

                };

                if (agent_index == env->human_agent_idx &&
                    !env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                    continue;
                }

                // --- Draw the car  ---
                Color car_color = GRAY; // default for static
                if (is_expert)
                    car_color = GOLD; // expert replay
                if (is_active_agent)
                    car_color = BLUE; // policy-controlled
                if (is_active_agent && env->entities[i].collision_state > 0)
                    car_color = RED;
                rlSetLineWidth(3.0f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(corners[j], corners[(j + 1) % 4], car_color);
                }
                // --- Draw a heading arrow pointing forward ---
                Vector3 arrowStart = position;
                Vector3 arrowEnd = {position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                                    position.y + sin_heading * half_len * 1.5f, position.z};

                DrawLine3D(arrowStart, arrowEnd, car_color);
                DrawSphere(arrowEnd, 0.2f, car_color); // arrow tip

            } else { // Agent view
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees

                // Select car model (skip index 0)
                Model car_model = client->cars[(i % 5) + 1]; // Cycles through indices 1-5

                if (agent_index == env->human_agent_idx) {
                    car_model = client->cars[0]; // Ego agent always uses red car
                } else if (is_active_agent) {

                    car_model = client->cars[(i % 5) + 1];

                    if (env->entities[i].collision_state > 0) {
                        car_model = client->cars[0]; // Collided agents use red
                    }
                }
                // Draw obs for selected agent index
                if (agent_index == env->human_agent_idx &&
                    (!env->entities[agent_index].metrics_array[REACHED_GOAL_IDX] ||
                     env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_STOP ||
                     env->goal_behavior == GOAL_SAMPLE_LANE_AHEAD)) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions
                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                      bounds.max.z - bounds.min.z};
                Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
                // if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                //     rlPopMatrix();
                //     continue;
                // }
                if (env->entities[i].type == CYCLIST) {
                    scale = (Vector3){0.01, 0.01, 0.01};
                    car_model = client->cyclist;
                }
                if (env->entities[i].type == PEDESTRIAN) {
                    scale = (Vector3){2, 2, 2};
                    car_model = client->pedestrian;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){half_len, -half_width, 0},  // Front-left
                        (Vector3){half_len, half_width, 0},   // Front-right
                        (Vector3){-half_len, half_width, 0},  // Back-right
                        (Vector3){-half_len, -half_width, 0}, // Back-left
                    };
                    Color wire_color = GRAY; // static
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1)
                        wire_color = GOLD; // expert replay
                    if (is_active_agent)
                        wire_color = BLUE; // policy
                    if (is_active_agent && env->entities[i].collision_state > 0)
                        wire_color = RED;
                    rlSetLineWidth(2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j + 1) % 4], wire_color);
                    }
                }
                rlPopMatrix();
            }

            // FPV Camera Control
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                Vector3 camera_position = (Vector3){position.x - (25.0f * cosf(heading)),
                                                    position.y - (25.0f * sinf(heading)), position.z + 15};

                Vector3 camera_target = (Vector3){position.x + 40.0f * cosf(heading),
                                                  position.y + 40.0f * sinf(heading), position.z - 5.0f};
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw route polyline for all agents that have one (IDM agents)
            if (env->entities[i].route_size >= 2 && !IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                rlSetLineWidth(3.0f);
                Color route_color = (i == env->active_agent_indices[env->human_agent_idx])
                    ? (Color){0, 255, 0, 230}   // bright green for ego
                    : (Color){0, 200, 255, 180}; // cyan for others
                for (int r = 0; r < env->entities[i].route_size - 1; r++) {
                    Vector3 rstart = {env->entities[i].route_x[r], env->entities[i].route_y[r], 2.0f};
                    Vector3 rend = {env->entities[i].route_x[r+1], env->entities[i].route_y[r+1], 2.0f};
                    DrawLine3D(rstart, rend, route_color);
                }
            }
            // Draw goal position for active agents
            if (!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 1}, 0.5f,
                           DARKGREEN);

                DrawCircle3D((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 0.1f},
                             env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.9f));
            }
        }
        // Draw road elements
        if (env->entities[i].type <= 3 && env->entities[i].type >= 7) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], 1};
            Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], 1};
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE)
                lineColor = Fade(SOFT_YELLOW, 0.25f);
            else if (env->entities[i].type == ROAD_LINE)
                lineColor = WHITE;
            else if (env->entities[i].type == ROAD_EDGE)
                lineColor = WHITE;
            else if (env->entities[i].type == DRIVEWAY)
                lineColor = RED;

            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                if (env->entities[i].type == ROAD_EDGE) {
                    draw_road_edge(env, start.x, start.y, end.x, end.y);
                } else if (env->entities[i].type == ROAD_LANE || env->entities[i].type == ROAD_LINE) {
                    // Draw road lanes and lines as purple lines
                    rlSetLineWidth(2.0f);
                    DrawLine3D(start, end, lineColor);
                }
            }
        }
    }

    EndMode3D();

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_WOSAC) {
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->height / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            // Ignore respawned agents
            if (env->entities[i].respawn_timestep != -1) {
                continue;
            }
            int agent_idx = env->active_agent_indices[i];
            int womd_track_idx = env->tracks_to_predict_indices[i];

            float raw_x = -env->entities[agent_idx].x * pixels_per_world_unit;
            float raw_y = env->entities[agent_idx].y * pixels_per_world_unit;

            int screen_x = (int)raw_x + client->width / 2 + 20;
            int screen_y = (int)raw_y + client->height / 2 - 25;

            if (screen_x >= 0 && screen_x <= client->width && screen_y >= 0 && screen_y <= client->height) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width / 2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
}

void c_render(Drive *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client *client = env->client;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);
    draw_scene(env, client, 0, 0, 0, 0);

    if (IsKeyPressed(KEY_TAB)) {
        env->human_agent_idx = (env->human_agent_idx + 1) % env->active_agent_count;
    }

    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)", client->camera.position.x, client->camera.position.y,
                        client->camera.position.z),
             10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)", client->camera.target.x, client->camera.target.y,
                        client->camera.target.z),
             10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);

    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);

    // Display current action values - yellow when controlling, white otherwise
    Color action_color = IsKeyDown(KEY_LEFT_SHIFT) ? YELLOW : PUFF_WHITE;

    if (env->action_type == 0) { // discrete
        int *action_array = (int *)env->actions;
        int action_val = action_array[env->human_agent_idx];

        if (env->dynamics_model == CLASSIC) {
            int num_steer = 13;
            int accel_idx = action_val / num_steer;
            int steer_idx = action_val % num_steer;
            float accel_value = ACCELERATION_VALUES[accel_idx];
            float steer_value = STEERING_VALUES[steer_idx];

            DrawText(TextFormat("Acceleration: %.2f m/s^2", accel_value), 10, 110, 20, action_color);
            DrawText(TextFormat("Steering: %.3f", steer_value), 10, 130, 20, action_color);
        } else if (env->dynamics_model == JERK) {
            int num_lat = 3;
            int jerk_long_idx = action_val / num_lat;
            int jerk_lat_idx = action_val % num_lat;
            float jerk_long_value = JERK_LONG[jerk_long_idx];
            float jerk_lat_value = JERK_LAT[jerk_lat_idx];

            DrawText(TextFormat("Longitudinal Jerk: %.2f m/s^3", jerk_long_value), 10, 110, 20, action_color);
            DrawText(TextFormat("Lateral Jerk: %.2f m/s^3", jerk_lat_value), 10, 130, 20, action_color);
        }
    } else { // continuous
        float (*action_array_f)[2] = (float (*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20, action_color);
        DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, action_color);
    }

    // Show key press status
    int status_y = 150;
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        DrawText("[shift pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }
    if (IsKeyDown(KEY_SPACE)) {
        DrawText("[space pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }
    if (IsKeyDown(KEY_LEFT_CONTROL)) {
        DrawText("[ctrl pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }

    // Controls help
    DrawText("Controls: SHIFT + W/S - Accelerate/Brake, SHIFT + A/D - Steer, TAB - Switch Agent", 10,
             client->height - 30, 20, PUFF_WHITE);

    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, status_y, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, status_y + 20, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client *client) {
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

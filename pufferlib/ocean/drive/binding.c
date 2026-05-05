/*
 * Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
 * SPDX-License-Identifier: AGPL-3.0
 *
 * This source code is derived from PufferDrive V2.0
 * (https://github.com/Emerge-Lab/PufferDrive/)
 * Copyright (c) 2026 PufferDrive, licensed under the MIT license.
 */

#include <Python.h>
#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#define MY_GET

// Forward declarations for snapshot functions
static PyObject* vec_create_snapshot(PyObject* self, PyObject* args);
static PyObject* vec_restore_snapshot(PyObject* self, PyObject* args);
static PyObject* vec_restore_snapshot_broadcast(PyObject* self, PyObject* args);
static PyObject* vec_clone_from_env(PyObject* self, PyObject* args);
static PyObject* vec_free_snapshot(PyObject* self, PyObject* args);
static PyObject* vec_get_ego_positions(PyObject* self, PyObject* args);
static PyObject* vec_get_agent_log(PyObject* self, PyObject* args);
static PyObject* vec_set_movement_mode(PyObject* self, PyObject* args);
static PyObject* vec_set_idm_proposals(PyObject* self, PyObject* args);
static PyObject* vec_set_idm_target_velocity(PyObject* self, PyObject* args);
static PyObject* vec_get_ego_state(PyObject* self, PyObject* args);
static PyObject* vec_set_idm_params(PyObject* self, PyObject* args);
static PyObject* vec_set_goals(PyObject* self, PyObject* args);
static PyObject* vec_set_positions(PyObject* self, PyObject* args);
// Forward declarations for observation-only env
static PyObject* init_obs_env(PyObject* self, PyObject* args);
static PyObject* compute_obs_external(PyObject* self, PyObject* args);

// Define MY_METHODS to include snapshot functions in the method table
#define MY_METHODS \
    {"vec_create_snapshot", vec_create_snapshot, METH_VARARGS, "Create snapshots for all environments in a VecEnv"}, \
    {"vec_restore_snapshot", vec_restore_snapshot, METH_VARARGS, "Restore snapshots for all environments in a VecEnv"}, \
    {"vec_restore_snapshot_broadcast", vec_restore_snapshot_broadcast, METH_VARARGS, "Restore one snapshot into all environments in a VecEnv"}, \
    {"vec_clone_from_env", vec_clone_from_env, METH_VARARGS, "Clone a single environment into a VecEnv batch"}, \
    {"vec_free_snapshot", vec_free_snapshot, METH_VARARGS, "Free a list of snapshots"}, \
    {"vec_get_ego_positions", vec_get_ego_positions, METH_VARARGS, "Get ego agent positions for all environments"}, \
    {"vec_get_agent_log", vec_get_agent_log, METH_VARARGS, "Get per-agent logs for a specific agent"}, \
    {"vec_set_movement_mode", vec_set_movement_mode, METH_VARARGS, "Set movement mode for agents (0=dynamics, 1=IDM)"}, \
    {"vec_set_idm_proposals", vec_set_idm_proposals, METH_VARARGS, "Set IDM mode with per-env velocity and lateral offset"}, \
    {"vec_set_idm_target_velocity", vec_set_idm_target_velocity, METH_VARARGS, "Set IDM target velocity for given agents"}, \
    {"vec_get_ego_state", vec_get_ego_state, METH_VARARGS, "Get ego agent speed and heading for all envs"}, \
    {"vec_set_idm_params", vec_set_idm_params, METH_VARARGS, "Set IDM parameters (min_gap, headway_time, accel_max, decel_max) on all sub-envs"}, \
    {"vec_set_goals", vec_set_goals, METH_VARARGS, "Set goal positions for specific agents"}, \
    {"vec_set_positions", vec_set_positions, METH_VARARGS, "Set positions and headings for specific agents"}, \
    {"init_obs_env", init_obs_env, METH_VARARGS, "Init lightweight env for observation computation from .bin"}, \
    {"compute_obs_external", compute_obs_external, METH_VARARGS, "Compute observations with external ego/agent state"}

#include "../env_binding.h"

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);

    PyObject *trunc = PyDict_GetItemString(kwargs, "truncations");
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return 1;
    }
    PyArrayObject *truncations = (PyArrayObject *)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return 1;
    }
    env->truncations = PyArray_DATA(truncations);
    return 0;
}


static PyObject* my_get(PyObject* dict, Env* env) {
    PyObject* v;
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "env is NULL");
        return NULL;
    }

    v = PyLong_FromLong(env->active_agent_count);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "active_agent_count", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_entities);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "num_entities", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    /* Map name / string fields */
    if (env->map_name) {
        PyObject* s = PyUnicode_FromString(env->map_name);
        if (!s) return NULL;
        if (PyDict_SetItemString(dict, "map_name", s) < 0) { Py_DECREF(s); return NULL; }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "map_name", Py_None) < 0) return NULL;
    }

    /* Lists (active agent indices) */
    if (env->active_agent_indices && env->active_agent_count > 0) {
        PyObject* lst = PyList_New(env->active_agent_count);
        if (!lst) return NULL;
        for (int i = 0; i < env->active_agent_count; i++) {
            PyObject* it = PyLong_FromLong(env->active_agent_indices[i]);
            if (!it) { Py_DECREF(lst); return NULL; }
            /* PyList_SetItem steals reference */
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "active_agent_indices", lst) < 0) { Py_DECREF(lst); return NULL; }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "active_agent_indices", Py_None) < 0) return NULL;
    }

    /* Optionally expose static agent indices if present */
    if (env->static_agent_indices && env->static_agent_count > 0) {
        PyObject* lst = PyList_New(env->static_agent_count);
        if (!lst) return NULL;
        for (int i = 0; i < env->static_agent_count; i++) {
            PyObject* it = PyLong_FromLong(env->static_agent_indices[i]);
            if (!it) { Py_DECREF(lst); return NULL; }
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "static_car_indices", lst) < 0) { Py_DECREF(lst); return NULL; }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "static_car_indices", Py_None) < 0) return NULL;
    }

    /* Expose entities array as a list of dicts */
    if (env->entities && env->num_entities > 0) {
        PyObject* ent_list = PyList_New(env->num_entities);
        if (!ent_list) return NULL;
        for (int i = 0; i < env->num_entities; i++) {
            Entity* e = &env->entities[i];
            PyObject* ent = PyDict_New();
            if (!ent) { Py_DECREF(ent_list); return NULL; }

            /* scalar ints */
            PyObject* tmp = PyLong_FromLong(e->type);
            if (!tmp) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
            PyDict_SetItemString(ent, "type", tmp); Py_DECREF(tmp);

            tmp = PyLong_FromLong(e->array_size);
            if (!tmp) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
            PyDict_SetItemString(ent, "array_size", tmp); Py_DECREF(tmp);

            /* trajectory float arrays: traj_x, traj_y, traj_z */
            if (e->traj_x && e->array_size > 0) {
                PyObject* lx = PyList_New(e->array_size);
                if (!lx) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_x[j]);
                    if (!fv) { Py_DECREF(lx); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lx, j, fv); /* steals ref */
                }
                PyDict_SetItemString(ent, "traj_x", lx); Py_DECREF(lx);
            } else {
                PyDict_SetItemString(ent, "traj_x", Py_None);
            }
            if (e->traj_y && e->array_size > 0) {
                PyObject* ly = PyList_New(e->array_size);
                if (!ly) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_y[j]);
                    if (!fv) { Py_DECREF(ly); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(ly, j, fv);
                }
                PyDict_SetItemString(ent, "traj_y", ly); Py_DECREF(ly);
            } else {
                PyDict_SetItemString(ent, "traj_y", Py_None);
            }
            if (e->traj_z && e->array_size > 0) {
                PyObject* lz = PyList_New(e->array_size);
                if (!lz) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_z[j]);
                    if (!fv) { Py_DECREF(lz); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lz, j, fv);
                }
                PyDict_SetItemString(ent, "traj_z", lz); Py_DECREF(lz);
            } else {
                PyDict_SetItemString(ent, "traj_z", Py_None);
            }

            /* optional velocity / heading / valid arrays for objects */
            if (e->traj_vx && e->array_size > 0) {
                PyObject* lvx = PyList_New(e->array_size);
                if (!lvx) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_vx[j]);
                    if (!fv) { Py_DECREF(lvx); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lvx, j, fv);
                }
                PyDict_SetItemString(ent, "traj_vx", lvx); Py_DECREF(lvx);
            } else {
                PyDict_SetItemString(ent, "traj_vx", Py_None);
            }
            if (e->traj_vy && e->array_size > 0) {
                PyObject* lvy = PyList_New(e->array_size);
                if (!lvy) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_vy[j]);
                    if (!fv) { Py_DECREF(lvy); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lvy, j, fv);
                }
                PyDict_SetItemString(ent, "traj_vy", lvy); Py_DECREF(lvy);
            } else {
                PyDict_SetItemString(ent, "traj_vy", Py_None);
            }
            if (e->traj_vz && e->array_size > 0) {
                PyObject* lvz = PyList_New(e->array_size);
                if (!lvz) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_vz[j]);
                    if (!fv) { Py_DECREF(lvz); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lvz, j, fv);
                }
                PyDict_SetItemString(ent, "traj_vz", lvz); Py_DECREF(lvz);
            } else {
                PyDict_SetItemString(ent, "traj_vz", Py_None);
            }
            if (e->traj_heading && e->array_size > 0) {
                PyObject* lhd = PyList_New(e->array_size);
                if (!lhd) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_heading[j]);
                    if (!fv) { Py_DECREF(lhd); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lhd, j, fv);
                }
                PyDict_SetItemString(ent, "traj_heading", lhd); Py_DECREF(lhd);
            } else {
                PyDict_SetItemString(ent, "traj_heading", Py_None);
            }
            if (e->traj_valid && e->array_size > 0) {
                PyObject* lval = PyList_New(e->array_size);
                if (!lval) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* iv = PyLong_FromLong(e->traj_valid[j]);
                    if (!iv) { Py_DECREF(lval); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lval, j, iv);
                }
                PyDict_SetItemString(ent, "traj_valid", lval); Py_DECREF(lval);
            } else {
                PyDict_SetItemString(ent, "traj_valid", Py_None);
            }

            /* scalar floats */
            PyObject* pf = PyFloat_FromDouble((double)e->width);
            PyDict_SetItemString(ent, "width", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->length);
            PyDict_SetItemString(ent, "length", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->height);
            PyDict_SetItemString(ent, "height", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->goal_position_x);
            PyDict_SetItemString(ent, "goal_position_x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->goal_position_y);
            PyDict_SetItemString(ent, "goal_position_y", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->goal_position_z);
            PyDict_SetItemString(ent, "goal_position_z", pf); Py_DECREF(pf);

            /* other scalar int/float fields */
            tmp = PyLong_FromLong(e->mark_as_expert);
            PyDict_SetItemString(ent, "mark_as_expert", tmp); Py_DECREF(tmp);

            tmp = PyLong_FromLong(e->collision_state);
            PyDict_SetItemString(ent, "collision_state", tmp); Py_DECREF(tmp);

            pf = PyFloat_FromDouble((double)e->x); PyDict_SetItemString(ent, "x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->y); PyDict_SetItemString(ent, "y", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->z); PyDict_SetItemString(ent, "z", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->vx); PyDict_SetItemString(ent, "vx", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->vy); PyDict_SetItemString(ent, "vy", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->vz); PyDict_SetItemString(ent, "vz", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->heading); PyDict_SetItemString(ent, "heading", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->heading_x); PyDict_SetItemString(ent, "heading_x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->heading_y); PyDict_SetItemString(ent, "heading_y", pf); Py_DECREF(pf);

            tmp = PyLong_FromLong(e->valid); PyDict_SetItemString(ent, "valid", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->collided_before_goal); PyDict_SetItemString(ent, "collided_before_goal", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->current_goal_reached); PyDict_SetItemString(ent, "reached_goal_this_episode", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->active_agent); PyDict_SetItemString(ent, "active_agent", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->removed); PyDict_SetItemString(ent, "removed", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->stopped); PyDict_SetItemString(ent, "stopped", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->movement_mode); PyDict_SetItemString(ent, "movement_mode", tmp); Py_DECREF(tmp);

            /* Collision snapshot fields */
            tmp = PyLong_FromLong(e->collided_with_index);
            PyDict_SetItemString(ent, "collided_with_index", tmp); Py_DECREF(tmp);
            pf = PyFloat_FromDouble((double)e->collision_x);
            PyDict_SetItemString(ent, "collision_x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->collision_y);
            PyDict_SetItemString(ent, "collision_y", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->collision_other_x);
            PyDict_SetItemString(ent, "collision_other_x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->collision_other_y);
            PyDict_SetItemString(ent, "collision_other_y", pf); Py_DECREF(pf);

            pf = PyFloat_FromDouble((double)e->steering_angle);
            PyDict_SetItemString(ent, "steering_angle", pf); Py_DECREF(pf);

            /* IDM route polyline */
            if (e->route_x && e->route_y && e->route_size > 0) {
                PyObject* rx = PyList_New(e->route_size);
                PyObject* ry = PyList_New(e->route_size);
                if (rx && ry) {
                    for (int j = 0; j < e->route_size; j++) {
                        PyList_SetItem(rx, j, PyFloat_FromDouble((double)e->route_x[j]));
                        PyList_SetItem(ry, j, PyFloat_FromDouble((double)e->route_y[j]));
                    }
                    PyDict_SetItemString(ent, "route_x", rx); Py_DECREF(rx);
                    PyDict_SetItemString(ent, "route_y", ry); Py_DECREF(ry);
                } else {
                    Py_XDECREF(rx); Py_XDECREF(ry);
                    PyDict_SetItemString(ent, "route_x", Py_None);
                    PyDict_SetItemString(ent, "route_y", Py_None);
                }
            } else {
                PyDict_SetItemString(ent, "route_x", Py_None);
                PyDict_SetItemString(ent, "route_y", Py_None);
            }

            /* Steal reference into list */
            PyList_SetItem(ent_list, i, ent);
        }
        if (PyDict_SetItemString(dict, "entities", ent_list) < 0) { Py_DECREF(ent_list); return NULL; }
        Py_DECREF(ent_list);
    } else {
        if (PyDict_SetItemString(dict, "entities", Py_None) < 0) return NULL;
    }

    /* Grid information */
    v = PyLong_FromLong(env->grid_map->grid_cols);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "grid_cols", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    v = PyLong_FromLong(env->grid_map->grid_rows);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "grid_rows", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    /* Map corners (bounding box) */
    if (env->map_corners) {
        PyObject* corners_list = PyList_New(4);
        if (!corners_list) return NULL;
        for (int i = 0; i < 4; i++) {
            PyObject* corner = PyFloat_FromDouble((double)env->map_corners[i]);
            if (!corner) { Py_DECREF(corners_list); return NULL; }
            PyList_SetItem(corners_list, i, corner);
        }
        if (PyDict_SetItemString(dict, "map_corners", corners_list) < 0) { Py_DECREF(corners_list); return NULL; }
        Py_DECREF(corners_list);
    } else {
        if (PyDict_SetItemString(dict, "map_corners", Py_None) < 0) return NULL;
    }

    /* Grid cells data - SKIPPED (not needed for visualization) */
    if (PyDict_SetItemString(dict, "grid_cells", Py_None) < 0) return NULL;

    /* Agent observations - SKIPPED (not needed for visualization) */
    if (PyDict_SetItemString(dict, "agent_observations", Py_None) < 0) return NULL;

    return dict;
}


static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    //char* map_dir = unpack_str(kwargs, "map_dir");
    char* split = unpack_str(kwargs, "split");
    char* data_root = unpack_str(kwargs, "data_root");

    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");

    if (!data_root || !split) {
        PyErr_SetString(PyExc_TypeError, "data_root and split must be set");
        return NULL;
    }
    assert(data_root && split);

    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    float goal_target_distance = unpack(kwargs, "goal_target_distance");
    float goal_lane_change_prob = (float)unpack(kwargs, "goal_lane_change_prob");
    int use_all_maps = unpack(kwargs, "use_all_maps");
    int specific_map_id = unpack(kwargs, "map_id");  // Default: -1
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = (specific_map_id >= 0) ? 1 : (use_all_maps ? num_maps : num_agents);
    int map_idx = 0;
    int maps_checked = 0;
    PyObject* agent_offsets = PyList_New(max_envs+1);
    PyObject* map_ids = PyList_New(max_envs);


    // getting env count
    while ((specific_map_id >= 0 && env_count == 0) ||
           (specific_map_id < 0 && (use_all_maps ? map_idx < max_envs : total_agent_count < num_agents && env_count < max_envs))) {
        char map_file[100];
        int map_id = (specific_map_id >= 0) ? specific_map_id : (use_all_maps ? map_idx++ : rand() % num_maps);
        Drive* env = calloc(1, sizeof(Drive));
        sprintf(map_file, "%s/%s/map_%06d.bin", data_root, split, map_id);
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->max_controlled_agents = max_controlled_agents;
        env->goal_behavior = goal_behavior;
        env->goal_target_distance = goal_target_distance;
        env->goal_lane_change_prob = goal_lane_change_prob;
        //snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
        // Set traffic mix params so set_active_agents() classifies correctly
        if (kwargs && PyDict_GetItemString(kwargs, "traffic_mix_ppo")) {
            env->traffic_mix_ppo = (float)unpack(kwargs, "traffic_mix_ppo");
        }
        if (kwargs && PyDict_GetItemString(kwargs, "traffic_mix_idm")) {
            env->traffic_mix_idm = (float)unpack(kwargs, "traffic_mix_idm");
        }
        if (kwargs && PyDict_GetItemString(kwargs, "traffic_mix_expert")) {
            env->traffic_mix_expert = (float)unpack(kwargs, "traffic_mix_expert");
        }
        if (kwargs && PyDict_GetItemString(kwargs, "idm_random_velocity")) {
            env->idm_random_velocity = (int)unpack(kwargs, "idm_random_velocity");
        }
        if (kwargs && PyDict_GetItemString(kwargs, "idm_default_velocity")) {
            env->idm_default_velocity = (float)unpack(kwargs, "idm_default_velocity");
        }

        env->entities = load_map_binary(map_file, env);
        if (env->entities == NULL) {
            maps_checked++;
            free(env);
            if (maps_checked >= num_maps * 3) {
                PyErr_SetString(PyExc_RuntimeError, "Too many map load failures");
                Py_DECREF(agent_offsets);
                Py_DECREF(map_ids);
                return NULL;
            }
            continue;
        }
        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0) {
            if (!use_all_maps) {
                maps_checked++;

                // Safeguard: if we've checked all available maps and found no active agents, raise an error
                if (maps_checked >= num_maps) {
                    for (int j = 0; j < env->num_entities; j++) {
                        free_entity(&env->entities[j]);
                    }
                    free(env->entities);
                    free(env->active_agent_indices);
                    free(env->static_agent_indices);
                    free(env->expert_static_agent_indices);
                    free(env);
                    Py_DECREF(agent_offsets);
                    Py_DECREF(map_ids);
                    char error_msg[256];
                    sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
                    PyErr_SetString(PyExc_ValueError, error_msg);
                    return NULL;
                }
            }

            for (int j = 0; j < env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
            free(env);
            continue;
        }

        // Store map_id
        PyObject *map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject *offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        for (int j = 0; j < env->num_entities; j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_agent_indices);
        free(env->expert_static_agent_indices);
        free(env);
    }
    // printf("Generated %d environments to cover %d agents (requested %d agents)\n", env_count, total_agent_count,
    // num_agents);
    if (!use_all_maps && total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }
    PyObject *final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject *final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "episode_length")) {
        conf.episode_length = (int)unpack(kwargs, "episode_length");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_speed_limit")) {
        conf.reward_speed_limit = (float)unpack(kwargs, "reward_speed_limit");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_lane_alignment")) {
        conf.reward_lane_alignment = (float)unpack(kwargs, "reward_lane_alignment");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_lane_distance")) {
        conf.reward_lane_distance = (float)unpack(kwargs, "reward_lane_distance");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_velocity")) {
        conf.reward_velocity = (float)unpack(kwargs, "reward_velocity");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_comfort")) {
        conf.reward_comfort = (float)unpack(kwargs, "reward_comfort");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_l_align")) {
        conf.reward_l_align = (float)unpack(kwargs, "reward_l_align");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_l_align_vel")) {
        conf.reward_l_align_vel = (float)unpack(kwargs, "reward_l_align_vel");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_l_center")) {
        conf.reward_l_center = (float)unpack(kwargs, "reward_l_center");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_l_center_bias")) {
        conf.reward_l_center_bias = (float)unpack(kwargs, "reward_l_center_bias");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_reverse")) {
        conf.reward_reverse = (float)unpack(kwargs, "reward_reverse");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_jerk_legacy")) {
        conf.reward_jerk_legacy = (float)unpack(kwargs, "reward_jerk_legacy");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_conditioning")) {
        conf.reward_conditioning = (int)unpack(kwargs, "reward_conditioning");
    }
    // action_type and dynamics_model kwargs override the INI file (needed at
    // eval time so the same evaluation.ini can serve both classic and jerk
    // planners without editing the INI).
    if (kwargs && PyDict_GetItemString(kwargs, "action_type_override")) {
        conf.action_type = (int)unpack(kwargs, "action_type_override");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "dynamics_model_override")) {
        conf.dynamics_model = (int)unpack(kwargs, "dynamics_model_override");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_timestep")) {
        conf.reward_timestep = (float)unpack(kwargs, "reward_timestep");
    }
    // Optional per-step reward breakdown buffer (shape: [max_agents, REWARD_COMPONENT_COUNT]).
    // When NULL, C skips all RC_* writes — zero overhead for training.
    env->reward_components = NULL;
    env->reward_components_raw = NULL;
    if (kwargs) {
        const char *rc_keys[2] = {"reward_components", "reward_components_raw"};
        float **rc_ptrs[2] = {&env->reward_components, &env->reward_components_raw};
        for (int k = 0; k < 2; k++) {
            PyObject *rc_obj = PyDict_GetItemString(kwargs, rc_keys[k]);
            if (rc_obj && rc_obj != Py_None) {
                if (!PyObject_TypeCheck(rc_obj, &PyArray_Type)) {
                    PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", rc_keys[k]);
                    return -1;
                }
                PyArrayObject *rc_arr = (PyArrayObject *)rc_obj;
                if (!PyArray_ISCONTIGUOUS(rc_arr)) {
                    PyErr_Format(PyExc_ValueError, "%s must be contiguous", rc_keys[k]);
                    return -1;
                }
                if (PyArray_TYPE(rc_arr) != NPY_FLOAT32) {
                    PyErr_Format(PyExc_ValueError, "%s must be float32", rc_keys[k]);
                    return -1;
                }
                *rc_ptrs[k] = PyArray_DATA(rc_arr);
            }
        }
    }

    // Deterministic creward override: 1 flag + CREWARD_FEATURES ego + traffic.
    // Ordered [delta_goal, alpha_collision, alpha_boundary, alpha_comfort,
    //          alpha_l_align, alpha_vel_align, alpha_l_center,
    //          alpha_center_bias, alpha_reverse, goal_speed].
    env->creward_deterministic = 0;
    if (kwargs && PyDict_GetItemString(kwargs, "creward_deterministic")) {
        env->creward_deterministic = (int)unpack(kwargs, "creward_deterministic");
    }
    env->emit_jerk_ego_obs = 0;
    if (kwargs && PyDict_GetItemString(kwargs, "emit_jerk_ego_obs")) {
        env->emit_jerk_ego_obs = (int)unpack(kwargs, "emit_jerk_ego_obs");
    }
    env->ego_entity_idx = -1;
    if (kwargs && PyDict_GetItemString(kwargs, "ego_entity_idx")) {
        env->ego_entity_idx = (int)unpack(kwargs, "ego_entity_idx");
    }
    static const char *CREWARD_FIELDS[CREWARD_FEATURES] = {
        "delta_goal", "alpha_collision", "alpha_boundary", "alpha_comfort",
        "alpha_l_align", "alpha_vel_align", "alpha_l_center",
        "alpha_center_bias", "alpha_reverse", "goal_speed",
    };
    for (int i = 0; i < CREWARD_FEATURES; i++) {
        char key[64];
        snprintf(key, sizeof(key), "creward_ego_%s", CREWARD_FIELDS[i]);
        env->creward_ego[i] = (kwargs && PyDict_GetItemString(kwargs, key))
            ? (float)unpack(kwargs, key) : 0.0f;
    }
    // Traffic creward profiles: count + per-profile CREWARD_FEATURES floats keyed
    // `creward_traffic_<i>_<field>`. Zero or missing count defaults to 1
    // with all fields zero (i.e. no override effect when creward_deterministic=0).
    env->creward_traffic_count = 1;
    if (kwargs && PyDict_GetItemString(kwargs, "creward_traffic_count")) {
        env->creward_traffic_count = (int)unpack(kwargs, "creward_traffic_count");
    }
    if (env->creward_traffic_count < 1) env->creward_traffic_count = 1;
    if (env->creward_traffic_count > MAX_TRAFFIC_PROFILES) env->creward_traffic_count = MAX_TRAFFIC_PROFILES;
    for (int p = 0; p < env->creward_traffic_count; p++) {
        for (int i = 0; i < CREWARD_FEATURES; i++) {
            char key[64];
            snprintf(key, sizeof(key), "creward_traffic_%d_%s", p, CREWARD_FIELDS[i]);
            env->creward_traffic[p][i] = (kwargs && PyDict_GetItemString(kwargs, key))
                ? (float)unpack(kwargs, key) : 0.0f;
        }
    }
    if (conf.episode_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "episode_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_speed_limit = conf.reward_speed_limit;
    env->reward_lane_alignment = conf.reward_lane_alignment;
    env->reward_lane_distance = conf.reward_lane_distance;
    env->reward_velocity = conf.reward_velocity;
    env->reward_comfort = conf.reward_comfort;
    env->reward_l_align = conf.reward_l_align;
    env->reward_l_align_vel = conf.reward_l_align_vel;
    env->reward_l_center = conf.reward_l_center;
    env->reward_l_center_bias = conf.reward_l_center_bias;
    env->reward_reverse = conf.reward_reverse;
    env->reward_jerk_legacy = conf.reward_jerk_legacy;
    env->reward_conditioning = conf.reward_conditioning;
    env->reward_timestep = conf.reward_timestep;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->episode_length = conf.episode_length;
    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->idm_others = (int)unpack(kwargs, "idm_others");
    env->dt = conf.dt;
    env->collision_shrink = conf.collision_shrink;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_lane_change_prob = (float)unpack(kwargs, "goal_lane_change_prob");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    //char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    char* split = unpack_str(kwargs, "split");
    char* data_root = unpack_str(kwargs, "data_root");


    int init_steps = unpack(kwargs, "init_steps");

    // Optional: include absolute global state in observations for SMART KL
    env->include_global_state = 0;
    if (kwargs && PyDict_GetItemString(kwargs, "include_global_state")) {
        env->include_global_state = (int)unpack(kwargs, "include_global_state");
    }
    env->map_id = map_id;

    // Optional: placeholder agents for user-added agents in demo viewer
    env->placeholder_agents = 0;
    if (kwargs && PyDict_GetItemString(kwargs, "placeholder_agents")) {
        env->placeholder_agents = (int)unpack(kwargs, "placeholder_agents");
    }

    // Optional: max observation partners
    env->max_obs_partners = MAX_OBS_PARTNERS;
    if (kwargs && PyDict_GetItemString(kwargs, "max_obs_partners")) {
        env->max_obs_partners = (int)unpack(kwargs, "max_obs_partners");
    }

    // Optional: traffic mix fractions
    env->traffic_mix_ppo = 0.0f;
    env->traffic_mix_idm = 0.0f;
    env->traffic_mix_expert = 0.0f;
    env->idm_random_velocity = 0;
    env->idm_default_velocity = 15.0f;
    if (kwargs && PyDict_GetItemString(kwargs, "traffic_mix_ppo")) {
        env->traffic_mix_ppo = (float)unpack(kwargs, "traffic_mix_ppo");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "traffic_mix_idm")) {
        env->traffic_mix_idm = (float)unpack(kwargs, "traffic_mix_idm");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "traffic_mix_expert")) {
        env->traffic_mix_expert = (float)unpack(kwargs, "traffic_mix_expert");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "idm_random_velocity")) {
        env->idm_random_velocity = (int)unpack(kwargs, "idm_random_velocity");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "idm_default_velocity")) {
        env->idm_default_velocity = (float)unpack(kwargs, "idm_default_velocity");
    }

    char map_file[100];
    // sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    sprintf(map_file, "%s/%s/map_%06d.bin", data_root, split, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "lane_aligned_steps", log->lane_aligned_steps);
    assign_to_dict(dict, "lane_distance_count", log->lane_distance_count);
    assign_to_dict(dict, "speed_limit_rate", log->speed_limit_rate);
    assign_to_dict(dict, "lane_distance_avg", log->lane_distance_count > 0 ? log->lane_distance_avg / log->lane_distance_count : 0.0f);
    assign_to_dict(dict, "velocity_reward_total", log->velocity_reward_total);
    assign_to_dict(dict, "comfort_violations", log->comfort_violations);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}

// ============================================================================
// OBSERVATION-ONLY ENV (for nuPlan integration)
// ============================================================================

// init_obs_env(bin_path, obs_array) -> env_handle
// Creates a lightweight Drive env that only loads roads and builds the grid.
// No stepping, no rewards — just for compute_observations().
static PyObject* init_obs_env(PyObject* self, PyObject* args) {
    const char* bin_path;
    PyObject* obs_obj;

    if (!PyArg_ParseTuple(args, "sO", &bin_path, &obs_obj)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(obs_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "obs must be a NumPy array");
        return NULL;
    }
    PyArrayObject* obs_array = (PyArrayObject*)obs_obj;
    if (!PyArray_ISCONTIGUOUS(obs_array)) {
        PyErr_SetString(PyExc_ValueError, "obs array must be contiguous");
        return NULL;
    }

    Drive* env = (Drive*)calloc(1, sizeof(Drive));
    if (!env) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate Drive env");
        return NULL;
    }

    env->observations = (float*)PyArray_DATA(obs_array);
    env->map_name = strdup(bin_path);
    env->num_agents = MAX_AGENTS;
    env->max_obs_partners = MAX_OBS_PARTNERS;
    env->init_steps = 0;
    env->timestep = 0;
    env->dynamics_model = 0;  // CLASSIC
    env->include_global_state = 0;
    env->max_controlled_agents = 1;
    env->control_mode = 1;  // CONTROL_SDC_ONLY

    // Load binary and init grid (same as init() but skip simulation setup)
    env->entities = load_map_binary(env->map_name, env);
    if (!env->entities) {
        free(env->map_name);
        free(env);
        PyErr_SetString(PyExc_IOError, "Failed to load map binary");
        return NULL;
    }
    set_means(env);
    init_grid_map(env);
    env->grid_map->vision_range = 21;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    set_active_agents(env);
    set_start_position(env);

    // Allocate dummy buffers that init() normally doesn't need
    if (env->active_agent_count > 0) {
        env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
    }

    // Pre-allocate static_agent_indices large enough for compute_obs_external
    // (set_active_agents allocated for static_agent_count which may be 0)
    {
        int* new_buf = (int*)calloc(MAX_AGENTS, sizeof(int));
        // Copy existing static indices if any
        for (int i = 0; i < env->static_agent_count && i < MAX_AGENTS; i++) {
            new_buf[i] = env->static_agent_indices[i];
        }
        if (env->static_agent_indices) {
            free(env->static_agent_indices);
        }
        env->static_agent_indices = new_buf;
    }
    env->static_agent_count = 0;

    return PyLong_FromVoidPtr(env);
}

// compute_obs_external(env_handle, ego_x, ego_y, heading, vx, vy, width, length,
//                      goal_x, goal_y, agents_array)
// Updates ego + agent entities, then calls compute_observations().
// agents_array: Nx8 float32 numpy [x, y, heading, vx, vy, width, length, type]
// Result goes into the obs numpy buffer passed to init_obs_env.
static PyObject* compute_obs_external(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    float ego_x, ego_y, ego_heading, ego_vx, ego_vy, ego_width, ego_length;
    float goal_x, goal_y;
    PyObject* agents_obj;

    if (!PyArg_ParseTuple(args, "OfffffffffO",
            &handle_obj,
            &ego_x, &ego_y, &ego_heading, &ego_vx, &ego_vy,
            &ego_width, &ego_length, &goal_x, &goal_y,
            &agents_obj)) {
        return NULL;
    }

    Drive* env = (Drive*)PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }

    // Subtract world mean (coordinates were centered during init)
    float cx = ego_x - env->world_mean_x;
    float cy = ego_y - env->world_mean_y;
    float gx = goal_x - env->world_mean_x;
    float gy = goal_y - env->world_mean_y;

    // Update ego entity
    if (env->active_agent_count > 0) {
        int ego_idx = env->active_agent_indices[0];
        Entity* ego = &env->entities[ego_idx];
        ego->x = cx;
        ego->y = cy;
        ego->heading = ego_heading;
        ego->heading_x = cosf(ego_heading);
        ego->heading_y = sinf(ego_heading);
        ego->vx = ego_vx;
        ego->vy = ego_vy;
        ego->width = ego_width;
        ego->length = ego_length;
        ego->goal_position_x = gx;
        ego->goal_position_y = gy;
        ego->collision_state = 0;
    }

    // Update agent entities (tracked objects from nuPlan)
    PyArrayObject* agents_array = NULL;
    int num_agents = 0;
    if (agents_obj != Py_None && PyObject_TypeCheck(agents_obj, &PyArray_Type)) {
        agents_array = (PyArrayObject*)agents_obj;
        num_agents = (int)PyArray_DIM(agents_array, 0);
    }

    // Only reset and re-populate agents when an agents array is provided.
    // When agents_obj is None, keep existing agent positions from the .bin.
    if (agents_array) {
        // Reset non-ego object entities to invalid
        for (int i = 0; i < env->num_objects; i++) {
            int is_ego = 0;
            for (int a = 0; a < env->active_agent_count; a++) {
                if (env->active_agent_indices[a] == i) { is_ego = 1; break; }
            }
            if (is_ego) continue;

            Entity* e = &env->entities[i];
            if (e->type < 1 || e->type > 3) continue;
            e->x = INVALID_POSITION;
            e->y = INVALID_POSITION;
        }
    }

    // Compute observations: ego features + roads from compute_observations,
    // then overwrite partner features directly from the agents array.
    // This avoids the entity-slot limitation where the binary might not have
    // enough slots for all tracked objects (e.g. dynamically spawned pedestrians).

    // First, clear all non-ego entities so compute_observations writes zero partners
    env->static_agent_count = 0;
    env->num_actors = env->active_agent_count;

    // Run compute_observations for ego + road features (partners will be zero)
    compute_observations(env);

    // Now write partner features directly from the agents array
    if (agents_array && num_agents > 0) {
        float* agent_data = (float*)PyArray_DATA(agents_array);
        int cols = (int)PyArray_DIM(agents_array, 1);

        // Get ego state for relative transforms
        int ego_idx = env->active_agent_indices[0];
        Entity* ego = &env->entities[ego_idx];
        float cos_h = ego->heading_x;
        float sin_h = ego->heading_y;

        // Collect candidates sorted by distance (closest first)
        struct { int idx; float dist_sq; } candidates[1024];
        int num_candidates = 0;

        for (int j = 0; j < num_agents && num_candidates < 1024; j++) {
            float* row = &agent_data[j * cols];
            float ax = row[0] - env->world_mean_x;
            float ay = row[1] - env->world_mean_y;
            float dx = ax - ego->x;
            float dy = ay - ego->y;
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq > 2500.0f) continue;  // > 50m away
            candidates[num_candidates].idx = j;
            candidates[num_candidates].dist_sq = dist_sq;
            num_candidates++;
        }

        // Sort by distance
        for (int a = 1; a < num_candidates; a++) {
            int tmp_idx = candidates[a].idx;
            float tmp_dist = candidates[a].dist_sq;
            int b = a - 1;
            while (b >= 0 && candidates[b].dist_sq > tmp_dist) {
                candidates[b + 1] = candidates[b];
                b--;
            }
            candidates[b + 1].idx = tmp_idx;
            candidates[b + 1].dist_sq = tmp_dist;
        }

        // Write closest partners into observation buffer
        int ego_dim = (env->dynamics_model == JERK || env->emit_jerk_ego_obs) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
        int limit = (num_candidates < env->max_obs_partners) ? num_candidates : env->max_obs_partners;
        float* obs = env->observations;

        for (int c = 0; c < limit; c++) {
            float* row = &agent_data[candidates[c].idx * cols];
            float ax = row[0] - env->world_mean_x;
            float ay = row[1] - env->world_mean_y;
            float a_heading = row[2];
            float a_vx = row[3];
            float a_vy = row[4];
            float a_width = row[5];
            float a_length = row[6];

            float a_type = (cols >= 8) ? row[7] : 1.0f;  // default: vehicle

            float dx = ax - ego->x;
            float dy = ay - ego->y;
            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;

            float a_hx = cosf(a_heading);
            float a_hy = sinf(a_heading);
            float rel_hx = a_hx * cos_h + a_hy * sin_h;
            float rel_hy = a_hy * cos_h - a_hx * sin_h;

            float speed = sqrtf(a_vx * a_vx + a_vy * a_vy);
            float v_dot_h = a_vx * a_hx + a_vy * a_hy;
            float signed_speed = (v_dot_h >= 0) ? speed : -speed;

            int obs_idx = ego_dim + c * PARTNER_FEATURES;
            obs[obs_idx + 0] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = a_width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = a_length / MAX_VEH_LEN;
            obs[obs_idx + 4] = rel_hx;
            obs[obs_idx + 5] = rel_hy;
            obs[obs_idx + 6] = signed_speed / MAX_SPEED;
            obs[obs_idx + 7] = a_type;
        }
    }

    Py_RETURN_NONE;
}

// ============================================================================
// SNAPSHOT BINDINGS
// ============================================================================

// Create snapshots for all environments in a VecEnv
// Returns a list of snapshot handles (integers)
static PyObject* vec_create_snapshot(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* snapshot_list = PyList_New(vec->num_envs);
    if (!snapshot_list) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        DriveSnapshot* snapshot = create_snapshot(vec->envs[i]);
        if (!snapshot) {
            // Clean up already created snapshots
            for (int j = 0; j < i; j++) {
                PyObject* handle = PyList_GetItem(snapshot_list, j);
                DriveSnapshot* prev_snap = (DriveSnapshot*)PyLong_AsVoidPtr(handle);
                free_snapshot(prev_snap);
            }
            Py_DECREF(snapshot_list);
            PyErr_SetString(PyExc_MemoryError, "Failed to create snapshot");
            return NULL;
        }
        PyObject* handle = PyLong_FromVoidPtr(snapshot);
        PyList_SetItem(snapshot_list, i, handle);  // steals reference
    }

    return snapshot_list;
}

// Restore snapshots for all environments in a VecEnv
// Takes the vec handle and a list of snapshot handles
static PyObject* vec_restore_snapshot(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_restore_snapshot requires 2 arguments: vec_handle, snapshot_list");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* snapshot_list = PyTuple_GetItem(args, 1);
    if (!PyList_Check(snapshot_list)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be a list of snapshot handles");
        return NULL;
    }

    Py_ssize_t list_size = PyList_Size(snapshot_list);
    if (list_size != vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "Snapshot list size must match number of environments");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        PyObject* handle = PyList_GetItem(snapshot_list, i);
        if (!PyLong_Check(handle)) {
            PyErr_SetString(PyExc_TypeError, "Snapshot handles must be integers");
            return NULL;
        }
        DriveSnapshot* snapshot = (DriveSnapshot*)PyLong_AsVoidPtr(handle);
        if (!snapshot) {
            PyErr_SetString(PyExc_ValueError, "Invalid snapshot handle");
            return NULL;
        }
        restore_snapshot(vec->envs[i], snapshot);
    }

    Py_RETURN_NONE;
}

// Restore a single snapshot into all environments in a VecEnv
// Takes the vec handle and a single snapshot handle
static PyObject* vec_restore_snapshot_broadcast(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_restore_snapshot_broadcast requires 2 arguments: vec_handle, snapshot_handle");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* handle = PyTuple_GetItem(args, 1);
    if (!PyLong_Check(handle)) {
        PyErr_SetString(PyExc_TypeError, "Snapshot handle must be an integer");
        return NULL;
    }
    DriveSnapshot* snapshot = (DriveSnapshot*)PyLong_AsVoidPtr(handle);
    if (!snapshot) {
        PyErr_SetString(PyExc_ValueError, "Invalid snapshot handle");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        restore_snapshot(vec->envs[i], snapshot);
    }

    Py_RETURN_NONE;
}

// Clone a single environment into a VecEnv batch
// Args: src_vec_handle, observations, actions, rewards, terminals, truncations,
//       collision_rewards, offroad_rewards, goal_rewards, goal_distances, jerk_rewards,
//       lane_distances, lane_alignments, num_envs, agents_per_env
static PyObject* vec_clone_from_env(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 15) {
        PyErr_SetString(PyExc_TypeError, "vec_clone_from_env requires 15 arguments");
        return NULL;
    }

    VecEnv* src_vec = unpack_vecenv(args);
    if (!src_vec) {
        return NULL;
    }
    if (src_vec->num_envs != 1 || src_vec->envs[0] == NULL) {
        PyErr_SetString(PyExc_ValueError, "vec_clone_from_env requires a single-env VecEnv");
        return NULL;
    }

    PyObject* obs_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(obs_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* observations = (PyArrayObject*)obs_obj;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(observations) < 2) {
        PyErr_SetString(PyExc_ValueError, "Observations must be at least 2D");
        return NULL;
    }

    PyObject* act_obj = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(act_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject* actions = (PyArrayObject*)act_obj;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return NULL;
    }

    PyObject* rew_obj = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(rew_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew_obj;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }

    PyObject* term_obj = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(term_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject* terminals = (PyArrayObject*)term_obj;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }

    PyObject* trunc_obj = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(trunc_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* truncations = (PyArrayObject*)trunc_obj;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }

    // Decomposed rewards arrays
    PyObject* coll_rew_obj = PyTuple_GetItem(args, 6);
    if (!PyObject_TypeCheck(coll_rew_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "collision_rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* collision_rewards = (PyArrayObject*)coll_rew_obj;
    if (!PyArray_ISCONTIGUOUS(collision_rewards) || PyArray_NDIM(collision_rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "collision_rewards must be contiguous 1D");
        return NULL;
    }

    PyObject* offr_rew_obj = PyTuple_GetItem(args, 7);
    if (!PyObject_TypeCheck(offr_rew_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "offroad_rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* offroad_rewards = (PyArrayObject*)offr_rew_obj;
    if (!PyArray_ISCONTIGUOUS(offroad_rewards) || PyArray_NDIM(offroad_rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "offroad_rewards must be contiguous 1D");
        return NULL;
    }

    PyObject* goal_rew_obj = PyTuple_GetItem(args, 8);
    if (!PyObject_TypeCheck(goal_rew_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "goal_rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* goal_rewards = (PyArrayObject*)goal_rew_obj;
    if (!PyArray_ISCONTIGUOUS(goal_rewards) || PyArray_NDIM(goal_rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "goal_rewards must be contiguous 1D");
        return NULL;
    }

    PyObject* goal_dist_obj = PyTuple_GetItem(args, 9);
    if (!PyObject_TypeCheck(goal_dist_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "goal_distances must be a NumPy array");
        return NULL;
    }
    PyArrayObject* goal_distances = (PyArrayObject*)goal_dist_obj;
    if (!PyArray_ISCONTIGUOUS(goal_distances) || PyArray_NDIM(goal_distances) != 1) {
        PyErr_SetString(PyExc_ValueError, "goal_distances must be contiguous 1D");
        return NULL;
    }

    PyObject* jerk_rew_obj = PyTuple_GetItem(args, 10);
    if (!PyObject_TypeCheck(jerk_rew_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "jerk_rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* jerk_rewards = (PyArrayObject*)jerk_rew_obj;
    if (!PyArray_ISCONTIGUOUS(jerk_rewards) || PyArray_NDIM(jerk_rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "jerk_rewards must be contiguous 1D");
        return NULL;
    }

    PyObject* lane_dist_obj = PyTuple_GetItem(args, 11);
    if (!PyObject_TypeCheck(lane_dist_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "lane_distances must be a NumPy array");
        return NULL;
    }
    PyArrayObject* lane_distances = (PyArrayObject*)lane_dist_obj;
    if (!PyArray_ISCONTIGUOUS(lane_distances) || PyArray_NDIM(lane_distances) != 1) {
        PyErr_SetString(PyExc_ValueError, "lane_distances must be contiguous 1D");
        return NULL;
    }

    PyObject* lane_align_obj = PyTuple_GetItem(args, 12);
    if (!PyObject_TypeCheck(lane_align_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "lane_alignments must be a NumPy array");
        return NULL;
    }
    PyArrayObject* lane_alignments = (PyArrayObject*)lane_align_obj;
    if (!PyArray_ISCONTIGUOUS(lane_alignments) || PyArray_NDIM(lane_alignments) != 1) {
        PyErr_SetString(PyExc_ValueError, "lane_alignments must be contiguous 1D");
        return NULL;
    }

    PyObject* num_envs_obj = PyTuple_GetItem(args, 13);
    if (!PyObject_TypeCheck(num_envs_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be an integer");
        return NULL;
    }
    int num_envs = (int)PyLong_AsLong(num_envs_obj);
    if (num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "num_envs must be > 0");
        return NULL;
    }

    PyObject* agents_per_env_obj = PyTuple_GetItem(args, 14);
    if (!PyObject_TypeCheck(agents_per_env_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "agents_per_env must be an integer");
        return NULL;
    }
    int agents_per_env = (int)PyLong_AsLong(agents_per_env_obj);
    if (agents_per_env <= 0) {
        PyErr_SetString(PyExc_ValueError, "agents_per_env must be > 0");
        return NULL;
    }

    Drive* src = (Drive*)src_vec->envs[0];
    if (agents_per_env != src->num_agents) {
        PyErr_SetString(PyExc_ValueError, "agents_per_env must match source env num_agents");
        return NULL;
    }

    DriveSnapshot* snapshot = create_snapshot(src);
    if (!snapshot) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create snapshot for cloning");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        free_snapshot(snapshot);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }
    vec->num_envs = num_envs;
    vec->envs = (Env**)calloc(num_envs, sizeof(Env*));
    if (!vec->envs) {
        free_snapshot(snapshot);
        free(vec);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    char* obs_base = (char*)PyArray_DATA(observations);
    char* act_base = (char*)PyArray_DATA(actions);
    char* rew_base = (char*)PyArray_DATA(rewards);
    char* term_base = (char*)PyArray_DATA(terminals);
    char* trunc_base = (char*)PyArray_DATA(truncations);
    char* coll_rew_base = (char*)PyArray_DATA(collision_rewards);
    char* offr_rew_base = (char*)PyArray_DATA(offroad_rewards);
    char* goal_rew_base = (char*)PyArray_DATA(goal_rewards);
    char* goal_dist_base = (char*)PyArray_DATA(goal_distances);
    char* jerk_rew_base = (char*)PyArray_DATA(jerk_rewards);
    char* lane_dist_base = (char*)PyArray_DATA(lane_distances);
    char* lane_align_base = (char*)PyArray_DATA(lane_alignments);
    npy_intp obs_stride = PyArray_STRIDE(observations, 0);
    npy_intp act_stride = PyArray_STRIDE(actions, 0);
    npy_intp rew_stride = PyArray_STRIDE(rewards, 0);
    npy_intp term_stride = PyArray_STRIDE(terminals, 0);
    npy_intp trunc_stride = PyArray_STRIDE(truncations, 0);
    npy_intp coll_rew_stride = PyArray_STRIDE(collision_rewards, 0);
    npy_intp offr_rew_stride = PyArray_STRIDE(offroad_rewards, 0);
    npy_intp goal_rew_stride = PyArray_STRIDE(goal_rewards, 0);
    npy_intp goal_dist_stride = PyArray_STRIDE(goal_distances, 0);
    npy_intp jerk_rew_stride = PyArray_STRIDE(jerk_rewards, 0);
    npy_intp lane_dist_stride = PyArray_STRIDE(lane_distances, 0);
    npy_intp lane_align_stride = PyArray_STRIDE(lane_alignments, 0);

    for (int i = 0; i < num_envs; i++) {
        Drive* env = (Drive*)calloc(1, sizeof(Drive));
        if (!env) {
            free_snapshot(snapshot);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
            return NULL;
        }
        vec->envs[i] = (Env*)env;

        int agent_offset = i * agents_per_env;
        env->observations = (void*)(obs_base + agent_offset * obs_stride);
        env->actions = (void*)(act_base + agent_offset * act_stride);
        env->rewards = (void*)(rew_base + agent_offset * rew_stride);
        env->terminals = (void*)(term_base + agent_offset * term_stride);
        env->truncations = (void*)(trunc_base + agent_offset * trunc_stride);
        // Decomposed rewards for CEM
        env->collision_rewards = (void*)(coll_rew_base + agent_offset * coll_rew_stride);
        env->offroad_rewards = (void*)(offr_rew_base + agent_offset * offr_rew_stride);
        env->goal_rewards = (void*)(goal_rew_base + agent_offset * goal_rew_stride);
        env->goal_distances = (void*)(goal_dist_base + agent_offset * goal_dist_stride);
        env->jerk_rewards = (void*)(jerk_rew_base + agent_offset * jerk_rew_stride);
        env->lane_distances = (void*)(lane_dist_base + agent_offset * lane_dist_stride);
        env->lane_alignments = (void*)(lane_align_base + agent_offset * lane_align_stride);

        env->action_type = src->action_type;
        env->reward_vehicle_collision = src->reward_vehicle_collision;
        env->reward_offroad_collision = src->reward_offroad_collision;
        env->reward_speed_limit = src->reward_speed_limit;
        env->reward_lane_alignment = src->reward_lane_alignment;
        env->reward_lane_distance = src->reward_lane_distance;
        env->reward_velocity = src->reward_velocity;
        env->reward_comfort = src->reward_comfort;
        env->reward_l_align = src->reward_l_align;
        env->reward_l_align_vel = src->reward_l_align_vel;
        env->reward_l_center = src->reward_l_center;
        env->reward_l_center_bias = src->reward_l_center_bias;
        env->reward_reverse = src->reward_reverse;
        env->reward_jerk_legacy = src->reward_jerk_legacy;
        env->reward_conditioning = src->reward_conditioning;
        // Cloned CEM envs share the decomposition path but not the buffer;
        // reward_components stays NULL here (main env keeps its own buffer).
        env->reward_components = NULL;
        env->reward_components_raw = NULL;
        env->creward_deterministic = src->creward_deterministic;
        env->emit_jerk_ego_obs = src->emit_jerk_ego_obs;
        env->ego_entity_idx = src->ego_entity_idx;
        env->creward_traffic_count = src->creward_traffic_count;
        for (int i = 0; i < CREWARD_FEATURES; i++) {
            env->creward_ego[i] = src->creward_ego[i];
        }
        for (int p = 0; p < MAX_TRAFFIC_PROFILES; p++) {
            for (int i = 0; i < CREWARD_FEATURES; i++) {
                env->creward_traffic[p][i] = src->creward_traffic[p][i];
            }
        }
        env->reward_timestep = src->reward_timestep;
        env->reward_goal = src->reward_goal;
        env->reward_goal_post_respawn = src->reward_goal_post_respawn;
        env->goal_radius = src->goal_radius;
        env->goal_speed = src->goal_speed;
        env->dt = src->dt;
        env->episode_length = src->episode_length;
        env->termination_mode = src->termination_mode;
        env->goal_behavior = src->goal_behavior;
        env->goal_target_distance = src->goal_target_distance;
        env->goal_lane_change_prob = src->goal_lane_change_prob;
        env->collision_behavior = src->collision_behavior;
        env->offroad_behavior = src->offroad_behavior;
        env->init_mode = src->init_mode;
        env->control_mode = src->control_mode;
        env->use_goal_generation = src->use_goal_generation;
        env->scenario_length = src->scenario_length;
        env->control_non_vehicles = src->control_non_vehicles;
        env->max_controlled_agents = src->max_controlled_agents;
        env->idm_others = src->idm_others;
        env->human_agent_idx = src->human_agent_idx;
        env->placeholder_agents = src->placeholder_agents;
        env->max_obs_partners = src->max_obs_partners;
        env->num_agents = src->num_agents;
        env->init_steps = src->init_steps;
        env->timestep = src->init_steps;
        env->collision_shrink = src->collision_shrink;
        env->idm_min_gap = src->idm_min_gap;
        env->idm_headway_time = src->idm_headway_time;
        env->idm_accel_max = src->idm_accel_max;
        env->idm_decel_max = src->idm_decel_max;

        if (src->map_name) {
            env->map_name = strdup(src->map_name);
        }
        if (src->ini_file) {
            env->ini_file = strdup(src->ini_file);
        }

        init(env);

        if (env->active_agent_indices) {
            free(env->active_agent_indices);
            env->active_agent_indices = NULL;
        }
        if (env->static_agent_indices) {
            free(env->static_agent_indices);
            env->static_agent_indices = NULL;
        }
        if (env->expert_static_agent_indices) {
            free(env->expert_static_agent_indices);
            env->expert_static_agent_indices = NULL;
        }
        if (env->logs) {
            free(env->logs);
            env->logs = NULL;
        }

        env->active_agent_count = src->active_agent_count;
        env->static_agent_count = src->static_agent_count;
        env->expert_static_agent_count = src->expert_static_agent_count;
        env->logs_capacity = src->logs_capacity;

        if (env->active_agent_count > 0) {
            env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
            memcpy(env->active_agent_indices, src->active_agent_indices, env->active_agent_count * sizeof(int));
        }
        if (env->static_agent_count > 0) {
            env->static_agent_indices = (int*)malloc(env->static_agent_count * sizeof(int));
            memcpy(env->static_agent_indices, src->static_agent_indices, env->static_agent_count * sizeof(int));
        }
        if (env->expert_static_agent_count > 0) {
            env->expert_static_agent_indices = (int*)malloc(env->expert_static_agent_count * sizeof(int));
            memcpy(env->expert_static_agent_indices, src->expert_static_agent_indices, env->expert_static_agent_count * sizeof(int));
        }
        if (env->active_agent_count > 0) {
            env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
            if (src->logs) {
                memcpy(env->logs, src->logs, env->active_agent_count * sizeof(Log));
            }
        }

        env->num_actors = env->active_agent_count + env->static_agent_count;

        restore_snapshot(env, snapshot);
    }

    free_snapshot(snapshot);
    return PyLong_FromVoidPtr(vec);
}

// Free snapshots from a list of snapshot handles
static PyObject* vec_free_snapshot(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_free_snapshot requires 1 argument: snapshot_list");
        return NULL;
    }

    PyObject* snapshot_list = PyTuple_GetItem(args, 0);
    if (!PyList_Check(snapshot_list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of snapshot handles");
        return NULL;
    }

    Py_ssize_t list_size = PyList_Size(snapshot_list);
    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* handle = PyList_GetItem(snapshot_list, i);
        if (!PyLong_Check(handle)) {
            PyErr_SetString(PyExc_TypeError, "Snapshot handles must be integers");
            return NULL;
        }
        DriveSnapshot* snapshot = (DriveSnapshot*)PyLong_AsVoidPtr(handle);
        if (snapshot) {
            free_snapshot(snapshot);
        }
    }

    Py_RETURN_NONE;
}

// Get ego agent positions for all environments in a VecEnv
// Args: vec_handle, out_x (numpy array), out_y (numpy array)
// Returns positions of the first active agent in each environment
static PyObject* vec_get_ego_positions(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "vec_get_ego_positions requires 4 arguments: vec_handle, out_x, out_y, ego_agent_idx");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* x_obj = PyTuple_GetItem(args, 1);
    PyObject* y_obj = PyTuple_GetItem(args, 2);
    int ego_agent_idx = (int)PyLong_AsLong(PyTuple_GetItem(args, 3));

    if (!PyObject_TypeCheck(x_obj, &PyArray_Type) || !PyObject_TypeCheck(y_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "out_x and out_y must be NumPy arrays");
        return NULL;
    }

    PyArrayObject* out_x = (PyArrayObject*)x_obj;
    PyArrayObject* out_y = (PyArrayObject*)y_obj;

    if (!PyArray_ISCONTIGUOUS(out_x) || !PyArray_ISCONTIGUOUS(out_y)) {
        PyErr_SetString(PyExc_ValueError, "Output arrays must be contiguous");
        return NULL;
    }

    if (PyArray_NDIM(out_x) != 1 || PyArray_NDIM(out_y) != 1) {
        PyErr_SetString(PyExc_ValueError, "Output arrays must be 1D");
        return NULL;
    }

    npy_intp n_envs = PyArray_DIM(out_x, 0);
    if (n_envs != vec->num_envs || PyArray_DIM(out_y, 0) != vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "Output arrays must have length equal to num_envs");
        return NULL;
    }

    float* x_data = (float*)PyArray_DATA(out_x);
    float* y_data = (float*)PyArray_DATA(out_y);

    // Extract ego positions from each environment
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (env && ego_agent_idx < env->active_agent_count && env->active_agent_indices) {
            int ego_idx = env->active_agent_indices[ego_agent_idx];
            Entity* ego = &env->entities[ego_idx];
            // Get current position from entity state (not trajectory arrays)
            x_data[i] = ego->x;
            y_data[i] = ego->y;
        } else {
            x_data[i] = 0.0f;
            y_data[i] = 0.0f;
        }
    }

    Py_RETURN_NONE;
}

// Get per-agent logs for a specific agent index
static PyObject* vec_get_agent_log(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get_agent_log requires 2 arguments: vec_handle, agent_idx");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* agent_idx_obj = PyTuple_GetItem(args, 1);
    if (!PyLong_Check(agent_idx_obj)) {
        PyErr_SetString(PyExc_TypeError, "agent_idx must be an integer");
        return NULL;
    }
    int agent_idx = (int)PyLong_AsLong(agent_idx_obj);

    // For now, assume single env and get logs from first env
    if (vec->num_envs < 1 || !vec->envs[0]) {
        PyErr_SetString(PyExc_ValueError, "No environments in VecEnv");
        return NULL;
    }

    Drive* env = vec->envs[0];
    if (agent_idx < 0 || agent_idx >= env->active_agent_count) {
        PyErr_SetString(PyExc_IndexError, "agent_idx out of range");
        return NULL;
    }

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

    // Get per-agent log
    Log* agent_log = &env->logs[agent_idx];

    // Also get entity info for goal reached status
    int entity_idx = env->active_agent_indices[agent_idx];
    Entity* entity = &env->entities[entity_idx];

    assign_to_dict(dict, "episode_return", agent_log->episode_return);
    assign_to_dict(dict, "episode_length", agent_log->episode_length);
    assign_to_dict(dict, "collision_rate", agent_log->collision_rate);
    assign_to_dict(dict, "offroad_rate", agent_log->offroad_rate);
    assign_to_dict(dict, "lane_alignment_rate", agent_log->lane_alignment_rate);
    assign_to_dict(dict, "lane_aligned_steps", agent_log->lane_aligned_steps);
    assign_to_dict(dict, "lane_distance_count", agent_log->lane_distance_count);
    assign_to_dict(dict, "speed_limit_rate", agent_log->speed_limit_rate);
    assign_to_dict(dict, "lane_distance_avg", agent_log->lane_distance_count > 0 ? agent_log->lane_distance_avg / agent_log->lane_distance_count : 0.0f);
    assign_to_dict(dict, "velocity_reward_total", agent_log->velocity_reward_total);
    assign_to_dict(dict, "comfort_violations", agent_log->comfort_violations);
    assign_to_dict(dict, "speed_at_goal", agent_log->speed_at_goal);
    assign_to_dict(dict, "collisions_per_agent", agent_log->collisions_per_agent);
    assign_to_dict(dict, "offroad_per_agent", agent_log->offroad_per_agent);

    // Add entity-level metrics
    assign_to_dict(dict, "goals_reached_this_episode", entity->goals_reached_this_episode);
    assign_to_dict(dict, "goals_sampled_this_episode", entity->goals_sampled_this_episode);
    float completion_rate = entity->goals_sampled_this_episode > 0 ?
        entity->goals_reached_this_episode / entity->goals_sampled_this_episode : 0.0f;
    assign_to_dict(dict, "completion_rate", completion_rate);
    assign_to_dict(dict, "stopped", (float)entity->stopped);
    assign_to_dict(dict, "removed", (float)entity->removed);

    return dict;
}

// Set movement mode for specific agents in all environments
// Args: vec_handle, agent_indices (list of int), mode (int)
static PyObject* vec_set_movement_mode(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_movement_mode requires 3 arguments: vec_handle, agent_indices, mode");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* indices_obj = PyTuple_GetItem(args, 1);
    if (!PyList_Check(indices_obj)) {
        PyErr_SetString(PyExc_TypeError, "agent_indices must be a list of integers");
        return NULL;
    }

    PyObject* mode_obj = PyTuple_GetItem(args, 2);
    if (!PyLong_Check(mode_obj)) {
        PyErr_SetString(PyExc_TypeError, "mode must be an integer");
        return NULL;
    }
    int mode = (int)PyLong_AsLong(mode_obj);

    Py_ssize_t num_indices = PyList_Size(indices_obj);

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;

        for (Py_ssize_t j = 0; j < num_indices; j++) {
            PyObject* idx_obj = PyList_GetItem(indices_obj, j);
            if (!PyLong_Check(idx_obj)) {
                PyErr_SetString(PyExc_TypeError, "agent_indices elements must be integers");
                return NULL;
            }
            int action_idx = (int)PyLong_AsLong(idx_obj);
            if (action_idx < 0 || action_idx >= env->active_agent_count) continue;
            int agent_idx = env->active_agent_indices[action_idx];
            int prev_mode = env->entities[agent_idx].movement_mode;
            env->entities[agent_idx].movement_mode = mode;
            // Rebuild the IDM route when entering IDM from a different mode
            // (e.g. PPO drove in DYNAMICS and may have changed lane), or when
            // no route exists yet. Skip rebuild on consecutive IDM steps to
            // avoid the per-step cost.
            if (mode == MOVEMENT_IDM &&
                (prev_mode != MOVEMENT_IDM || env->entities[agent_idx].route_size == 0)) {
                build_route_for_agent(env, agent_idx);
            }
        }
    }

    Py_RETURN_NONE;
}

// Set IDM proposals: per-env velocity and lateral offset for a given action_idx
// Args: vec_handle, action_idx (int), velocities (numpy float32), offsets (numpy float32)
static PyObject* vec_set_idm_proposals(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "vec_set_idm_proposals requires 4 arguments: vec_handle, action_idx, velocities, offsets");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* idx_obj = PyTuple_GetItem(args, 1);
    if (!PyLong_Check(idx_obj)) {
        PyErr_SetString(PyExc_TypeError, "action_idx must be an integer");
        return NULL;
    }
    int action_idx = (int)PyLong_AsLong(idx_obj);

    PyArrayObject* vel_arr = (PyArrayObject*)PyTuple_GetItem(args, 2);
    PyArrayObject* off_arr = (PyArrayObject*)PyTuple_GetItem(args, 3);
    float* velocities = (float*)PyArray_DATA(vel_arr);
    float* offsets = (float*)PyArray_DATA(off_arr);

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;
        if (action_idx < 0 || action_idx >= env->active_agent_count) continue;
        int agent_idx = env->active_agent_indices[action_idx];
        int prev_mode = env->entities[agent_idx].movement_mode;
        env->entities[agent_idx].movement_mode = MOVEMENT_IDM;
        env->entities[agent_idx].idm_target_velocity = velocities[i];
        env->entities[agent_idx].idm_lateral_offset = offsets[i];
        // Rebuild the route when entering IDM from a different mode (the agent
        // may have driven onto a different lane in DYNAMICS), or when no route
        // exists yet. This also covers the PDM-batch-env path: each restored
        // clone sees prev_mode = DYNAMICS (snapshot was taken while PPO drove)
        // and rebuilds against the current ego position.
        if (prev_mode != MOVEMENT_IDM || env->entities[agent_idx].route_size == 0) {
            build_route_for_agent(env, agent_idx);
        }
    }

    Py_RETURN_NONE;
}

// Set IDM target velocity for given agent indices across all envs
// Args: vec_handle, agent_indices (list of int), target_velocity (float)
static PyObject* vec_set_idm_target_velocity(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_idm_target_velocity requires 3 arguments: vec_handle, agent_indices, target_velocity");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* indices_list = PyTuple_GetItem(args, 1);
    if (!PyList_Check(indices_list)) {
        PyErr_SetString(PyExc_TypeError, "agent_indices must be a list");
        return NULL;
    }

    PyObject* vel_obj = PyTuple_GetItem(args, 2);
    float target_velocity = (float)PyFloat_AsDouble(vel_obj);

    Py_ssize_t num_indices = PyList_Size(indices_list);

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;
        for (Py_ssize_t j = 0; j < num_indices; j++) {
            int action_idx = (int)PyLong_AsLong(PyList_GetItem(indices_list, j));
            if (action_idx < 0 || action_idx >= env->active_agent_count) continue;
            int agent_idx = env->active_agent_indices[action_idx];
            env->entities[agent_idx].idm_target_velocity = target_velocity;
        }
    }

    Py_RETURN_NONE;
}

// Set IDM parameters on all sub-envs
// Args: vec_handle, min_gap (float), headway_time (float), accel_max (float), decel_max (float)
static PyObject* vec_set_idm_params(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "vec_set_idm_params requires 5 arguments: vec_handle, min_gap, headway_time, accel_max, decel_max");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    float min_gap = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
    float headway_time = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
    float accel_max = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 3));
    float decel_max = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;
        env->idm_min_gap = min_gap;
        env->idm_headway_time = headway_time;
        env->idm_accel_max = accel_max;
        env->idm_decel_max = decel_max;
    }

    Py_RETURN_NONE;
}

// Set goal positions for specific agents
// Args: vec_handle, agent_indices (list of int), goal_xs (list of float), goal_ys (list of float)
static PyObject* vec_set_goals(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "vec_set_goals requires 4 arguments: vec_handle, agent_indices, goal_xs, goal_ys");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* indices_list = PyTuple_GetItem(args, 1);
    PyObject* gx_list = PyTuple_GetItem(args, 2);
    PyObject* gy_list = PyTuple_GetItem(args, 3);

    if (!PyList_Check(indices_list) || !PyList_Check(gx_list) || !PyList_Check(gy_list)) {
        PyErr_SetString(PyExc_TypeError, "agent_indices, goal_xs, goal_ys must be lists");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(indices_list);
    if (PyList_Size(gx_list) != n || PyList_Size(gy_list) != n) {
        PyErr_SetString(PyExc_ValueError, "All lists must have the same length");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;
        for (Py_ssize_t j = 0; j < n; j++) {
            int action_idx = (int)PyLong_AsLong(PyList_GetItem(indices_list, j));
            if (action_idx < 0 || action_idx >= env->active_agent_count) continue;
            int agent_idx = env->active_agent_indices[action_idx];
            float gx = (float)PyFloat_AsDouble(PyList_GetItem(gx_list, j));
            float gy = (float)PyFloat_AsDouble(PyList_GetItem(gy_list, j));
            env->entities[agent_idx].goal_position_x = gx;
            env->entities[agent_idx].goal_position_y = gy;
        }
    }

    Py_RETURN_NONE;
}

// Set positions and headings for specific agents
// Args: vec_handle, agent_indices (list of int), xs (list of float), ys (list of float), headings (list of float)
static PyObject* vec_set_positions(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "vec_set_positions requires 5 arguments: vec_handle, agent_indices, xs, ys, headings");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* indices_list = PyTuple_GetItem(args, 1);
    PyObject* x_list = PyTuple_GetItem(args, 2);
    PyObject* y_list = PyTuple_GetItem(args, 3);
    PyObject* h_list = PyTuple_GetItem(args, 4);

    if (!PyList_Check(indices_list) || !PyList_Check(x_list) || !PyList_Check(y_list) || !PyList_Check(h_list)) {
        PyErr_SetString(PyExc_TypeError, "agent_indices, xs, ys, headings must be lists");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(indices_list);
    if (PyList_Size(x_list) != n || PyList_Size(y_list) != n || PyList_Size(h_list) != n) {
        PyErr_SetString(PyExc_ValueError, "All lists must have the same length");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;
        for (Py_ssize_t j = 0; j < n; j++) {
            int action_idx = (int)PyLong_AsLong(PyList_GetItem(indices_list, j));
            if (action_idx < 0 || action_idx >= env->active_agent_count) continue;
            int agent_idx = env->active_agent_indices[action_idx];
            float x = (float)PyFloat_AsDouble(PyList_GetItem(x_list, j));
            float y = (float)PyFloat_AsDouble(PyList_GetItem(y_list, j));
            float h = (float)PyFloat_AsDouble(PyList_GetItem(h_list, j));
            env->entities[agent_idx].x = x;
            env->entities[agent_idx].y = y;
            env->entities[agent_idx].heading = h;
            env->entities[agent_idx].heading_x = cosf(h);
            env->entities[agent_idx].heading_y = sinf(h);
            env->entities[agent_idx].vx = 0.0f;
            env->entities[agent_idx].vy = 0.0f;
            env->entities[agent_idx].removed = 0;
            env->entities[agent_idx].valid = 1;
        }
    }

    Py_RETURN_NONE;
}

// Get ego agent (action_idx=0) speed and heading for all envs
// Args: vec_handle, speeds_out (numpy float32), headings_out (numpy float32)
static PyObject* vec_get_ego_state(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_get_ego_state requires 3 arguments: vec_handle, speeds_out, headings_out");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyArrayObject* speeds_arr = (PyArrayObject*)PyTuple_GetItem(args, 1);
    PyArrayObject* headings_arr = (PyArrayObject*)PyTuple_GetItem(args, 2);
    float* speeds = (float*)PyArray_DATA(speeds_arr);
    float* headings = (float*)PyArray_DATA(headings_arr);

    for (int i = 0; i < vec->num_envs; i++) {
        Drive* env = vec->envs[i];
        if (!env) continue;
        if (env->active_agent_count <= 0) {
            speeds[i] = 0.0f;
            headings[i] = 0.0f;
            continue;
        }
        int agent_idx = env->active_agent_indices[0];
        Entity* e = &env->entities[agent_idx];
        speeds[i] = sqrtf(e->vx * e->vx + e->vy * e->vy);
        headings[i] = e->heading;
    }

    Py_RETURN_NONE;
}

// ============================================================================

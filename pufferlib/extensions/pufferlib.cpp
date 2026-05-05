/*
 * Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
 * SPDX-License-Identifier: AGPL-3.0
 *
 * This source code is derived from PufferDrive V2.0
 * (https://github.com/Emerge-Lab/PufferDrive/)
 * Copyright (c) 2026 PufferDrive, licensed under the MIT license.
 */

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace pufferlib {

void puff_advantage_row(float* values, float* rewards, float* terminations, float* truncations,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    // values are extended, so we can start at horizon-1!
    for (int t = horizon-1; t >= 0; t--) {
        int t_next = t + 1;
        // return 0.0 if truncation timestep -> invalid advantage! (next state is after the reset)
        if (truncations[t] == 1.0){ 
            advantages[t] = 0.0;
            lastpufferlam = 0.0; // for sequence until then we need to reset lastpufferlam
        } else {
        float nextnonterminal = 1.0 - terminations[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        // if (lastpufferlam == 0.0){
        //     printf("There was a 0 by coincidence.\n");
        // }
        advantages[t] = lastpufferlam;
        }
    }

    // printf("Maximum t processed: %d\n", horizon - 2);
}

void vtrace_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor terminations, torch::Tensor truncations, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, terminations, truncations, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) >= horizon, "Second dimension must at least be horizon length (it can be +1 for values, rewards and dones)");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}


// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* terminations, float* truncations, float* importance,
        float* advantages, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, const int horizon){
    int extended_offset;
    int offset;
    for (int step= 0; step < num_steps; step+=1) {
        extended_offset = step*(horizon+1);
        offset = step*horizon;
        puff_advantage_row(values + extended_offset, rewards + extended_offset,
            terminations + extended_offset, truncations + offset, importance + offset, advantages + offset,
            gamma, lambda, rho_clip, c_clip, horizon
        );
    }
    // for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
    //     int extended_offset;
    //     if (offset != 0) {
    //         extended_offset = offset + num_steps;
    //     } else {
    //         extended_offset = offset;
    //     }
    //     // printf("Extended offset: %d\n", extended_offset);
    //     // printf(" offset: %d\n", offset);
    //     puff_advantage_row(values + extended_offset, rewards + extended_offset,
    //         dones + extended_offset, importance + offset, advantages + offset,
    //         gamma, lambda, rho_clip, c_clip, horizon
    //     );
    // }
}


void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor terminations, torch::Tensor truncations, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0); 
    int horizon = values.size(1)-1; // values has shape[num_agents, horizon+1]
    vtrace_check(values, rewards, terminations, truncations, importance, advantages, num_steps, horizon);
    puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        terminations.data_ptr<float>(), truncations.data_ptr<float>(), importance.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, rho_clip, c_clip, num_steps, horizon
    );
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) terminations, Tensor(d!) truncations, Tensor(e!) importance, Tensor(f!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
 }

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

}

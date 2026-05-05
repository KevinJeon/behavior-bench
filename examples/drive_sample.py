# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

import torch
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl
import torch.nn as nn

import configparser
import os
import numpy as np
from pathlib import Path


from pufferlib.ocean.drive.drive import process_all_maps



#### Sample structure for policy network based on the Default Policy
class Policy(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            num_obs = np.prod(env.single_observation_space.shape)
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )
            
        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_atns), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values

if __name__ == "__main__":

    env_name = 'puffer_drive'
    split = "training"

    #print("Processing maps")
    
    cur_bin_dir = Path(os.path.join(os.environ['DRIVE_BINARIES_DATA_ROOT'], split))
    cur_json_dir = Path(os.path.join(os.environ['DRIVE_DATA_ROOT'], split))
    #process_all_maps(cur_json_dir, cur_bin_dir)


    args = pufferl.load_config(env_name)

    print(args)


    args['env']['num_maps'] = 1000 ## 1000 for GPUDrive Mini and 100 000 for the full dataset


    args['train']['use_rnn'] = "Recurrent"
    #args['env']['split'] = "testing" ## override the split the environment is loaded from

    env_creator = pufferlib.ocean.env_creator(env_name)
    vecenv = pufferlib.vector.make(env_creator, env_kwargs=args['env'],**args['vec'])

    #### Default model
    policy = pufferlib.ocean.torch.Drive(vecenv.driver_env,**args['policy']).cuda()
    policy = pufferlib.ocean.torch.Recurrent(vecenv.driver_env,policy,**args['rnn']).cuda() # -> Some torch module!

    #### Custom model
    #policy = Policy(vecenv.driver_env).cuda()


    train_config = dict(**args['train'], env=env_name)

    args['wandb_project'] = "puffer_drive"
    args['wandb_group'] = "your_wandb_group"
    args['tag'] = "default_sample"

    logger = pufferl.WandbLogger(args)
    trainer = pufferl.PuffeRL(train_config, vecenv, policy,logger)

    while trainer.global_step < train_config['total_timesteps']:
        trainer.evaluate()
        logs = trainer.train() 

    trainer.print_dashboard()
    trainer.close()
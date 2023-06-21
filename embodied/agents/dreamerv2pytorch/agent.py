import sys

import embodied
import numpy as np
import ruamel.yaml as yaml
import torch
from torch import nn

from .world_model import WorldModel
# from .world_model_test import WorldModel
from .actor_critic import ActorCritic
from .utils import action_noise

class Agent(nn.Module):
    configs = yaml.YAML(typ='safe').load((
        embodied.Path(sys.argv[0]).parent / 'configs.yaml').read())
    
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        if self.config.torch.platform == "gpu":
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.world_model = WorldModel(obs_space, self.act_space, config, self.device)
        self.actor_critic = ActorCritic(self.world_model, self.act_space, self.config, self.device)

        self.initial_policy_state = lambda obs: (
            self.world_model.RSSM._init_rssm_state(len(obs['is_first'])),
            torch.zeros((len(obs['is_first']),) + self.act_space.shape).to(self.device))
        
        # path = '/data/home/xyq/gym/Pendulum/torch_run1/'
        # self.actor_critic.load_state_dict(torch.load(path+'actor_critic'))
        # self.world_model.load_state_dict(torch.load(path+'world_model'))

    def policy(self, obs, state=None, mode='train'):
        with torch.no_grad():
            if state is None:
                state = self.initial_policy_state(obs)
            obs = self.preprocess(obs)
            latent, action = state
            embed = self.world_model.obs_encoder(obs['image'])
            _, posterior_rssm_state = self.world_model.RSSM.rssm_observe(
                embed, torch.tensor(action).to(self.device), obs['is_first'].unsqueeze(1), latent)
            model_state = self.world_model.RSSM.get_model_state(posterior_rssm_state)

            noise = self.config.expl_noise
            action = self.actor_critic.actor.act(model_state, mode=mode)
            outs = {'action': action.cpu().detach().numpy()}
            if mode == 'eval':
                noise = self.config.eval_noise
            elif mode == 'train':
                noise = self.config.eval_noise
            action, _ = self.actor_critic.actor(model_state)
            action = action_noise(action, noise, self.act_space)
            outs = {'action': action.cpu().detach().numpy()}
            state = (latent, outs['action'])
            return outs, state

    def train(self, data, state=None):
        data = self.preprocess(data)
        metrics = {}

        # train world model
        state, wm_outs, m1 = self.world_model.train(data)
        context = {**data} #, **wm_outs}
        start = {}
        for k, v in context.items():
            start[k] = v.reshape([-1] + list(v.shape[2:]))
        metrics.update(m1)

        # train actor critic
        m2 = self.actor_critic.train(wm_outs)
        metrics.update(m2)

        outs = {}
        if 'key' in data:
            criteria = {**data} #, **wm_outs}
            outs.update(key=data['key'], priority=criteria[self.config.priority])

        return outs, state, metrics

    def preprocess(self, obs):
        obs = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in obs.items()}
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if len(value.shape) > 3 and value.dtype == torch.uint8:
                value = value.to(torch.float) / 255.0
            else:
                value = value.to(torch.float)
        obs[key] = value
        obs['cont'] = 1.0 - obs['is_terminal'].to(torch.float)
        return obs

    def dataset(self, generator):
        if self.config.data_loader == 'embodied':
            return embodied.Prefetch(
                sources=[generator] * self.config.batch_size,
                workers=8, prefetch=4)

    def report(self, data):
        return {}

    def save(self, path=None):
        if path is None:
            path = self.config.logdir+'/'
        torch.save(self.world_model.state_dict(), path + 'world_model')
        torch.save(self.actor_critic.state_dict(), path + 'actor_critic')
    
    def load(self, values):
        pass
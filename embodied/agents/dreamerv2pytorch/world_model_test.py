import sys

import embodied
import numpy as np
import ruamel.yaml as yaml
import torch
from torch import nn
import torch.optim as optim
import torch.distributions as td
from collections import namedtuple
from typing import Union

from .utils import AutoAdapt, get_parameters, build_model
from .multi_coders import MLP, MultiDecoder, MultiEncoder

RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])  

RSSMState = Union[RSSMDiscState, RSSMContState]

class RSSM(nn.Module):
    def __init__(self, config, embedding_size, act_space, device):
        super().__init__()

        self.device = device
        self.config = config
        self.stoch_size = config.rssm.stoch * config.rssm.classes
        self.deter_size = config.rssm.deter
        self.node_size = config.rssm.units
        self.action_size = act_space.shape[0]
        self.embedding_size = embedding_size
        self.category_size = config.rssm.stoch
        self.class_size = config.rssm.classes

        # self.GRU_embed = nn.Sequential(
        #     nn.Linear(self.stoch_size + self.action_size, self.deter_size),
        #     nn.ELU()
        # ).to(self.device)
        self.GRU_embed = build_model(
            config.rssm.gru_layers, self.stoch_size + self.action_size, self.deter_size, self.node_size
        ).to(self.device)
        self.GRU = nn.GRUCell(self.deter_size, self.deter_size).to(self.device)

        self.posterior = build_model(
            config.rssm.post_layers, self.deter_size + self.embedding_size, self.stoch_size, self.node_size
        ).to(self.device)

        self.prior = build_model(
            config.rssm.prior_layers, self.deter_size, self.stoch_size, self.node_size
        ).to(self.device)

        self.complete_state_size = self.stoch_size + self.deter_size

    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        '''
        imagine one step forward
        (z_{t-1}, a_{t-1}) -> (h_t, z_hat_t)
        '''
        gru_embed = self.GRU_embed(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))
        deter_state = self.GRU(gru_embed, prev_rssm_state.deter*nonterms)
        prior_logit = self.prior(deter_state)
        prior_stoch_state = self.get_stoch_state(prior_logit)
        return RSSMDiscState(prior_logit, prior_stoch_state, deter_state)

    def get_stoch_state(self, logit):
        '''
        use categorial trick
        '''
        shape = logit.shape
        logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
        dist = torch.distributions.OneHotCategorical(logits=logit)        
        stoch = dist.sample()
        stoch += dist.probs - dist.probs.detach()
        return torch.flatten(stoch, start_dim=-2, end_dim=-1)

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        '''
        imagine h forward steps.
        for each step, RSSM computes (z_{t-1}, a_{t-1}) -> (h_t, z_hat_t), actor computes (h_t, z_hat_t) -> a_t
        '''
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        # batch???
        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy
    
    def get_model_state(self, rssm_state):
        return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

    def rssm_stack_states(self, rssm_states, dim):
        return RSSMDiscState(
            torch.stack([state.logit for state in rssm_states], dim=dim),
            torch.stack([state.stoch for state in rssm_states], dim=dim),
            torch.stack([state.deter for state in rssm_states], dim=dim),
        )

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        '''
        gives prior & posterior outputs at the same time.
        '''
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        posterior_logit = self.posterior(x)
        posterior_stoch_state = self.get_stoch_state(posterior_logit)
        posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        priors = []
        posteriors = []

        for t in range(seq_len):
            prev_action = action[:, t] * nonterms[:,t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[:,t], prev_action, nonterms[:,t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state

        prior = self.rssm_stack_states(priors, dim=1)
        post = self.rssm_stack_states(posteriors, dim=1)

        return prior, post

    def _init_rssm_state(self, batch_size, **kwargs):
        return RSSMDiscState(
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
        )

    '''
    below are helper functions
    '''

    def get_dist(self, rssm_state):
        shape = rssm_state.logit.shape
        logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
        return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
 
    def rssm_detach(self, rssm_state):
        return RSSMDiscState(
            rssm_state.logit.detach(),  
            rssm_state.stoch.detach(),
            rssm_state.deter.detach(),
        )
    
    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        return RSSMDiscState(
            seq_to_batch(rssm_state.logit[:, :seq_len]),
            seq_to_batch(rssm_state.stoch[:, :seq_len]),
            seq_to_batch(rssm_state.deter[:, :seq_len])
        )

def seq_to_batch(sequence_data):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, config, device):
        super().__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}

        self.config = config
        self.obs_space = obs_space
        self.wmkl_balance = config.wmkl_balance
        self.loss_scale = config.loss_scales
        self.device = device

        self.obs_encoder = MultiEncoder(device, shapes, **config.encoder)
        self.RSSM = RSSM(config, self.obs_encoder.embed_size, act_space, device)
        self.obs_decoder = MultiDecoder(shapes, self.RSSM.complete_state_size, **config.decoder)
        self.reward_decoder = MLP(self.RSSM.complete_state_size, output_shape=1, **config.reward_head)
        self.discount_decoder = MLP(self.RSSM.complete_state_size,  output_shape=1, **config.cont_head)
        self.models = [self.obs_encoder, self.RSSM, self.reward_decoder, self.obs_decoder, self.discount_decoder]
        # self.models = [self.obs_encoder, self.RSSM, self.obs_decoder]
        for model in self.models:
            model.to(self.device)
        # self.load() # TODO
        self.criterion = nn.MSELoss(reduction='sum') # TODO testing, remove when needed

        self.optim = optim.Adam(get_parameters(self.models), lr=1e-3, eps=config.model_opt.eps)
        self.wmkl = AutoAdapt((), **self.config.wmkl, inverse=False)

        self.cnt = 0
            
    def train(self, data):
        obs = data['image']
        actions = data['action']
        rewards = data['reward'].unsqueeze(-1)
        nonterms = data['cont'].unsqueeze(-1)
        # torch.save(data, 'data')
        # exit(0)

        for i in range(1):
            # forward computing
            embed = self.obs_encoder(obs)
            prev_rssm_state = self.RSSM._init_rssm_state(self.config.batch_size)   # TODO: need extra init?
            prior, posterior = self.RSSM.rollout_observation(
                self.config.replay_chunk, embed, actions, nonterms, prev_rssm_state
            )
            post_modelstate = self.RSSM.get_model_state(posterior) 

            obs_dist = self.obs_decoder(post_modelstate[:,:-1])
            reward_dist = self.reward_decoder(post_modelstate[:,:-1])
            pcont_dist = self.discount_decoder(post_modelstate[:,:-1])

            # obs_res = self.obs_decoder.train(post_modelstate[:,:-1]).reshape(obs.shape[0], -1, obs.shape[2],obs.shape[3],obs.shape[4])
            # reward_res = self.reward_decoder.train(post_modelstate[:,:-1])
            # dis_res = self.discount_decoder.train(post_modelstate[:,:-1])
            
            # calculate loss
            # obs_loss = self.criterion(obs_dist)
            # obs_loss = self._obs_loss(obs_dist, obs[:,:-1])
            # reward_loss = self._reward_loss(reward_dist, rewards[:,1:])
            # pcont_loss = self._pcont_loss(pcont_dist, nonterms[:,1:])

            # obs_loss = self.criterion(obs_res, obs[:,:-1])
            # obs_loss = self._obs_loss(obs_dist, obs[:,:-1])
            obs_loss = self.obs_decoder.loss(obs_dist, obs[:,:-1])
            reward_loss = self.reward_decoder.loss(reward_dist, rewards[:,1:])
            pcont_loss = self.discount_decoder.loss(pcont_dist, nonterms[:,1:])
            # reward_loss = self.criterion(reward_res, rewards[:,:-1])
            # pcont_loss = self.criterion(dis_res, nonterms[:,:-1])
            prior_dist, post_dist, kl_loss = self._kl_loss(prior, posterior, training=True)

            model_loss = self.loss_scale['kl'] * kl_loss  + self.loss_scale['image'] * obs_loss + self.loss_scale['reward'] * reward_loss+ self.loss_scale['cont'] * pcont_loss

            # backward update
            self.optim.zero_grad()
            model_loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.models), self.config.model_opt.clip)
            self.optim.step()

            # record metrics
            metric = {
                "prior_ent_l": torch.mean(prior_dist.entropy()).item(),
                "post_ent_l": torch.mean(post_dist.entropy()).item(),
                "obs_l": obs_loss.item(),
                "model_l": model_loss.item(),
                "reward_l": reward_loss.item(),
                "kl_l": kl_loss.item(),
                "pcont_l": pcont_loss.item(),
            }

            if i % 100 == 0:
                print(i, metric)

            # if i % 1000 == 0:
                # self.save(i)
            
            if i % 10000 == 0:
                # TODO for test
                # forward computing
                # embed = self.obs_encoder(obs)
                # prev_rssm_state = self.RSSM._init_rssm_state(self.config.batch_size)   # TODO: need extra init?
                # prior, posterior = self.RSSM.rollout_observation(
                #     self.config.replay_chunk, embed, actions, nonterms, prev_rssm_state
                # )
                # post_modelstate = self.RSSM.get_model_state(posterior) 

                obs_pred = self.obs_decoder.test(post_modelstate[:,:-1])
                torch.save(obs.detach().cpu(), 'test/contcar/obs_'+str(i)+'.pt')
                torch.save(obs_pred.detach().cpu(), 'test/contcar/pred_'+str(i)+'.pt')
                # prior_modelstate = self.RSSM.get_model_state(prior) 
                # obs_pred2 = self.obs_decoder.test(prior_modelstate[:,:-1])
                # torch.save(obs_pred, 'pred_prior_'+str(self.cnt)+'.pt')

                torch.save(post_modelstate[0].detach().cpu(),'test/contcar/post_state.pt')
                # torch.save(post_modelstate,'test/contcar/post_state_whole.pt')
                # torch.save(prior_modelstate[0],'prior_state.pt')

                torch.save(rewards[:,1:].detach().cpu(), 'test/contcar/reward.pt')
                # torch.save(rewards, 'test/contcar/reward_whole.pt')
                pred_reward = self.reward_decoder.test(post_modelstate[:,:-1])
                torch.save(pred_reward.detach().cpu(), 'test/contcar/pred_reward_'+str(i)+'.pt')
        
                torch.save(nonterms[:,1:].detach().cpu(), 'test/contcar/nonterms.pt')
                pred_dis = self.discount_decoder.test(post_modelstate[:,:-1])
                torch.save(pred_dis.detach().cpu(), 'test/contcar/pred_dis.pt')
        print('done')
        exit(0)

        return data, posterior, metric
        
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior, training=True):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        alpha = self.wmkl_balance
        kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
        kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
        kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        # scale
        kl_loss, mets = self.wmkl(kl_loss, update=training)
        return prior_dist, post_dist, kl_loss

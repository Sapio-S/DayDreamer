import torch
from torch import nn
import torch.optim as optim
import torch.distributions as td

from .multi_coders import MLP
from .utils import AutoAdapt, FreezeParameters, get_parameters, Normalize

class ActorCritic(nn.Module):
    def __init__(self, world_model, act_space, config, device):
        super().__init__()
        self.RSSM = world_model.RSSM
        self.world_model = world_model
        state_size = config.rssm.deter+config.rssm.stoch*config.rssm.classes
        self.actor = Actor(act_space, state_size, config).to(device)
        self.config = config
        self.critic = Critic(state_size, config).to(device)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {'action': self.actor(state)}, carry

    def train(self, posterior):
        actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)

        self.actor.optim.zero_grad()
        self.critic.optim.zero_grad()

        actor_loss.backward()
        value_loss.backward()

        grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100) # self.config.advnorm.max)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.retnorm.max)

        self.actor.optim.step()
        self.critic.optim.step()
        target_info.update({
            'actor_grad_norm': grad_norm_actor.detach().cpu(),
            'critic_grad_norm': grad_norm_value.detach().cpu()
        })

        return target_info

    def actorcritc_loss(self, posterior):
        shp = tuple(posterior.deter.shape)

        batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, shp[0], shp[1]-1))
        
        with FreezeParameters([self.world_model]):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.config.imag_horizon, self.actor, batched_posterior)
            imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
            imag_reward = self.world_model.reward_decoder.run(imag_modelstates)
            discount_arr = self.config.discount * self.world_model.discount_decoder.run(imag_modelstates)
        
        # # v1
        # with FreezeParameters([self.critic.net]):
        #     imag_value = self.critic.net.run(imag_modelstates)
        # imag_value[1:] -=  self.config.actent.scale * policy_entropy[1:].unsqueeze(-1)
        # lambda_returns = cal_returns(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.config.return_lambda)
        # weight = torch.cumprod(torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[:-2]], 0), 0).detach()
        # actor_loss = -torch.mean(weight * lambda_returns)
        # actor_info = {}

        # v2
        with FreezeParameters([self.critic.target_net]):
            imag_value = self.critic.target_net.run(imag_modelstates)
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        weight = torch.cumprod(discount_arr[:-1], 0)

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.config.return_lambda)
        value_loss = self.critic._value_loss(imag_modelstates, weight, lambda_returns) 
        actor_loss, actor_info = self.actor._actor_loss(weight, lambda_returns, policy_entropy, imag_log_prob, imag_value)

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
            'critic_loss':value_loss.item(),
        }
        info.update(actor_info)

        return actor_loss, value_loss, info

class Actor(nn.Module):
    def __init__(self, act_space, state_size, config):
        super().__init__()
        self.act_space = act_space
        self.config = config
        if not act_space.discrete:
            self.model = MLP(input_shape=state_size, output_shape=act_space.shape[0], **self.config.actor, dist=config.actor_dist_cont)
        else:
            self.model = MLP(input_shape=state_size, output_shape=act_space.shape[0], **self.config.actor, dist=config.actor_dist_disc)
        self.optim = optim.Adam(self.model.parameters(), config.actor_opt.lr, eps=config.actor_opt.eps)
        self.grad = (config.actor_grad_disc if act_space.discrete else config.actor_grad_cont)
        self.actent_scale = self.config.actent.scale

        self.advnorm = Normalize(**self.config.advnorm)
        self.retnorm = Normalize(**self.config.retnorm)
        self.scorenorm = Normalize(**self.config.scorenorm)
    
    def forward(self, model_state):
        if self.act_space.discrete:
            action_dist = self.model(model_state)
            action = action_dist.sample()
            action = torch.round(action + action_dist.probs - action_dist.probs.detach()) # straight-through gradients
        else:
            action_dist = self.model(model_state)
            action = action_dist.rsample()
        return action, action_dist

    @torch.no_grad()
    def act(self, state, mode='train'):
        if mode == 'eval':
            action = self.model(state, deterministic=True)
        else:
            action, _ = self.forward(state)
        return action

    def _actor_loss(self, weight, lambda_returns, policy_entropy, imag_log_prob, imag_value):
        if self.grad == 'reinforce':
            policy_entropy = policy_entropy[1:].unsqueeze(-1)
            advantage = (lambda_returns).detach() # can substract the baseline imag_value[:-1]
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage
            actor_loss = -torch.mean(torch.mean(weight * (objective + policy_entropy * self.actent_scale), 0)) 
        elif self.grad == 'backprop':
            lambda_returns = self.retnorm(weight * lambda_returns) # misplaced weight?
            imag_value = self.retnorm(imag_value[:-1], update=False)
            objective = lambda_returns - imag_value
            objective = self.scorenorm(objective)
            score = self.advnorm(torch.mean(objective, -1))
            actor_loss = -torch.mean(score + policy_entropy[1:] * self.actent_scale)
        else:
            raise NotImplementedError

        info = {
            'actor_loss': actor_loss.item(),
            'objective': torch.mean(objective).detach().cpu(),
            'actent': torch.mean(policy_entropy).detach().cpu()
        }
        return actor_loss, info

class Critic(nn.Module):
    def __init__(self, state_size, config):
        super().__init__()
        assert 'action' not in config.critic.inputs, config.critic.inputs
        self.config = config
        self.net = MLP(input_shape=state_size, output_shape=1, **self.config.critic)
        if self.config.slow_target:
            self.target_net = MLP(input_shape=state_size, output_shape=1, **self.config.critic)
            self.updates = -1
        else:
            self.target_net.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), config.critic_opt.lr, eps=config.critic_opt.eps)

    def update_slow(self):
        if not self.config.slow_target:
            return
        initialize = (self.updates == -1)
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates = 0
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)
        self.updates += 1

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.net(value_modelstates) 
        value_loss = self.net.loss(value_dist, value_target, value_discount)
        self.update_slow()
        return value_loss

def compute_return(
                reward: torch.Tensor,
                value: torch.Tensor,
                discount: torch.Tensor,
                bootstrap: torch.Tensor,
                lambda_: float
            ):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns
    
def cal_returns(reward, value, pcont, bootstrap, lambda_):
    """
    Calculate the target value, following equation (5-6) in Dreamer
    :param reward, value: imagined rewards and values, dim=[horizon, (chuck-1)*batch, reward/value_shape]
    :param bootstrap: the last predicted value, dim=[(chuck-1)*batch, 1(value_dim)]
    :param pcont: gamma
    :param lambda_: lambda
    :return: the target value, dim=[horizon, (chuck-1)*batch, value_shape]
    """
    assert list(reward.shape) == list(value.shape), "The shape of reward and value should be similar"
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)

    next_value = torch.cat((value[1:], bootstrap[None]), 0)  # bootstrap[None] is used to extend additional dim
    inputs = reward + pcont * next_value * (1 - lambda_)  # dim=[horizon, (chuck-1)*B, 1]
    outputs = []
    last = bootstrap

    for t in reversed(range(reward.shape[0])): # for t in horizon
        inp = inputs[t]
        last = inp + pcont[t] * lambda_ * last
        outputs.append(last)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns
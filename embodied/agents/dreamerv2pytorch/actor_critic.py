import torch
from torch import nn
import torch.optim as optim
import torch.distributions as td

from .multi_coders import MLP
from .utils import AutoAdapt, FreezeParameters

class ActorCritic(nn.Module):
    def __init__(self, world_model, act_space, config, device):
        super().__init__()
        self.RSSM = world_model.RSSM
        self.world_model = world_model
        state_size = config.rssm.deter+config.rssm.stoch*config.rssm.classes
        self.actor = Actor(act_space, state_size, config).to(device)
        self.config = config
        rewfn = lambda s: world_model.reward_decoder(s).mean()[1:]
        # if config.critic_type == 'vfunction':
        #     critics = {'extr': Critic(rewfn, config)}
        self.critic = Critic(rewfn, state_size, config).to(device)

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

        grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.advnorm.max)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.retnorm.max)

        self.actor.optim.step()
        self.critic.optim.step()

        return target_info

    def actorcritc_loss(self, posterior):
        shp = tuple(posterior.deter.shape)

        batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, shp[0], shp[1]-1))
        
        with FreezeParameters([self.world_model]):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.config.imag_horizon, self.actor, batched_posterior)
            imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
            imag_reward = self.world_model.reward_decoder.run(imag_modelstates)
            # discount_arr = self.config.discount * torch.ones_like(imag_reward) 
            discount_arr = self.config.discount * self.world_model.discount_decoder.run(imag_modelstates)
        
        imag_value = self.critic.target_net.run(imag_modelstates)

        actor_loss, discount, lambda_returns = self.actor._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self.critic._value_loss(imag_modelstates, discount, lambda_returns)     

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
            'actor_loss':actor_loss.item(),
            'critic_loss':value_loss.item(),
        }

        return actor_loss, value_loss, target_info

class Actor(nn.Module):
    def __init__(self, act_space, state_size, config):
        super().__init__()
        self.act_space = act_space
        self.config = config
        self.model = MLP(input_shape=state_size,output_shape=act_space.shape[0], **self.config.actor, dist=(
            config.actor_dist_disc if act_space.discrete
            else config.actor_dist_cont))
        self.actent = AutoAdapt(
            act_space.shape[:-1] if act_space.discrete else act_space.shape,
            **self.config.actent, inverse=True)
        self.optim = optim.Adam(self.model.parameters(), config.actor_opt.lr, eps=config.actor_opt.eps)

    def forward(self, model_state):
        action_dist = self.model(model_state)
        # change precision to prevent precision loss
        if self.config.actor_grad_disc == 'dynamics':
            action = action_dist.sample()
        if self.config.actor_grad_disc == 'reinforce': # TODO: check?
            action = action_dist.sample()
            # action = action + action_dist.probs.to(torch.float64) - action_dist.probs.detach().to(torch.float64)
        return action, action_dist

    @torch.no_grad()
    def act(self, state, mode='train'):
        action = self.model.run(state)
        return action

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.config.return_lambda)
        
        if self.config.actor_grad_disc == 'reinforce':
            advantage = (lambda_returns-imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage
        elif self.config.actor_grad_disc == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError   

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)

        actor_loss = -torch.sum(torch.mean(discount * (objective + self.config.critic.outscale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

class Critic(nn.Module):
    def __init__(self, rewfn, state_size, config):
        super().__init__()
        self.rewfn = rewfn
        assert 'action' not in config.critic.inputs, config.critic.inputs
        self.config = config
        self.net = MLP(input_shape=state_size, output_shape=1, **self.config.critic)
        if self.config.slow_target:
            self.target_net = MLP(input_shape=state_size, output_shape=1, **self.config.critic)
            self.updates = -1
        else:
            self.target_net.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), config.critic_opt.lr, eps=config.critic_opt.eps)

    def score(self, traj, actor):
        return self.target(traj, self.rewfn(traj), self.config.actor_return)

    def update_slow(self):
        if not self.config.slow_target:
            return
        # assert self.net.variables
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
    
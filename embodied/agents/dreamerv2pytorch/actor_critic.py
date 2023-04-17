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

        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, shp[0], shp[1]-1))
        
        with FreezeParameters([self.RSSM]):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.config.imag_horizon, self.actor, batched_posterior)
        
        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters([self.world_model]):
            imag_reward_dist = self.world_model.reward_decoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            discount_dist = self.world_model.discount_decoder(imag_modelstates)
            discount_arr = self.config.discount*torch.round(discount_dist.base_dist.probs)
        
        with FreezeParameters([self.critic]):
            imag_value_dist = self.critic.target_net(imag_modelstates)
            imag_value = imag_value_dist.mean

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
        action = action_dist.sample().to(torch.float64)
        action = action + action_dist.probs.to(torch.float64) - action_dist.probs.detach().to(torch.float64)
        return action.to(torch.float32), action_dist

    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action

        raise NotImplementedError

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
        # if self.config.actent_norm:
        #     lo = policy.minent / np.prod(self.actent.shape)
        #     hi = policy.maxent / np.prod(self.actent.shape)
        #     ent = (ent - lo) / (hi - lo)
        # ent_loss, mets = self.actent(policy_entropy)
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

    def train(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target = tf.stop_gradient(self.target(
            traj, reward, self.config.critic_return)[0])
        with tf.GradientTape() as tape:
            dist = self.net({k: v[:-1] for k, v in traj.items()})
            loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
        metrics.update(self.opt(tape, loss, self.net))
        metrics.update({
            'critic_loss': loss,
            'imag_reward_mean': reward.mean(),
            'imag_reward_std': reward.std(),
            'imag_critic_mean': dist.mean().mean(),
            'imag_critic_std': dist.mean().std(),
            'imag_return_mean': target.mean(),
            'imag_return_std': target.std(),
        })
        self.update_slow()
        return metrics

    def score(self, traj, actor):
        return self.target(traj, self.rewfn(traj), self.config.actor_return)

    def target(self, traj, reward, impl):
        if len(reward) != len(traj['action']) - 1:
            raise AssertionError('Should provide rewards for all but last action.')
        disc = traj['cont'][1:] * self.config.discount
        value = self.target_net(traj).mean()
        if impl == 'gae':
            advs = [tf.zeros_like(value[0])]
            deltas = reward + disc * value[1:] - value[:-1]
            for t in reversed(range(len(disc))):
                advs.append(deltas[t] + disc[t] * self.config.return_lambda * advs[-1])
            adv = tf.stack(list(reversed(advs))[:-1])
            return adv + value[:-1], value[:-1]
        elif impl == 'gve':
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            ret = tf.stack(list(reversed(vals))[:-1])
            return ret, value[:-1]
        else:
            raise NotImplementedError(impl)


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
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
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
    
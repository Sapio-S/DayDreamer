import torch
from torch import nn
from typing import Iterable
import numpy as np

class AutoAdapt():

  def __init__(
        self, shape, impl, scale, target, min, max,
        vel=0.1, thres=0.1, inverse=False):
    self._shape = tuple(shape)
    self._impl = impl
    self._target = target
    self._min = min
    self._max = max
    self._vel = vel
    self._inverse = inverse
    self._thres = thres
    if self._impl == 'fixed':
        self._scale = torch.tensor(scale)
    elif self._impl == 'mult':
        self._scale = torch.ones(shape, dtype=torch.float32, requires_grad=False)
    elif self._impl == 'prop':
        self._scale = torch.ones(shape, dtype=torch.float32, requires_grad=False)
    else:
        raise NotImplementedError(self._impl)

  @property
  def shape(self):
    return self._shape

  def __call__(self, reg, update=True):
    update and self.update(reg)
    scale = self.scale()
    loss = scale * (-reg if self._inverse else reg)
    metrics = {
        'mean': reg.mean(), 'std': reg.std(),
        'scale_mean': scale.mean(), 'scale_std': scale.std()}
    return loss, metrics

  def scale(self):
    if self._impl == 'fixed':
        scale = self._scale
    elif self._impl == 'mult':
        scale = self._scale
    elif self._impl == 'prop':
        scale = self._scale
    else:
        raise NotImplementedError(self._impl)
    return torch.tensor(scale).detach()

  def update(self, reg):
    avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
    if self._impl == 'fixed':
        pass
    elif self._impl == 'mult':
        below = avg < (1 / (1 + self._thres)) * self._target
        above = avg > (1 + self._thres) * self._target
        if self._inverse:
            below, above = above, below
        inside = ~below & ~above
        adjusted = (
            above.to(torch.float32) * self._scale * (1 + self._vel) +
            below.to(torch.float32) * self._scale / (1 + self._vel) +
            inside.to(torch.float32) * self._scale)
        self._scale = torch.clamp(adjusted, self._min, self._max)
    elif self._impl == 'prop':
        direction = avg - self._target
        if self._inverse:
            direction = -direction
        self._scale = torch.clamp(adjusted, self._min, self._max)
    else:
        raise NotImplementedError(self._impl)

def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    amount = amount.to(action.dtype)
    if act_space.discrete:
        probs = amount / action.shape[-1] + (1 - amount) * action
        sample = torch.distributions.OneHotCategorical(probs=probs).sample()
        return sample
    else:
        return torch.clamp(torch.distributions.Normal(action, amount).sample(), -1, 1)

def build_model(num_layer, input_size, output_size, hidden_size):
    if num_layer == 1:
        model = [nn.Linear(input_size, output_size)]
        model += [nn.LayerNorm(output_size)]                # TODO: check if needed
        model += [nn.ELU()]                                 # TODO: check if needed
    else:
        model = [nn.Linear(input_size, hidden_size)]
        model += [nn.LayerNorm(hidden_size)]
        model += [nn.ELU()]
        for i in range(num_layer-2):
            model += [nn.Linear(hidden_size,hidden_size)]
            model += [nn.LayerNorm(hidden_size)]
            model += [nn.ELU()]
        model += [nn.Linear(hidden_size, int(np.prod(output_size)))]
        model += [nn.LayerNorm(int(np.prod(output_size)))]  # TODO: check if needed
        model += [nn.ELU()]                                 # TODO: check if needed
    return nn.Sequential(*model)
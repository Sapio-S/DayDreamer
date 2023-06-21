
import re
import numpy as np
import torch.nn as nn
import torch.distributions as td
import torch
from .utils import build_model
import torch.nn.functional as F

# def normal_tanh(x, min_std=0.03, max_std=1.0):
#     # Normal(tanh(x))
#     mean_, std_ = x.chunk(2, -1)
#     mean = torch.tanh(mean_)
#     std = (max_std - min_std) * torch.sigmoid(std_) + min_std
#     normal = td.normal.Normal(mean, std)
#     normal = td.independent.Independent(normal, 1)
#     return normal

def normal_tanh(mean, std, min_std=0.03, max_std=1.0):
    # Normal(tanh(x))
    mean = torch.tanh(mean)
    std = (max_std - min_std) * torch.sigmoid(std) + min_std
    normal = td.normal.Normal(mean, std)
    normal = td.independent.Independent(normal, 1)
    return normal
    
def tanh_normal(x):
    # TanhTransform(Normal(5 tanh(x/5)))
    mean_, std_ = x.chunk(2, -1)
    mean = 5 * torch.tanh(mean_ / 5)  # clip tanh arg to (-5, 5)
    std = F.softplus(std_) + 0.1  # min_std = 0.1
    normal = td.normal.Normal(mean, std)
    normal = td.independent.Independent(normal, 1)
    tanh = td.TransformedDistribution(normal, [td.TanhTransform()])
    tanh.entropy = normal.entropy  # HACK: need to implement correct tanh.entorpy (need Jacobian of TanhTransform?)
    return tanh

def normal(x, min_std=0.03, max_std=1.0):
    mean, std_ = x.chunk(2, -1)
    std = (max_std - min_std) * torch.sigmoid(std_) + min_std
    normal = td.normal.Normal(mean, std)
    normal = td.independent.Independent(normal, 1)
    return normal

class MultiEncoder(nn.Module):

    def __init__(
        self, device, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
        mlp_units=512, cnn='simple', cnn_depth=48, cnn_kernels=(4, 4, 4, 4),
        cnn_blocks=2, **kw):
        super().__init__()

        excluded = ('is_first', 'is_last')
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items()
            if re.match(cnn_keys, k) and len(v) == 3}
        self.mlp_shapes = {
            k: v for k, v in shapes.items()
            if re.match(mlp_keys, k) and len(v) in (0, 1)}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)
        if self.cnn_shapes:
            self._cnn = ImageEncoder(cnn_depth, cnn_kernels, self.cnn_shapes, device, **kw)
            self.embed_size = self._cnn.test_shape()[-1]
        elif self.mlp_shapes:
            self._mlp = MLP(self.mlp_shapes['image'][-1], mlp_layers, mlp_units, self.mlp_shapes, dist=None, has_layer_norm=True, **kw)
            self.embed_size = mlp_units

    def forward(self, data):
        outputs = []
        if self.cnn_shapes:
            x = self._cnn(data)
            outputs.append(x)
        elif self.mlp_shapes:
            outputs.append(self._mlp(data))
        outputs = torch.concat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):

    def __init__(
        self, shapes, state_size, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
        mlp_layers=4, mlp_units=512, cnn='simple', cnn_depth=48,
        cnn_kernels=(5, 5, 6, 6), cnn_blocks=2, image_dist='mse', **kw):

        super().__init__()
        excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items()
            if re.match(cnn_keys, k) and len(v) == 3}
        self.mlp_shapes = {
            k: v for k, v in shapes.items()
            if re.match(mlp_keys, k) and len(v) == 1}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Decoder CNN shapes:', self.cnn_shapes)
        print('Decoder MLP shapes:', self.mlp_shapes)

        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            merged = shapes[0][:-1] + (sum(x[-1] for x in shapes),) #(64, 64, 1)
            self.input_shape = merged
            if cnn == 'simple':
                self._cnn = ImageDecoder(merged, cnn_depth, cnn_kernels, **kw)
                self._cnn_linear = nn.Linear(state_size, np.prod(self._cnn.conv_shape))
            else:
                raise NotImplementedError(cnn)
            self._image_dist = image_dist

        elif self.mlp_shapes:
            self._mlp = MLP(state_size, mlp_layers, mlp_units, self.mlp_shapes, output_shape=self.mlp_shapes['image'][-1], dist=image_dist, has_layer_norm=True, **kw)

    def forward(self, inputs):
        features = inputs
        if self.cnn_shapes:
            batch_shape = features.shape[:-1]
            flat = features.reshape([-1, features.shape[-1]])
            x = self._cnn_linear(flat).reshape(-1, *self._cnn.conv_shape)
            output = self._cnn(x)
            mean = torch.reshape(output, (*batch_shape, *self.input_shape))
            obs_dist = td.Independent(td.Normal(mean, 1), len(self.input_shape))
            return obs_dist
        elif self.mlp_shapes:
            dist = self._mlp(features)
            return dist

    def loss(self, dist, target):
        if self.cnn_shapes:
            obs_loss = -torch.mean(dist.log_prob(obs))
            return obs_loss
        else:
            return self._mlp.loss(dist, target)

    @torch.no_grad()
    def test(self, input):
        if self.cnn_shapes:
            batch_shape = features.shape[:-1]
            flat = features.reshape([-1, features.shape[-1]])
            x = self._cnn_linear(flat).reshape(-1, *self._cnn.conv_shape)
            output = self._cnn(x)
            return output
        else:
            return self._mlp.test(input)


class MLP(nn.Module):

    def __init__(self, input_shape=None, layers=4, units=512, inputs=['tensor'], dims=None, mlp_shape=None, output_shape=None, has_layer_norm=True, **kw):
        assert input_shape is None or isinstance(input_shape, (int, tuple, dict)), input_shape
        super().__init__()
        if input_shape is None:
            input_shape = units
        if output_shape is None:
            output_shape = units

        self._layers = layers
        self._units = units
        self._input_shape = input_shape
        self._output_shape = output_shape

        distkeys = ('dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix')
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist = {k: v for k, v in kw.items() if k in distkeys}
        if self._dist['dist'] == 'normal':
            self.model = build_model(
                self._layers, input_shape, int(np.prod(self._output_shape)), self._units, has_layer_norm
            )
            self.std = nn.Linear(input_shape, int(np.prod(self._output_shape)))
        else:
            self.model = build_model(
                self._layers, input_shape, int(np.prod(self._output_shape)), self._units, has_layer_norm
            )

    def forward(self, input, deterministic=False):
        dist_inputs = self.model(input)
        if self._dist['dist'] == 'normal':
            if deterministic:
                # mean_, std_ = dist_inputs.chunk(2, -1)
                mean = torch.tanh(dist_inputs)
                return mean
            std = self.std(input)
            dist = normal_tanh(dist_inputs, std)
            # dist = tanh_normal(dist_inputs)
            # dist = normal(dist_inputs)
            return dist
        if self._dist['dist'] == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=dist_inputs), 1)
        if self._dist['dist'] == 'symlog':
            return dist_inputs
        if self._dist['dist'] == 'mse':
            return dist_inputs
        if self._dist['dist'] == 'onehot':
            if deterministic:
                return td.OneHotCategorical(logits=dist_inputs).sample()
            # return dist_inputs
            return td.OneHotCategorical(logits=dist_inputs)
        if self._dist['dist'] == None:
            return dist_inputs

        raise NotImplementedError(self._dist)
    
    @torch.no_grad()
    def test(self, input):
        if self._dist['dist'] == 'symlog':
            return symexp(self.model(input))
        return self.model(input) # not suitable for binary

    def run(self, input):
        # if self._dist['dist'] == 'normal': # should not be called, used for actor only
        output = self.forward(input)
        if self._dist['dist'] == 'binary':
            return torch.round(output.base_dist.probs)
        if self._dist['dist'] == 'symlog':
            return symexp(output)
        if self._dist['dist'] == 'mse':
            return output
        # if self._dist['dist'] == 'onehot': # should not be called, used for actor only
        #     return output.sample()
        if self._dist['dist'] == None:
            return output

        raise NotImplementedError(self._dist)

    def loss(self, dist, target, discount=None):
        # if self._dist['dist'] == 'normal': # should not be called, used for actor only
        if self._dist['dist'] == 'binary':
            return -torch.mean(dist.log_prob(target))
        if self._dist['dist'] == 'symlog':
            target = symlog(target)
            if discount is not None:
                return torch.mean((dist-target)**2 * discount)
            return torch.mean((dist - target) ** 2)
        if self._dist['dist'] == 'mse':
            if discount is not None:
                return torch.mean((dist-target)**2 * discount)
            return torch.mean((dist - target) ** 2)
        # if self._dist['dist'] == 'onehot': # should not be called, used for actor only
        #     return -torch.sum(dist.log_prob(target))
        if self._dist['dist'] == None:
            return torch.sum((dist - target) ** 2)

        raise NotImplementedError(self._dist)
        
class ImageEncoder(nn.Module):

    def __init__(self, depth, kernels, cnn_shape, device, **kw):
        self._depth = depth
        self._kernels = kernels
        self._kw = kw
        self.device = device
        self.shape = cnn_shape['image']
        stride = 2 # default

        super(ImageEncoder, self).__init__()
        input_shape = cnn_shape['image'][-1]
        convolutions = []
        # depth //= 2 ** (len(kernels)-1)
        convolutions.append(nn.Conv2d(input_shape, depth, self._kernels[0], stride))
        # convolutions.append(nn.LayerNorm(?))              # TODO: check if needed
        convolutions.append(nn.ELU())
        for k in self._kernels[1:]:
            convolutions.append(nn.Conv2d(depth, 2*depth, k, stride))
            # convolutions.append(nn.LayerNorm(?))          # TODO: check if needed
            convolutions.append(nn.ELU())
            depth *= 2
        convolutions.append(nn.Flatten())
        self.convolutions = nn.Sequential(*convolutions)
        self.convolutions.to(device)

    def test_shape(self):
        obs = torch.zeros(1,*self.shape).to(self.device).permute(0, 3, 1, 2)
        return self.convolutions(obs).shape

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        obs = torch.reshape(obs, (-1, *img_shape)).permute(0, 3, 1, 2)
        embed = self.convolutions(obs)
        embed = torch.reshape(embed, (*batch_shape, -1))
        
        return embed

class ImageDecoder(nn.Module):

    def __init__(self, shape, depth, kernels, **kw):
        self._shape = shape
        self._depth = depth
        self._kernels = kernels
        self._kw = kw
        stride = 2

        super(ImageDecoder, self).__init__()
        c = shape[-1]
        # decoder = [nn.Unflatten(-1, (depth, 1, 1)),]
        decoder = []
        depth = self._depth * 2 ** (len(kernels)-1)
        for i in range(len(kernels)-1):
            decoder.append(nn.ConvTranspose2d(
                depth, depth // 2, kernels[i], stride
                ))
            # decoder.append(nn.LayerNorm(?))                       # TODO: check if needed
            decoder.append(nn.ELU())
            depth //= 2
        decoder.append(nn.ConvTranspose2d(depth, c, kernels[-1], stride))
        # decoder.append(nn.LayerNorm(?))                           # TODO: check if needed
        # decoder.append(nn.ELU())
        self.decoder = nn.Sequential(*decoder)

        output_shape = (shape[2],shape[0],shape[1])
        conv_shape = conv_out_shape(output_shape[1:], 0, kernels[-1], stride)
        for i in range(len(kernels)-1):
            kernel = kernels[len(kernels)-2-i]
            conv_shape = conv_out_shape(conv_shape, 0, kernel, stride)
        self.conv_shape = (self._depth * 2 ** (len(kernels)-1), *conv_shape)

    def forward(self, x):
        mean = self.decoder(x)
        return mean

    def test_shape(self):
        input_data = torch.zeros(1,)

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def symlog(x):
  return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
  return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
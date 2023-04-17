
import re
import numpy as np
import torch.nn as nn
import torch.distributions as td
import torch

from .utils import build_model

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
            # out = self._cnn(torch.randn(self.cnn_shapes['image'], device=device).unsqueeze(0)).reshape(1,-1)
            # self._cnn_mlp = MLP(out.shape[1], mlp_layers, mlp_units, dist=None)
            # self._cnn_mlp = nn.Linear(out.shape[1], mlp_units)
        else:
            raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, self.mlp_shapes, dist='none', **kw)
            self.embed_size = mlp_units

    def forward(self, data):
        # some_key, some_shape = list(self.shapes.items())[0]
        # batch_dims = tuple(data[some_key].shape[:-len(some_shape)])
        # data = {
        #     k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        #     for k, v in data.items()}
        outputs = []
        if self.cnn_shapes:
            x = self._cnn(data)
            # output = self._cnn_mlp(x)
            # inputs = torch.concat([data[k] for k in self.cnn_shapes], -1)
            # output = self._cnn(inputs)
            # output = output.reshape((output.shape[0], -1))
            outputs.append(x)
        if self.mlp_shapes:
            # inputs = [
            #     data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
            #     for k in self.mlp_shapes]
            # inputs = torch.concat(inputs, -1)
            # outputs.append(self._mlp(inputs))
            outputs.append(self._mlp(data))
        outputs = torch.concat(outputs, -1)
        # outputs = outputs.reshape(batch_dims + outputs.shape[1:])
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
            # elif cnn == 'resnet':
            #     self._cnn = ImageDecoderResnet(merged, cnn_depth, cnn_blocks, **kw)
            else:
                raise NotImplementedError(cnn)
            self._image_dist = image_dist

        if self.mlp_shapes:
            self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, **kw)
        # self._inputs = Input(inputs)
        

    def forward(self, inputs):
        features = inputs
        # features = self._inputs(inputs)
        # dists = {}
        if self.cnn_shapes:
            batch_shape = features.shape[:-1]
            flat = features.reshape([-1, features.shape[-1]])
            x = self._cnn_linear(flat).reshape(-1, *self._cnn.conv_shape)
            output = self._cnn(x)
            # output = output.reshape(features.shape[:-1] + output.shape[1:])
            # means = torch.split(output, [v[-1] for v in self.cnn_shapes.values()], -1)
            # dists.update({
            #     key: self._make_image_dist(key, mean)
            #     for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
            mean = torch.reshape(output, (*batch_shape, *self.input_shape))
            obs_dist = td.Independent(td.Normal(mean, 1), len(self.input_shape))
            return obs_dist
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.to(torch.float32)
        if self._image_dist == 'normal':
            normal = torch.distributions.normal.Normal(mean, 1)
            return torch.distributions.independent.Independent(normal, 3)
        # if self._image_dist == 'mse':
        #     return nn.MSELoss(mean, 3, 'sum')
        raise NotImplementedError(self._image_dist)

    @torch.no_grad()
    def test(self, features):
        if self.cnn_shapes:
            batch_shape = features.shape[:-1]
            flat = features.reshape([-1, features.shape[-1]])
            x = self._cnn_linear(flat).reshape(-1, *self._cnn.conv_shape)
            output = self._cnn(x)
            # output = output.reshape(features.shape[:-1] + output.shape[1:])
            # means = torch.split(output, [v[-1] for v in self.cnn_shapes.values()], -1)
            # dists.update({
            #     key: self._make_image_dist(key, mean)
            #     for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
            # mean = torch.reshape(output, (*batch_shape, *self.input_shape))
            # obs_dist = td.Independent(td.Normal(mean, 1), len(self.input_shape))
            return output

    def train(self, features):
        if self.cnn_shapes:
            batch_shape = features.shape[:-1]
            flat = features.reshape([-1, features.shape[-1]])
            x = self._cnn_linear(flat).reshape(-1, *self._cnn.conv_shape)
            output = self._cnn(x)
            return output  


class MLP(nn.Module):

    def __init__(self, input_shape=None, layers=4, units=512, inputs=['tensor'], dims=None, mlp_shape=None, output_shape=None, **kw):
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
        self.model = build_model(
            self._layers, input_shape, int(np.prod(self._output_shape)), self._units
        )

    def forward(self, input):
        dist_inputs = self.model(input)
        if self._dist['dist'] == 'normal':
            return td.independent.Independent(td.Normal(dist_inputs, 1), 1)
        if self._dist['dist'] == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=dist_inputs), 1)
        if self._dist['dist'] == 'symlog':
            return td.independent.Independent(td.Normal(dist_inputs, 1), 1)
        if self._dist['dist'] == 'mse':
            return td.independent.Independent(td.Normal(dist_inputs, 1), 1)
        if self._dist['dist'] == 'onehot':
            return td.OneHotCategorical(logits=dist_inputs)  # td.independent.Independent(td.Normal(dist_inputs, 1), 1)
        if self._dist['dist'] == None:
            return dist_inputs

        raise NotImplementedError(self._dist)
    
    @torch.no_grad()
    def test(self, input):
        return self.model(input)

    def train(self, input):
        return self.model(input)
        
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

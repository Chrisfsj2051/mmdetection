import torch
import torch.nn as nn
import torch.nn.modules.batchnorm
import torchvision
import torch.nn.functional as F

# base
# weight & gamma
# running_mean & running_var
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose


def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        var_val = x.var([0, 2, 3], unbiased=False)

    x = x - mean_val[None, ..., None, None]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x


def create_inputs():
    return torch.randn(1, 3, 20, 20)


transform = ToTensor()
mnist = torchvision.datasets.MNIST(root='mnist', download=True, transform=transform)
dataloader = DataLoader(dataset=mnist, batch_size=8)

toy_model = nn.Sequential(nn.Linear(28 ** 2, 128), nn.BatchNorm1d(128),
                          nn.ReLU(), nn.Linear(128, 10), nn.Sigmoid())
optimizer = torch.optim.SGD(toy_model.parameters(), lr=0.01)
bn_layer = toy_model[1]
print(f'Initial weight is {bn_layer.weight[:4].tolist()}...')
print(f'Initial bias is {bn_layer.bias[:4].tolist()}...')
toy_model.named_parameters()
for (i, data) in enumerate(dataloader):
    output = toy_model(data[0].view(data[0].shape[0], -1))
    (F.cross_entropy(output, data[1])).backward()
    print(f'Gradient of weight is {bn_layer.weight.grad[:4].tolist()}...')
    print(f'Gradient of bias is {bn_layer.bias.grad[:4].tolist()}...')
    optimizer.step()
    optimizer.zero_grad()
    if i > 1:
        break
print(f'Now weight is {bn_layer.weight[:4].tolist()}...')
print(f'Now bias is {bn_layer.bias[:4].tolist()}...')

inputs = torch.randn(4, 128)
bn_outputs = bn_layer(inputs)
new_bn = nn.BatchNorm1d(128)
bn_outputs_no_weight_bias = new_bn(inputs)

assert not torch.allclose(bn_outputs, bn_outputs_no_weight_bias)
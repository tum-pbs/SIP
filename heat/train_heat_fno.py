""" Fourier Neural Operator (FNO) version
"""
import os
import time

from inv_diffuse import generate_heat_example, inv_diffuse
from phi.torch.flow import *
from os.path import expanduser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


math.set_global_precision(64)
assert backend.default_backend().set_default_device('GPU')


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, in_features, out_features, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_features)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = torch.permute(x, (0, 3, 1, 2))
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


for seed in range(1):
    math.seed(seed)
    net = FNO2d(1, 1, modes1=12, modes2=12, width=32).to(TORCH.get_default_device().ref)  # Default values from GitHub are (12, 12, 32)  https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    # 12, 32 -> 1.2 M parameters (1188353)
    # 12, 16 -> 300 k parameters
    # Ref U-Net 38 k
    # width=16 initially faster in time but flattens
    # keep width > modes
    # lr 1e-4 best
    print(parameter_count(net))
    os.path.exists(expanduser(f"~/phi/heat_net2/{seed}")) or os.mkdir(expanduser(f"~/phi/heat_net2/{seed}"))
    torch.save(net.state_dict(), expanduser(f"~/phi/heat_net2/{seed}/init.pth"))

    for mi, method in enumerate(['FNO']):  # 'SGD', 'Adam + PG', 'Adam'
        for learning_rate in [1e-5]:
            for batch_size in [128]:
                BATCH = batch(batch=batch_size)
                scene = Scene.create(f"~/phi/heat_net2/{seed}", name=f"{method}_lr_{learning_rate}_bs{batch_size}")
                print(scene)
                viewer = view('x_gt, y_target, x, y, dx, amp, raw_kernel, kernel, sig_prob', scene=scene, port=8001 + mi, select='batch', gui='console')
                torch.save(net.state_dict(), viewer.scene.subpath(f'net_init.pth'))

                net.load_state_dict(torch.load(expanduser(f"~/phi/heat_net2/{seed}/init.pth")))
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                math.seed(0)
                viewer.info(f"Training method: {method}")
                start_time = time.perf_counter()
                for training_step in viewer.range():
                    optimizer.zero_grad()
                    x_gt = generate_heat_example(spatial(x=64, y=64), BATCH)
                    y_target = diffuse.fourier(x_gt, 8., 1)
                    with math.precision(32):
                        prediction = field.native_call(net, field.to_float(y_target)).vector[0]
                        prediction += field.mean(x_gt) - field.mean(prediction)
                    prediction = field.to_float(prediction)  # is this call necessary?
                    x = field.stop_gradient(prediction)
                    if not field.isfinite(prediction):
                        raise RuntimeError(net.state_dict())
                    y = diffuse.fourier(prediction, 8., 1)
                    y_l2 = loss = field.l2_loss(y - y_target)
                    loss.sum.backward()
                    viewer.log_scalars(x_l1=field.l1_loss(x_gt - x), y_l1=field.l1_loss(y_target - y), y_l2=y_l2)
                    optimizer.step()
                    if time.perf_counter() - start_time > 60 * 60 * 6:  # time limit
                        break
                torch.save(net.state_dict(), viewer.scene.subpath(f'net_{method}.pth'))
print("All done.")
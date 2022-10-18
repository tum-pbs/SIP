import os
import time

from os.path import expanduser

import torch_kfac
from torch_optimizer import Adahessian
from hessianfree.optimizer import HessianFree


from phi.torch.flow import *


TORCH.set_default_device('GPU')

for seed in range(1):
    math.seed(seed)
    net = u_net(1, 1)
    os.path.exists(expanduser(f"~/phi/poisson_net2/{seed}")) or os.mkdir(expanduser(f"~/phi/poisson_net2/{seed}"))
    torch.save(net.state_dict(), expanduser(f"~/phi/poisson_net2/{seed}/init.pth"))
    
    for method in ['SIP', 'SGD', 'Adam', 'AdaHessian', 'HessianFreeGGN', 'kFac']:
        scene = Scene.create(f"~/phi/poisson_net2/{seed}", name=method)
        print(scene)
        viewer = view(scene=scene, select='batch', gui='console')

        if method == 'kFac':
            optimizer = torch_kfac.KFAC(net, learning_rate=1e-3, damping=1e-3)  # KFAC doesn't work
        elif method == 'AdaHessian':
            optimizer = Adahessian(net.parameters(), lr=1e-6)  # 1e-4 diverges, 1e-5 too large, 1e-6 stable but bad, 1e-7 in paper
        elif method == 'HessianFreeH':
            optimizer = HessianFree(net.parameters(), verbose=True, curvature_opt='hessian')
        elif method == 'HessianFreeGGN':
            optimizer = HessianFree(net.parameters(), verbose=True, curvature_opt='ggn')
        elif method == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=3e-12, momentum=0.9)  # 1e-10 diverges, 3e-12 not stable
        else:
            optimizer = optim.Adam(net.parameters(), lr=1e-3)

        net.load_state_dict(torch.load(expanduser(f"~/phi/poisson_net2/{seed}/init.pth")))
        math.seed(0)
        viewer.info(f"Training method: {method}")
        start_time = time.perf_counter()
        for training_step in viewer.range():
            if method == 'kFac':
                net.zero_grad()
            else:
                optimizer.zero_grad()
            x_gt = CenteredGrid(Noise(batch(batch=128)), x=64, y=64)
            y_target = field.solve_linear(field.laplace, x_gt, Solve('CG', 1e-5, 0, x0=x_gt * 0))
            if method in ('Adam', 'SGD'):
                prediction = field.native_call(net, y_target)
                y = field.solve_linear(field.laplace, prediction, Solve('CG', 1e-5, 0, x0=x_gt * 0))
                loss = y_l2 = field.l2_loss(y - y_target)
                loss.sum.backward()
                optimizer.step()
            elif method in ['HessianFreeGGN', 'HessianFreeH']:
                prediction = field.native_call(net, y_target)
                y = field.solve_linear(field.laplace, prediction, Solve('CG', 1e-5, 0, x0=x_gt * 0))
                y_l2 = field.l2_loss(y - y_target)
                def forward():
                    prediction = field.native_call(net, y_target)
                    y = field.solve_linear(field.laplace, prediction, Solve('CG', 1e-5, 0, x0=x_gt * 0))
                    loss = field.l2_loss(y - y_target)
                    return loss.sum, prediction.values.native(prediction.values.shape)
                optimizer.step(forward=forward)
            elif method == 'SIP':
                prediction = field.native_call(net, y_target)
                x = field.stop_gradient(prediction)
                y = field.solve_linear(field.laplace, x, Solve('CG', 1e-5, 0, x0=x_gt * 0))
                y_l2 = field.l2_loss(y - y_target)
                correction = x - field.laplace(y - y_target)
                loss = field.l2_loss(prediction - correction)
                loss.sum.backward()
                optimizer.step()
            elif method == 'kFac':
                prediction = field.native_call(net, y_target)
                with optimizer.track_forward():
                    y = field.solve_linear(field.laplace, prediction, Solve('CG', 1e-5, 0, x0=x_gt * 0))
                    loss = y_l2 = field.l2_loss(y - y_target)
                    loss_sum = loss.sum
                with optimizer.track_backward():
                    loss_sum.backward()
                optimizer.step(loss=loss_sum)
            elif method == 'AdaHessian':
                prediction = field.native_call(net, y_target)
                y = field.solve_linear(field.laplace, prediction, Solve('CG', 1e-5, 0, x0=x_gt * 0))
                loss = y_l2 = field.l2_loss(y - y_target)
                loss.sum.backward(create_graph=True)
                optimizer.step()
            else:
                raise ValueError(method)
            viewer.log_scalars(x_l1=field.l1_loss(x_gt - prediction), y_l2=y_l2)
            if time.perf_counter() - start_time > 60 * 60 * 7:  # time limit
                break
        torch.save(net.state_dict(), viewer.scene.subpath(f'net_{method}.pth'))
print("All done.")

import os
import time

from inv_diffuse import generate_heat_example, inv_diffuse
from phi.torch.flow import *
from os.path import expanduser

from torch_optimizer import Adahessian
from hessianfree.optimizer import HessianFree


math.set_global_precision(64)
assert backend.default_backend().set_default_device('GPU')


def run_net():
    with math.precision(32):
        prediction = field.native_call(net, field.to_float(y_target)).vector[0]
        prediction += field.mean(x_gt) - field.mean(prediction)
    prediction = field.to_float(prediction)  # is this call necessary?
    if not field.isfinite(prediction):
        raise RuntimeError(net.state_dict())
    return prediction


for seed in range(1):
    math.seed(seed)
    net = u_net(1, 1)
    os.path.exists(expanduser(f"~/phi/heat_net2/{seed}")) or os.mkdir(expanduser(f"~/phi/heat_net2/{seed}"))
    torch.save(net.state_dict(), expanduser(f"~/phi/heat_net2/{seed}/init.pth"))

    for mi, method in enumerate(['SGD', 'AdaHessian', 'Adam + PG', 'Adam', 'HessianFreeGGN', ]):
        for batch_size in [128]:  # [32, 128, 4]:
            for learning_rate in [1e-3]:
                scene = Scene.create(f"~/phi/heat_net2/{seed}", name=f"{method}_lr_{learning_rate}_bs{batch_size}")
                viewer = view('x_gt, y_target, x, y, dx, amp, raw_kernel, kernel, sig_prob', scene=scene, port=8001 + mi, select='batch', gui='console')
                viewer.info(f"Training {method} with batch={batch_size}, lr={learning_rate}")
                torch.save(net.state_dict(), viewer.scene.subpath(f'net_init.pth'))
                print(scene)
                net.load_state_dict(torch.load(expanduser(f"~/phi/heat_net2/{seed}/init.pth")))
                if method == 'AdaHessian':
                    optimizer = Adahessian(net.parameters(), lr=0.001)
                elif method == 'HessianFreeH':
                    optimizer = HessianFree(net.parameters(), verbose=True, curvature_opt='hessian')
                elif method == 'HessianFreeGGN':
                    optimizer = HessianFree(net.parameters(), verbose=True, curvature_opt='ggn')
                else:
                    optimizer = optim.SGD(net.parameters(), lr=2e-6) if method == 'SGD' else optim.Adam(net.parameters(), lr=0.001)
                math.seed(0)
                start_time = time.perf_counter()
                for training_step in viewer.range():
                    optimizer.zero_grad()
                    # x_gt = train_x_b.batches[training_step % train_x_b.batches.size_or_1]
                    # y_target = train_y_b.batches[training_step % train_y_b.batches.size_or_1]
                    x_gt = generate_heat_example(spatial(x=64, y=64), batch(batch=batch_size))
                    y_target = diffuse.fourier(x_gt, 8., 1)
                    if method in ('Adam', 'SGD'):
                        prediction = run_net()
                        y = diffuse.fourier(prediction, 8., 1)
                        y_l2 = loss = field.l2_loss(y - y_target)
                        loss.sum.backward()
                        optimizer.step()
                    elif method in ['HessianFreeGGN', 'HessianFreeH']:
                        def forward():
                            global prediction, y_l2, y
                            prediction = run_net()
                            y = diffuse.fourier(prediction, 8., 1)
                            y_l2 = loss = field.l2_loss(y - y_target)
                            return loss.sum, prediction.values.native(prediction.values.shape)
                        optimizer.step(forward=forward)
                    elif method == 'Adam + PG':
                        prediction = run_net()
                        x = field.stop_gradient(prediction)
                        y = diffuse.fourier(x, 8., 1)
                        dx, _, amp, raw_kernel, kernel, sig_prob = inv_diffuse(y_target - y, 8., uncertainty=abs(y_target - y) * 1e-6)
                        correction = x + dx
                        loss = field.l2_loss(prediction - correction)
                        y_l2 = field.l2_loss(y - y_target)
                        loss.sum.backward()
                        optimizer.step()
                    elif method == 'AdaHessian':
                        prediction = run_net()
                        y = diffuse.fourier(prediction, 8., 1)
                        y_l2 = loss = field.l2_loss(y - y_target)
                        loss.sum.backward(create_graph=True)
                        optimizer.step()
                    else:
                        raise ValueError(method)
                    viewer.log_scalars(x_l1=field.l1_loss(x_gt - prediction), y_l1=field.l1_loss(y_target - y), y_l2=y_l2)
                    if time.perf_counter() - start_time > 6000:  # 300  # time limit
                        break
                torch.save(net.state_dict(), viewer.scene.subpath(f'net_{method}.pth'))
print("All done.")

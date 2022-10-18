import os
import time
from typing import Optional

from newton import newton_minimization
from phi.torch.flow import *

SEED = 0
OPTIMIZERS = {
    'Newton': lambda: optim.Adam(net.parameters(), lr=1e-3 * learning_rate_scale),
    'Adam': lambda: optim.Adam(net.parameters(), lr=1e-3 * learning_rate_scale),
    'SGD': lambda: optim.SGD(net.parameters(), lr=1e-2 * learning_rate_scale / ill_conditioning ** 2),
    'Adadelta': lambda: optim.Adadelta(net.parameters(), lr=3e-3 * learning_rate_scale),
    'Adagrad': lambda: optim.Adagrad(net.parameters(), lr=3e-3 * learning_rate_scale),
    'RMSprop': lambda: optim.RMSprop(net.parameters(), lr=3e-5 * learning_rate_scale),
}
DATASET_SIZE_IN_BATCHES: Optional[int] = None

math.seed(SEED)
BATCH = batch(batch=100)
VECTOR = channel(vector=2)
method = vis.control('Newton', tuple(OPTIMIZERS.keys()))
ill_conditioning = vis.control(10, (1, 100))
angle = vis.control(45, (0, 45))
global_scale = 10.  # This is the best value for Adam training.
learning_rate_scale = vis.control(1.)
dset_str = "" if DATASET_SIZE_IN_BATCHES is None else f" dset_{DATASET_SIZE_IN_BATCHES * BATCH.volume}"


def process(vec):
    vec = math.rotate_vector(vec, math.degrees(angle))
    vec *= (global_scale / ill_conditioning, global_scale * ill_conditioning)
    x, y = vec.vector.unstack()
    x, y = math.sin(x), y
    return math.stack([x, y], channel('vector'))


def eval_loss(x, target):
    return math.l2_loss(process(x) - target)


eval_hessian = math.hessian(eval_loss, 'x')


def closest_solution(vec, target):
    target_x, target_y = target.vector.unstack()
    vec = math.rotate_vector(vec, math.degrees(angle))
    vec *= (global_scale / ill_conditioning, global_scale * ill_conditioning)
    x, y = vec.vector.unstack()
    sol_x1 = math.arcsin(target_x)  # [-pi/2, pi/2]
    n1f = (x - sol_x1) / 2 / math.pi
    n1 = math.round(n1f)
    sol_x2 = math.pi - sol_x1
    n2f = (x - sol_x2) / 2 / math.pi
    n2 = math.round(n2f)
    sol1_better = abs(n1-n1f) < abs(n2-n2f)
    n = math.where(sol1_better, n1, n2)
    sol_x = math.where(sol1_better, sol_x1, sol_x2)
    sol_x += n * 2 * math.pi
    sol_y = target_y
    sol = math.stack([sol_x, sol_y], channel('vector'))
    sol /= (global_scale / ill_conditioning, global_scale * ill_conditioning)
    sol = math.rotate_vector(sol, math.degrees(-angle))
    return sol


TORCH.set_default_device('GPU')
math.seed(SEED)
net = dense_net(2, 2, [32, 64, 32])
optimizer = optim.Adam(net.parameters())  # will be initialized in reset()
os.path.isdir(f"~/phi/sin_lin_net/{SEED}") or os.makedirs(f"~/phi/sin_lin_net/{SEED}")
torch.save(net.state_dict(), f"~/phi/sin_lin_net/{SEED}/net_init.pth")


@vis.action  # make this function callable from the user interface
def save_model(step):
    path = viewer.scene.subpath(f"net_{step}.pth")
    torch.save(net.state_dict(), path)
    viewer.info(f"Model saved to {path}.")


@vis.action
def reset():
    math.seed(SEED)
    global optimizer
    optimizer = OPTIMIZERS[method]()
    net.load_state_dict(torch.load(f"~/phi/sin_lin_net/{SEED}/net_init.pth"))


math.seed(SEED)
if DATASET_SIZE_IN_BATCHES:
    TRAINING_SET = [math.random_uniform(BATCH, channel(vector=2)) * 2 - 1 for _ in range(DATASET_SIZE_IN_BATCHES)]


for mi, method in enumerate(OPTIMIZERS.keys()):
    viewer = view('loss_by_target, pred_by_target, newton_by_target', scene=Scene.create(f"~/phi/sin_lin_net/{SEED}", name=f"{method} ill_{ill_conditioning} angle_{angle}{dset_str}"), namespace=globals(), select='batch')
    save_model(0)
    reset()  # Ensure that the first run will be identical to every time reset() is called
    start_time = time.perf_counter()
    for step in viewer.range():
        # Load or generate training data
        y_target_val = math.random_uniform(BATCH, channel(vector=2)) * 2 - 1
        y_target = TRAINING_SET[step % DATASET_SIZE_IN_BATCHES] if DATASET_SIZE_IN_BATCHES else y_target_val
        # Initialize optimizer
        optimizer.zero_grad()
        # Prediction
        x_predicted = math.native_call(net, y_target)  # calls net with shape (BATCH_SIZE, channels, spatial...)
        # Simulation and Loss
        if method == 'Newton':
            y_l2, grad, hessian = eval_hessian(x_predicted, y_target)
            newton_min = newton_minimization(grad, hessian, max_loss_change=0.2)
            loss = math.l1_loss(x_predicted - math.stop_gradient(x_predicted + newton_min))
        else:
            y_l2 = loss = eval_loss(x_predicted, y_target)
        # Show curves in user interface
        x_predicted_val = math.native_call(net, y_target_val) if DATASET_SIZE_IN_BATCHES else x_predicted
        closest_val = closest_solution(x_predicted_val, y_target_val)
        y_l2_val = eval_loss(x_predicted_val, y_target_val) if DATASET_SIZE_IN_BATCHES else y_l2
        viewer.log_scalars(y_l2_train=y_l2, loss_train=loss, y_l2=y_l2_val, x_l1=math.l1_loss(x_predicted_val - closest_val), x_dist=math.vec_length(x_predicted_val - closest_val))
        # Compute gradients and update weights
        loss.mean.backward()
        optimizer.step()
        if (step + 1) % 1000 == 0:
            save_model(step + 1)
        if time.perf_counter() - start_time > 600:  # time limit
            break
    save_model(step + 1)

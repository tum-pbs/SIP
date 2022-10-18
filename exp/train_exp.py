from phi.torch.flow import *


TORCH.set_default_device('GPU')
math.seed(0)
BATCH = batch(batch=100)
net = dense_net(1, 1, [16, 64, 16], activation='Sigmoid')
optimizer = optim.SGD(net.parameters(), lr=1e-2)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
method = vis.control('Adjoint', ('Adjoint', 'Ground Truth', 'Newton', 'Newton Clipped', 'Sign'))


@vis.action
def save_model():
    path = viewer.scene.subpath(f"net_{viewer.steps}.pth" if viewer.scene else f"net_{viewer.steps}.pth")
    torch.save(net.state_dict(), path)
    viewer.info(f"Model saved to {path}.")
    # To load the network: net.load_state_dict(torch.load('net.pth'))


def exp_loss(x_prediction, y_label):
    y_prediction = math.exp(x_prediction)
    return math.l2_loss(y_prediction - y_label), y_prediction


exp_loss_hessian = math.hessian(exp_loss, 'x_prediction', get_output=True, get_gradient=True)
exp_loss_grad = math.gradient(exp_loss, get_output=True)


viewer = view('grid', play=True, scene=Scene.create('~/phi/exp_Sigmoid', name=method), namespace=globals(), select='batch')

for step in viewer.range(100001):
    x = math.random_uniform(batch(batch=100)) * -12
    y = math.exp(x)
    optimizer.zero_grad()
    x_prediction = math.native_call(net, y)  # calls net with shape (BATCH_SIZE, channels, spatial...)
    if method == 'Adjoint':
        y_l2, y_prediction = exp_loss(x_prediction, y)
        loss = y_l2
    elif method == 'Ground Truth':
        loss = math.l2_loss(x_prediction - x)
        y_l2, y_prediction = exp_loss(x_prediction, y)
    elif method == 'Newton':
        y_l2, y_prediction = exp_loss(x_prediction, y)
        # (y_l2, y_prediction), grad, hessian = exp_loss_hessian(x_prediction, y)
        grad, hessian = (y_prediction - y) * y_prediction, 2 * math.exp(2 * x_prediction) - y * y_prediction
        newton = abs(math.divide_no_nan(grad, hessian)) * math.sign(-grad)
        x_correction = math.stop_gradient(x_prediction + newton)
        loss = math.l2_loss(x_prediction - x_correction)
    elif method == 'Newton Clipped':
        y_l2, y_prediction = exp_loss(x_prediction, y)
        # (y_l2, y_prediction), grad, hessian = exp_loss_hessian(x_prediction, y)
        grad, hessian = (y_prediction - y) * y_prediction, 2 * math.exp(2 * x_prediction) - y * y_prediction
        newton = math.minimum(abs(math.divide_no_nan(grad, hessian)), .1) * math.sign(-grad)
        x_correction = math.stop_gradient(x_prediction + newton)
        loss = math.l2_loss(x_prediction - x_correction)
    elif method == 'Newton Damped':
        y_l2, y_prediction = exp_loss(x_prediction, y)
        # (y_l2, y_prediction), grad, hessian = exp_loss_hessian(x_prediction, y)
        grad, hessian = (y_prediction - y) * y_prediction, 2 * math.exp(2 * x_prediction) - y * y_prediction
        DAMPING = 1
        newton = abs(math.divide_no_nan(grad, hessian + .01)) * math.sign(-grad)
        x_correction = math.stop_gradient(x_prediction + newton)
        loss = math.l2_loss(x_prediction - x_correction)
    elif method == 'Sign':
        y_l2, y_prediction = exp_loss(x_prediction, y)
        x_correction = math.stop_gradient(x_prediction + math.sign(x - x_prediction))
        loss = math.l2_loss(x_prediction - x_correction)
    # elif method == 'Inverse Gradient':
    #     (y_l2, y_prediction), grad = exp_loss_grad(x_prediction, y)
    #     delta =
    #     x_correction = math.stop_gradient(x_prediction - math.minimum(math.divide_no_nan(1, grad), .1))
    #     loss = math.l2_loss(x_prediction - x_correction)
    else:
        raise ValueError(method)
    viewer.log_scalars(y_l2=y_l2, y_l1=math.l1_loss(y_prediction - y), x_l2=math.l2_loss(x_prediction - x), x_l1=math.l1_loss(x_prediction - x))

    grid = CenteredGrid(math.map_s2b(lambda y: math.native_call(net, y)), 0, x=100, bounds=Box['x', -12:0])

    # Compute gradients and update weights
    loss.sum.backward()
    optimizer.step()
    if step % 1000 == 0:
        save_model()

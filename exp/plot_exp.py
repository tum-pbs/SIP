from os.path import basename, join

from phi.torch.flow import *


PATHS = [
    "~/phi/exp_Sigmoid/Adjoint_000003",
    # "~/phi/exp_Sigmoid/Ground Truth",
    "~/phi/exp_Sigmoid/Adam",
    # "~/phi/exp_Sigmoid/Newton",
    # "~/phi/exp_Sigmoid/Newton Damped 0.01",
    "~/phi/exp_Sigmoid/Sign",
    # "~/phi/exp_Sigmoid/SGD 1e-2",
    # "~/phi/exp_Sigmoid/SGD 1e-4",
]

# for x_axis in ['time', 'steps']:
#     vis.plot_scalars(PATHS, names=['x_l1', 'y_l1'],
#     log_scale='y', x=x_axis, reduce='scenes', smooth=64)
#     # vis.savefig()
#     vis.show()


import pylab

pylab.rc('font', family='Arial', weight='normal', size=8)
# pylab.style.use('dark_background')
cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']


def plot_error_by_x(path: str, label: str):
    net = dense_net(1, 1, [16, 64, 16], activation='Sigmoid')
    net.load_state_dict(torch.load(path))
    x = math.linspace(-12, 0, 1000, batch('batch'))
    y = math.exp(x)
    x_prediction = math.native_call(net, y)
    x_error = math.l1_loss(x - x_prediction)
    pylab.plot(x.numpy(), x_error.numpy(), label=label)


for path in PATHS:
    plot_error_by_x(join(path, "net_10000.pth"), basename(path))

# pylab.yscale('log')
pylab.legend()
pylab.show()


def plot_result_by_x(path: str, label: str, color: int):
    net = dense_net(1, 1, [16, 64, 16], activation='Sigmoid')
    net.load_state_dict(torch.load(path))
    x = math.linspace(-12, 0, 1000, batch('batch'))
    y = math.exp(x)
    x_prediction = math.native_call(net, y)
    pylab.plot(x.numpy(), x_prediction.numpy(), label=label, color=cycle[color])


pylab.figure(figsize=(2.5, 2.2))
x = math.linspace(-11, 0, 1000, batch('batch'))
pylab.plot(x.numpy(), x.numpy(), '--', color='grey')
plot_result_by_x("~/phi/exp_Sigmoid/SGD 1e-2/net_10000.pth", "SGD", 3)
plot_result_by_x("~/phi/exp_Sigmoid/Adam/net_10000.pth", "Adam", 1)
plot_result_by_x("~/phi/exp_Sigmoid/Sign/net_10000.pth", "Rescaled", 0)

# pylab.yscale('log')
pylab.legend()
pylab.xlabel("$x^*$")
pylab.ylabel("Predicted $x$")
pylab.tight_layout()
pylab.savefig('plots/exp/exp_pred_vs_sol.pdf', transparent=True)
pylab.show()


from matplotlib.ticker import ScalarFormatter

from phi.flow import *
import pylab


# pylab.rc('font', family='Arial', weight='normal', size=8)
# fig = vis.plot_scalars([
#     "~/phi/heat_net2/0/Adam + PG_000000",
#     "~/phi/heat_net2/0/FNO_lr_0.0001_bs128_000003",
#     "~/phi/heat_net2/0/FNO_lr_3e-05_bs128_000001",
#     "~/phi/heat_net2/0/FNO_lr_0.0001_bs128_000004",
#     "~/phi/heat_net2/0/FNO_lr_1e-05_bs128_000001",
# ], names='x_l1', reduce='scenes',
#     labels=wrap(["SIP (Ours)", "FNO 1e-4", "FNO 3e-5", "FNO 1e-4", "FNO 1e-5"], batch('scenes')),
#     # colors=wrap([3, 1, 2, 0], batch('scenes')),
#     size=(3.2, 1.7), log_scale='xy', smooth=64, smooth_alpha=0.1, smooth_linewidth=1,
#     titles=False, ylabel="Inferred $x$ MAE", x='steps')
# # pylab.xlim((-150, 4000))
# vis.show()


pylab.rc('font', family='Arial', weight='normal', size=8)
fig = vis.plot_scalars([
    "~/phi/heat_net2/0/FNO_lr_0.0001_bs128_000004",
    "~/phi/heat_net2/0/SGD_000000",
    "~/phi/heat_net2/0/Adam_000000",
    "~/phi/heat_net2/0/AdaHessian_000005",
    "~/phi/heat_net2/0/Adam + PG_000000",
    "~/phi/heat_net2/0/HessianFreeGGN_lr_0.001_bs128_000000",
], names='x_l1', reduce='scenes',
    labels=wrap(["FNO", "SGD", "Adam", "AdaHessian", "SIP (Ours)", "H-free"], batch('scenes')),
    colors=wrap([4, 3, 1, 2, 0, 5], batch('scenes')),
    size=(3.2, 1.7), log_scale='y', smooth=wrap([64, 64, 64, 64, 64, 8], batch('scenes')), smooth_alpha=0.1, smooth_linewidth=1,
    titles=False, ylabel="Inferred $x$ MAE", x='time',  grid=False)
fig.axes[0].set_yticks([100, 500])
fig.axes[0].get_yaxis().set_major_formatter(ScalarFormatter())
pylab.tight_layout()
pylab.xlim((-150, 4000))
vis.savefig('plots/heat/learning-curves.pdf')
vis.show()


# vis.plot_scalars([
#     "~/phi/heat_net2/0/AdaHessian_000000",  # 0.01
#     "~/phi/heat_net2/0/AdaHessian_000001",  # 0.003
#     "~/phi/heat_net2/0/AdaHessian_000003",  # 0.001 best so far
#     "~/phi/heat_net2/0/AdaHessian_000004",  # 0.0003
# ],
#     labels=wrap(["$\\eta=0.01$", "$\\eta=0.003$", "$\\eta=0.001$", "$\\eta=0.0003$"], batch('scenes')),
#     names='x_l1', reduce='scenes', log_scale='y', smooth=64, smooth_alpha=0.1, x='time', titles="AdaHessian")
# vis.savefig('plots/heat/AdaHessian_learning-curves.pdf')
# vis.show()


# from phi.vis._plot_util import smooth_uniform_curve
#
# pylab.style.use('dark_background')
# vis.plot_scalars([
#     "~/phi/heat_net/sim_000000"
# ], names=['x_loss_Adam', 'x_loss_Adam + L-BFGS-B', 'x_loss_Adam + PG', ],
#     labels=wrap(['Adam', 'Adam + PG (L-BFGS-B)', 'Adam + PG (Inv.phys.)', ], batch('names')),
#     colors=wrap([1, 2, 0], batch('names')),
#     size=(8, 6), log_scale='xy', smooth=64, smooth_alpha=0.1, smooth_linewidth=1,
#     titles=False, ylabel="Inferred $x$ MAE", xlabel="Iteration", grid='xy', xlim=(30, 1e4))
# vis.show()
#
# pg_i, pg = smooth_uniform_curve(*np.loadtxt("~/phi/heat_net/sim_000000/log_x_loss_Adam + PG.txt").T, n=16)
# gd_i, gd = smooth_uniform_curve(*np.loadtxt("~/phi/heat_net/sim_000000/log_x_loss_Adam.txt").T, n=16)
# bfgs_i, bfgs = smooth_uniform_curve(*np.loadtxt("~/phi/heat_net/sim_000000/log_x_loss_Adam + L-BFGS-B.txt").T, n=16)
#
# print(np.mean(bfgs / gd))
# print(np.mean(gd[9984] / pg[9984]))
# print(pg_i[np.argwhere(pg < gd / 3)[0, 0]])
# print(pg_i[np.argwhere(bfgs < np.min(gd))[0, 0]])

# Supplemental
# vis.plot_scalars([
#     "~/phi/heat_net/sim_000001",
#     "~/phi/heat_net/sim_000004",
# ], names=['x_loss_Adam + PG', 'x_loss_Adam', 'x_loss_Adam + L-BFGS-B'],
#     labels=math.wrap(['Adam + PG (Inv.phys.)', 'Adam', 'Adam + PG (L-BFGS-B)'], 'names'),
#     size=(7.1, 3), log_scale='y', smooth=64, titles=False, grid=False,
#     smooth_alpha=0.1, smooth_linewidth=1)
# vis.savefig('plots/heat/SI_heat-learning-curves.pdf')
# vis.show()

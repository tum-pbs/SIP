from phi.flow import *
import pylab


pylab.rc('font', family='Arial', weight='normal', size=8)
vis.plot_scalars([
                  "~/phi/poisson_net2/0/HessianFreeGGN_000002",
                  ], 'x_l1',
                 reduce='scenes',
                 log_scale='y', size=(3.2, 1.7), legend='lower left', grid=None, transform=lambda c: c[:, :10_000],
                 titles=False, x='time', ylabel="Inferred $x$ MAE")
# vis.savefig('plots/poisson/training-curves-7cm.pdf', transparent=True)
vis.show()


# pylab.rc('font', family='Arial', weight='normal', size=8)
# vis.plot_scalars([
#                   "~/phi/poisson_net2/0/SGD_000000",
#                   "~/phi/poisson_net2/0/Adam_000000",
#                   "~/phi/poisson_net2/0/AdaHessian_000002",
#                   "~/phi/poisson_net2_FNO/0/FNO_lr0.003_000001",
#                   "~/phi/poisson_net2/0/SIP_000000",
#                   "~/phi/poisson_net2/0/HessianFreeGGN_000002",
#                   ], 'x_l1',
#                  labels=wrap(["SGD", "Adam", "AdaHessian", "FNO", "SIP (Ours)", "H-free"], batch('scenes')), reduce='scenes',
#                  colors=wrap([3, 1, 2, 4, 0, 5], batch('scenes')),
#                  log_scale='y', size=(3.2, 1.7), legend='lower left', grid=None,
#                  smooth=wrap([64, 64, 64, 64, 64, 1], batch('scenes')),
#                  smooth_alpha=0.1, smooth_linewidth=1, transform=lambda c: c[:, :10_000],
#                  titles=False, x='time', ylabel="Inferred $x$ MAE", xlim=(-150, 3900))
# import pylab
# pylab.savefig('plots/poisson/training-curves-7cm.pdf', transparent=True)
# vis.show()
# pylab.show()

# pylab.style.use('dark_background')
# vis.plot_scalars(["~/phi/poisson_net_sgd/sim_000001", "~/phi/poisson_net/sim_000000", "~/phi/poisson_net/sim_000000"],
#                  wrap(['SGD', 'Adam', 'Adam + PG'], batch('scenes')),
#                  labels=wrap(["SGD", "Adam", "Adam + PG (Inv.phys.)"], batch('scenes')), reduce='scenes',
#                  colors=wrap([3, 1, 0], batch('scenes')),
#                  smooth=64, log_scale='xy', size=(8, 6), legend='lower left', grid='xy', smooth_alpha=0.1, smooth_linewidth=1,
#                  xlim=[10, 12000],
#                  titles=False, xlabel="Iteration", ylabel="Inferred $x$ MAE")
# vis.show()



# from phi.vis._plot_util import smooth_uniform_curve
#
# pg_i, pg = smooth_uniform_curve(*np.loadtxt("~/phi/poisson_net/sim_000000/log_Adam + PG.txt").T, n=16)
# gd_i, gd = smooth_uniform_curve(*np.loadtxt("~/phi/poisson_net/sim_000000/log_Adam.txt").T, n=16)
#
# print(gd[100] / pg[100])
# print(gd[1000] / pg[1000])
# print(gd[10000] / pg[10000])
# print(pg_i[np.argwhere(pg < gd / 3)[0, 0]])


# Supplemental
# vis.plot_scalars([
#     "~/phi/poisson_net/sim_000000",
#     "~/phi/poisson_net/sim_000001",
#     "X:/phi/poisson_net/sim_000002_4",
#     # "X:/phi/poisson_net/sim_000005",
# ], ['Adam + PG', 'Adam'], down='', smooth=1, log_scale='y', size=(7.1, 3), legend='lower left', titles=False)
# vis.savefig("plots/SI_poisson_curves.pdf")
# vis.show()

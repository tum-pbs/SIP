import pylab
from matplotlib.ticker import ScalarFormatter

from phi.flow import *


# pylab.rc('font', family='Arial', weight='normal', size=8)
# fig = vis.plot_scalars([
#     "~/phi/fluid_v0_net_swirl/Adam",
#     "~/phi/fluid_v0_net_swirl/Adam + PG",
# ], ['gt_v0_l1'],
#     labels=wrap(["Adam", "SIP (Ours)"], batch('scenes')),
#     colors=wrap([1, 0], batch('scenes')),
#     reduce='scenes', smooth=64, transform=lambda x: (x[0, :20_000], x[1, :20_000] / 64 / 64),
#     size=(3.2, 1.7), log_scale='y', smooth_alpha=0.1, smooth_linewidth=1, grid='', legend='center right',
#     titles=False, ylabel="Inferred $x$ MAE", x='time')
# fig.axes[0].set_yticks([2, 5, 10])
# fig.axes[0].get_yaxis().set_major_formatter(ScalarFormatter())
# pylab.tight_layout()
# pylab.savefig("plots/fluid/learning-curves-7cm.pdf", transparent=True)
# pylab.show()



from phi.vis._plot_util import smooth_uniform_curve

pg_i, pg = smooth_uniform_curve(np.loadtxt("~/phi/fluid_v0_net_swirl/Adam + PG/log_gt_v0_l1.txt"), n=64).T
gd_i, gd = smooth_uniform_curve(np.loadtxt("~/phi/fluid_v0_net_swirl/Adam/log_gt_v0_l1.txt"), n=64).T

print(gd[350] / pg[350])
print(pg.shape)
print(gd.shape)

# Average


print(pg_i[np.argwhere(pg < np.min(gd[:131]))[0, 0]])
print(pg_i[np.argwhere(pg < np.min(gd[:1000]))[0, 0]])
print(pg_i[np.argwhere(pg < np.min(gd[:10_000]))[0, 0]])

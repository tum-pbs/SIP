import pylab
from fluid_base import *


net = u_net(2, 2, levels=5, filters=16)
print(f"Parameter count: {parameter_count(net)}")

math.seed(0)
m0, mt, gt, gtv = generate_example()
train_marker_keys = field.stack([m0, mt], dim=channel('keyframe'))

net.load_state_dict(torch.load('~/phi/fluid_v0_net_swirl/Adam/net_16000.pth'))
prediction_gd = CenteredGrid(math.native_call(net, train_marker_keys.values), **DOMAIN)
loss_gd, gd, gdv = eval_physics_loss(prediction_gd, train_marker_keys)
print(loss_gd)

net.load_state_dict(torch.load('~/phi/fluid_v0_net_swirl/Adam + PG/net_16000.pth'))
prediction_pg = CenteredGrid(math.native_call(net, train_marker_keys.values), **DOMAIN)
loss_pg, pg, pgv = eval_physics_loss(prediction_pg, train_marker_keys)
print(loss_pg)

cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    reg_index = np.linspace(start, stop, 257)  # regular index to compute the colors
    shift_index = np.hstack([  # shifted index to match the data
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


cmap = shiftedColorMap(matplotlib.cm.RdBu, midpoint=0)


def plot_vel(axis, vel: CenteredGrid):
    vorticity = field.curl(vel)
    axis.imshow(vorticity.values.numpy('y,x'), origin='lower', cmap=cmap)

    # vel = field.downsample2x(field.downsample2x(vel)) * 2
    # x, y = [t.numpy('x,y') for t in vel.points.vector.unstack()]
    # u, v = [t.numpy('x,y') for t in vel.values.vector.unstack()]
    # axis.quiver(x - u / 2, y - v / 2, u, v, color='black')


# o = 5
# fig, axes = pylab.subplots(nrows=3 * 3, ncols=6, figsize=(6, 9))  # (7.1, 8)
# for b in range(3):
#     axes[3 * b, 0].set_ylabel(f"GT")
#     axes[3 * b + 1, 0].set_ylabel(f"A+PG")
#     axes[3 * b + 2, 0].set_ylabel(f"Adam")
#     plot_vel(axes[3 * b, 0], gtv.batch[b + o].frames[0])
#     plot_vel(axes[3 * b + 1, 0], pgv.batch[b + o].frames[0].at_centers())
#     plot_vel(axes[3 * b + 2, 0], gdv.batch[b + o].frames[0].at_centers())
#     for f in range(5):
#         axes[3 * b, f + 1].imshow(gt.batch[b+o].frames[2 * f].values.numpy('y,x'), origin='lower')
#         axes[3 * b + 1, f + 1].imshow(pg.batch[b+o].frames[2 * f].values.numpy('y,x'), origin='lower')
#         axes[3 * b + 2, f + 1].imshow(gd.batch[b+o].frames[2 * f].values.numpy('y,x'), origin='lower')
# for y in range(axes.shape[0]):
#     for x in range(axes.shape[1]):
#         axes[y, x].set_xticks([])
#         axes[y, x].set_yticks([])
#         for spine in axes[y, x].spines.values():
#             if y % 3 in (1, 2):
#                 spine.set_color(cycle[y % 3 - 1])
#                 spine.set_linewidth(2)
# pylab.tight_layout()
# pylab.subplots_adjust(wspace=0.05, hspace=0.05)
# pylab.savefig("plots/SI_fluid.pdf", transparent=True)
# pylab.show()

b = 3
fig, axes = pylab.subplots(nrows=1, ncols=7, figsize=(3.5, 1.05))
axes[0].imshow(gt.batch[b].frames[0].values.numpy('y,x'), origin='lower')
axes[0].set_title("$m_0$", fontsize=8)
axes[1].imshow(gt.batch[b].frames[-1].values.numpy('y,x'), origin='lower')
axes[1].set_title("$y^*$", fontsize=8)
axes[2].imshow(gd.batch[b].frames[-1].values.numpy('y,x'), origin='lower')
axes[2].set_title(r"$y_\mathrm{Adam}$", fontsize=8)
axes[3].imshow(pg.batch[b].frames[-1].values.numpy('y,x'), origin='lower')
axes[3].set_title(r"$y_\mathrm{SIP}$", fontsize=8)
plot_vel(axes[4], gdv.batch[b].frames[0].at_centers())
axes[4].set_title(r"$x_\mathrm{Adam}$", fontsize=8)
plot_vel(axes[5], pgv.batch[b].frames[0].at_centers())
axes[5].set_title(r"$x_\mathrm{SIP}$", fontsize=8)
plot_vel(axes[6], gtv.batch[b].frames[0])
axes[6].set_title(r"$x^*$", fontsize=8)
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        if i in (2, 4):
            spine.set_color(cycle[1])
            spine.set_linewidth(2)
        if i in (3, 5):
            spine.set_color(cycle[0])
            spine.set_linewidth(2)
pylab.tight_layout()
pylab.subplots_adjust(wspace=0.05, hspace=0.05)
pylab.savefig("plots/fluid/v0-net-example-vorticity.pdf", transparent=True)
pylab.show()

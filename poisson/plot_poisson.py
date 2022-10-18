from phi.torch.flow import *
import pylab


net = u_net(1, 1)

math.seed(999999)
x_gt = CenteredGrid(Noise(batch(batch=128)), 0, x=64, y=64)
y_target = field.solve_linear(field.laplace, x_gt, Solve('CG', 1e-5, 0, x0=x_gt * 0))

net.load_state_dict(torch.load("~/phi/poisson_net/sim_000000/net_GD.pth"))
x_gd = field.native_call(net, y_target).vector[0]
# y_gd = field.solve_linear(field.laplace, x_gd, Solve('CG', 1e-5, 0, x0=x_gt * 0))

net.load_state_dict(torch.load("~/phi/poisson_net/sim_000000/net_PG.pth"))
x_pg = field.native_call(net, y_target).vector[0]
# y_pg = field.solve_linear(field.laplace, x_pg, Solve('CG', 1e-5, 0, x0=x_gt * 0))

net.load_state_dict(torch.load("~/phi/poisson_net_sgd/sim_000001/net_SGD.pth"))
x_sgd = field.native_call(net, y_target).vector[0]
# y_sgd = field.solve_linear(field.laplace, x_pg, Solve('CG', 1e-5, 0, x0=x_gt * 0))

cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']

# fig, axes = pylab.subplots(nrows=3, ncols=6, figsize=(7.1, 5))  # axes=(batch, 6)
# for b in range(3):
#     axes[b, 0].imshow(y_target.batch[b].values.numpy('y,x'), origin='lower')
#     axes[b, 0].set_title(f"$y^*$")
#     axes[b, 2].imshow(y_pg.batch[b].values.numpy('y,x'), origin='lower')
#     axes[b, 2].set_title(f"Adam+PG $y$")
#     axes[b, 1].imshow(y_gd.batch[b].values.numpy('y,x'), origin='lower')
#     axes[b, 1].set_title(f"Adam $y$")
#     axes[b, 4].imshow(x_pg.batch[b].values.numpy('y,x'), origin='lower')
#     axes[b, 4].set_title(f"Adam+PG $x$")
#     axes[b, 3].imshow(x_gd.batch[b].values.numpy('y,x'), origin='lower')
#     axes[b, 3].set_title(f"Adam $x$")
#     axes[b, 5].imshow(x_gt.batch[b].values.numpy('y,x'), origin='lower')
#     axes[b, 5].set_title(f"$x^*$")
#     for r in range(6):
#         axes[b, r].set_xticks([])
#         axes[b, r].set_yticks([])
#         for spine in axes[b, r].spines.values():
#             if r in (1, 3):
#                 spine.set_color(cycle[1])
#                 spine.set_linewidth(3)
#             if r in (2, 4):
#                 spine.set_color(cycle[0])
#                 spine.set_linewidth(3)
# pylab.tight_layout()
# pylab.subplots_adjust(wspace=0.06, hspace=0.06)
# for b in range(3):
#     for r in range(6):
#         p = axes[b, r].get_position()
#         axes[b, r].set_position([p.x0 + (0.01 if r >= 3 else -0.01), p.y0, p.width, p.height])
# pylab.savefig("plots/SI_poisson_xy.pdf", transparent=True)
# pylab.show()


b = 10
fig, axes = pylab.subplots(nrows=1, ncols=5, figsize=(3.4, 1.4))
axes[0].imshow(y_target.batch[b].values.numpy('y,x'), origin='lower')
axes[0].set_title(f"$y^*$", fontsize=8)
axes[1].imshow(x_sgd.batch[b].values.numpy('y,x'), origin='lower')
axes[1].set_title(r"$x_\mathrm{SGD}$", fontsize=8)
axes[2].imshow(x_gd.batch[b].values.numpy('y,x'), origin='lower')
axes[2].set_title(r"$x_\mathrm{Adam}$", fontsize=8)
axes[3].imshow(x_pg.batch[b].values.numpy('y,x'), origin='lower')
axes[3].set_title(r"$x_\mathrm{SIP}$", fontsize=8)
axes[4].imshow(x_gt.batch[b].values.numpy('y,x'), origin='lower')
axes[4].set_title(f"$x^*$", fontsize=8)
for r in range(len(axes)):
    axes[r].set_xticks([])
    axes[r].set_yticks([])
    for spine in axes[r].spines.values():
        if r == 1:
            spine.set_color(cycle[3])
            spine.set_linewidth(2)
        if r == 2:
            spine.set_color(cycle[1])
            spine.set_linewidth(2)
        if r == 3:
            spine.set_color(cycle[0])
            spine.set_linewidth(2)
pylab.tight_layout()
pylab.subplots_adjust(wspace=0.06, hspace=0.06)
for r in range(len(axes)):
    p = axes[r].get_position()
    axes[r].set_position([p.x0 + (0.005 if r >= 1 else -0.005), p.y0, p.width, p.height])
pylab.savefig("plots/poisson/poisson-example.pdf", transparent=True)
pylab.show()

import time

from inv_diffuse import generate_heat_example, inv_diffuse
from phi.torch.flow import *
import pylab

# TORCH.set_default_device('GPU')
math.set_global_precision(64)
viewer = view('x_pg, dx, y, dy, x_gt, x_gd, x_bfgs, y_target', select='batch')

math.seed(0)
x_gt = generate_heat_example(batch(batch=128), spatial(x=64, y=64))
y_target = diffuse.fourier(x_gt, 8., 1)


def loss_function(x):
    return field.l2_loss(diffuse.fourier(x, 8., 1) - y_target)


net = u_net(1, 1)
net.load_state_dict(torch.load("~/phi/heat_net2/0/Adam_000000/net_Adam.pth"))

# # Adam
# viewer.info('Adam')
# t = time.perf_counter()
# with math.precision(32):
#     x_adam = field.native_call(net, field.to_float(y_target)).vector[0]
#     x_adam += field.mean(x_gt) - field.mean(x_adam)
# x_adam = field.to_float(x_adam)
# x_adam = x_adam.with_values(math.maximum(0, x_adam.values))
# y_adam = diffuse.fourier(x_adam, 8., 1)
# viewer.info(f"MSE: {field.l1_loss(x_gt - x_adam)}")
# print(f"Time elapsed: {time.perf_counter() - t}")
# numpy.save('heat/data/NN_Adam_x_l1', field.l1_loss(x_gt - x_adam).numpy('batch'))
#
# # Adam + PG
# viewer.info('SIP')
# t = time.perf_counter()
# net.load_state_dict(torch.load("~/phi/heat_net2/0/Adam + PG_000000/net_Adam + PG.pth"))
# with math.precision(32):
#     x_sip = field.native_call(net, field.to_float(y_target)).vector[0]
#     x_sip += field.mean(x_gt) - field.mean(x_sip)
# x_sip = field.to_float(x_sip)
# x_sip = x_sip.with_values(math.maximum(0, x_sip.values))
# y_adam_pg = diffuse.fourier(x_sip, 8., 1)
# viewer.info(f"MSE: {field.l1_loss(x_gt - x_sip)}")
# print(f"Time elapsed: {time.perf_counter() - t}")
# numpy.save('heat/data/NN_SIP_x_l1', field.l1_loss(x_gt - x_sip).numpy('batch'))
#
# # Adam + BFGS
# viewer.info('SGD')
# t = time.perf_counter()
# net.load_state_dict(torch.load("~/phi/heat_net2/0/SGD_000000/net_SGD.pth"))
# with math.precision(32):
#     x_sgd = field.native_call(net, field.to_float(y_target)).vector[0]
#     x_sgd += field.mean(x_gt) - field.mean(x_sgd)
# x_sgd = field.to_float(x_sgd)
# x_sgd = x_sgd.with_values(math.maximum(0, x_sgd.values))
# y_sgd = diffuse.fourier(x_sgd, 8., 1)
# viewer.info(f"MSE: {field.l1_loss(x_gt - x_sgd)}")
# print(f"Time elapsed: {time.perf_counter() - t}")
# numpy.save('heat/data/NN_SGD_x_l1', field.l1_loss(x_gt - x_sgd).numpy('batch'))

# GD
viewer.info('GD')
t = time.perf_counter()
dl_dx = field.functional_gradient(loss_function, get_output=True)
x_gd = 0 * x_gt
gd_losses = []
for _ in viewer.range(1000):
    loss, (grad,) = dl_dx(x_gd)
    x_gd -= 1. * grad
    viewer.log_scalars(gd_loss=loss.mean)
    gd_losses.append(field.l1_loss(x_gt - x_gd).numpy('batch'))
y_gd = diffuse.fourier(x_gd, 8., 1)
viewer.info(f"GD Time: {time.perf_counter() - t}")
numpy.save('heat/data/GD_x_l1', np.stack(gd_losses, -1))
#
# # PG
# viewer.info('PG')
# x_pg = 0 * x_gt
# for _ in viewer.range(100, warmup=1):
#     y = diffuse.fourier(x_pg, 8., 1)
#     dy = y_target - y
#     dx, _, amp, raw_kernel, kernel, sig_prob = inv_diffuse(dy, 8., uncertainty=abs(dy) * 1e-6 + 1e-8)
#     x_pg += dx
#     viewer.log_scalars(pg_dist=field.l1_loss(x_gt - x_pg).mean.cpu())
# y_pg = diffuse.fourier(x_pg, 8., 1)
#
# BFGS
x_l1_adam = 125.001
x_l1_sip = 50.408
x_l1_sgd = 118.939
t = time.perf_counter()
# viewer.info('BFGS')
# with math.SolveTape(record_trajectories=True) as solves:
#     x_bfgs = field.minimize(loss_function, Solve('L-BFGS-B', 0, 1e-14, 1000, x0=0 * x_gt, suppress=[NotConverged]))
# assert solves[0].x.values.default_backend == math.NUMPY
# # y_bfgs = diffuse.fourier(x_bfgs, 8., 1)
# bfgs_losses = field.l1_loss(solves[0].x - field.convert(x_gt, math.NUMPY), 'x,y')  # Shape (batch, trajectory)
# # numpy.save('heat/data/BFGS_x_l1', bfgs_losses.numpy('batch,trajectory'))
# viewer.info(f"BFGS Time: {time.perf_counter() - t}")

# print(math.mean(bfgs_losses, 'batch').trajectory[::10])
# viewer.info(math.sum(math.mean(field.l1_loss(bfgs_losses), 'batch') > x_l1_sip, 'trajectory'))

viewer.info('Done.')

cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']

# Large plots for Appendix
# o = 1
# fig, axes = pylab.subplots(nrows=2 * 2, ncols=7, figsize=(7.1, 4.6))
# for b in range(2):
#     axes[2 * b, 0].set_ylabel(f"$y$", fontsize=10)
#     axes[2 * b, 0].set_title(f"$*$", fontsize=10)
#     axes[2 * b, 0].imshow(y_target.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b, 1].set_title(r"GD", fontsize=10)
#     axes[2 * b, 1].imshow(y_gd.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b, 2].set_title(r"BFGS", fontsize=10)
#     axes[2 * b, 2].imshow(y_bfgs.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b, 3].set_title(r"Inv.phys.", fontsize=10)
#     axes[2 * b, 3].imshow(y_pg.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b, 4].set_title(r"Adam", fontsize=10)
#     axes[2 * b, 4].imshow(y_adam.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b, 5].set_title(r"A.+BFGS", fontsize=10)
#     axes[2 * b, 5].imshow(y_adam_bfgs.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b, 6].set_title(r"A.+PG", fontsize=10)
#     axes[2 * b, 6].imshow(y_adam_pg.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#
#     axes[2 * b + 1, 0].set_ylabel(f"$x$", fontsize=10)
#     axes[2 * b + 1, 0].imshow(x_gt.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b + 1, 1].imshow(x_gd.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b + 1, 2].imshow(x_bfgs.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b + 1, 3].imshow(x_pg.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b + 1, 4].imshow(x_adam.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b + 1, 5].imshow(x_adam_bfgs.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2 * b + 1, 6].imshow(x_adam_pg.batch[b + o].values.numpy('y,x'), origin='lower', cmap='magma')
# for b in range(4):
#     for r in range(7):
#         axes[b, r].set_xticks([])
#         axes[b, r].set_yticks([])
#         for spine in axes[b, r].spines.values():
#             if r in (1, 4):
#                 spine.set_color(cycle[1])
#                 spine.set_linewidth(2)
#             if r in (3, 6):
#                 spine.set_color(cycle[0])
#                 spine.set_linewidth(2)
#             if r in (2, 5):
#                 spine.set_color(cycle[2])
#                 spine.set_linewidth(2)
# pylab.tight_layout()
# pylab.subplots_adjust(wspace=0.06, hspace=0.06)
# for row in range(4):
#     for i in range(7):
#         p = axes[row, i].get_position()
#         axes[row, i].set_position([p.x0 + (0.01 if i > 3 else (-0.01 if i == 0 else 0)),
#                                    p.y0 + (-0.01 if row % 2 == 0 else 0.01),
#                                    p.width, p.height])
# pylab.savefig("plots/SI-heat-examples.pdf", transparent=True)
# pylab.show()


# Pictures
# for b in range(10):
#     fig, axes = pylab.subplots(nrows=1, ncols=5, figsize=(3.4, 1.4))
#     axes[0].set_title(f"$y^*$", fontsize=8)
#     axes[0].imshow(y_target.batch[b].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[1].set_title(r"$x_\mathrm{SGD}$", fontsize=8)
#     axes[1].imshow(x_sgd.batch[b].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[2].set_title(r"$x_\mathrm{Adam}$", fontsize=8)
#     axes[2].imshow(x_adam.batch[b].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[3].set_title(r"$x_\mathrm{A+PG}$", fontsize=8)
#     axes[3].imshow(x_sip.batch[b].values.numpy('y,x'), origin='lower', cmap='magma')
#     axes[4].set_title("$x^*$", fontsize=8)
#     axes[4].imshow(x_gt.batch[b].values.numpy('y,x'), origin='lower', cmap='magma')
#     for r in range(5):
#         axes[r].set_xticks([])
#         axes[r].set_yticks([])
#         for spine in axes[r].spines.values():
#             if r == 1:
#                 spine.set_color(cycle[3])
#                 spine.set_linewidth(2)
#             if r == 2:
#                 spine.set_color(cycle[1])
#                 spine.set_linewidth(2)
#             if r == 3:
#                 spine.set_color(cycle[0])
#                 spine.set_linewidth(2)
#     pylab.tight_layout()
#     pylab.subplots_adjust(wspace=0.06, hspace=0.06)
#     for r in range(5):
#         p = axes[r].get_position()
#         axes[r].set_position([p.x0 + (0.01 if r > 3 else (-0.01 if r == 0 else 0)), p.y0, p.width, p.height])
#     pylab.savefig(f"plots/heat/heat-example-{b}.pdf", transparent=True)
#     pylab.show()
#     pylab.close()
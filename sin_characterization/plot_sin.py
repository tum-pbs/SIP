from phi.flow import *
import pylab

pylab.rc('font', family='Arial', weight='normal', size=8)
cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']


SEED = 0
METHODS = ['Newton', 'Adam', 'SGD', 'Adadelta', 'Adagrad', 'RMSprop']
COLORS = [cycle[i] for i in [0, 1, 3, 2, 4, 5]]
LABELS = ["SIP (Ours)"] + METHODS[1:]
GRID = dict(which='major', axis='y', linestyle='--', linewidth=0.5)


# Plot loss for angle=45, ill_conditioning=[low (1), high (32)]
paths = [f"~/phi/sin_lin_net/{SEED}/{method} ill_1 angle_45_000000" for method in METHODS]
vis.plot_scalars(paths, x='time', names='x_dist', smooth=256, reduce='scenes', log_scale='y', titles=False, grid=GRID,
                 labels=wrap(METHODS, batch('scenes')), size=(1.8, 1.6), legend=False, smooth_alpha=0.1, smooth_linewidth=1, ylim=(2e-4, 2e-2))
vis.savefig("plots/sin/curves-well-conditioned-mixed-infinite.pdf")
vis.show()
pylab.close()
paths = [f"~/phi/sin_lin_net/{SEED}/{method} ill_32 angle_45_000000" for method in METHODS]
vis.plot_scalars(paths, x='time', names='x_dist', smooth=256, reduce='scenes', log_scale='y', titles=False,
                 labels=wrap(METHODS, batch('scenes')), size=(1.8, 1.6), legend=False, smooth_alpha=0.1, smooth_linewidth=1, grid=GRID)
vis.savefig("plots/sin/curves-ill-conditioned-mixed-infinite.pdf")
vis.show()


# Plot converged performance (last 10% steps) vs ill-conditioning for angle=0
# Plot converged performance (last 10% steps) vs angle [0-45] for ill-conditioning=medium
# fig, (ax1, ax2) = pylab.subplots(1, 2, figsize=(3.4, 1.6))
#
# ILL_COND_POINTS = [1, 3, 5, 10, 20, 32, 48, 64]
# for mi, method in enumerate(METHODS):
#     curve = []
#     for ill_conditioning in ILL_COND_POINTS:
#         scene = Scene.list(f"~/phi/sin_lin_net/{SEED}", name=f"{method} ill_{ill_conditioning} angle_0")[-1]
#         _, y_l2 = np.loadtxt(scene.subpath("log_x_dist.txt")).T
#         y_l2_last = y_l2[len(y_l2) * 90 // 100:] / ill_conditioning
#         curve.append(np.mean(y_l2_last))
#
#     ax1.plot(ILL_COND_POINTS, curve, label=LABELS[mi], color=COLORS[mi])
# ax1.set_xlabel("Conditioning $\\xi$")
# ax1.set_ylabel("Relative $x$-Distance")
# ax1.set_yscale('log')
#
# ANGLE_POINTS = [0, 1, 3, 5, 10, 23, 35, 40, 45]
# for mi, method in enumerate(METHODS):
#     curve = []
#     for angle in ANGLE_POINTS:
#         scene = Scene.list(f"~/phi/sin_lin_net/{SEED}", name=f"{method} ill_32 angle_{angle}")[-1]
#         _, y_l2 = np.loadtxt(scene.subpath("log_x_dist.txt")).T
#         y_l2_last = y_l2[len(y_l2) * 90 // 100:]
#         curve.append(np.mean(y_l2_last))
#     pylab.plot(ANGLE_POINTS, curve, label=LABELS[mi], color=COLORS[mi])
# ax2.set_xlabel("Rotation $\\phi$")  # covariance
# ax2.set_ylabel("Mean $x$-Distance")
# ax2.set_yscale('log')
#
# # pylab.legend()
# pylab.tight_layout()
# pylab.savefig("plots/sin/dist-by-conditioning-long.pdf", transparent=True)
# pylab.show()


# Plot loss landscape for ill-conditioning=2 and angle=20, target=(0.9, 0)






# Plot all loss curves as a row of figures (Appendix)
# for ill_conditioning in ILL_COND_POINTS:
#     paths = [f"~/phi/sin_lin_net/{SEED}/{method} ill_{ill_conditioning} angle_0_000000" for method in METHODS]
#     vis.plot_scalars(paths, x='time', names=['x_dist', 'y_l2'], smooth=64, reduce='scenes', log_scale='y',
#                      titles=f'$\\xi = {ill_conditioning}$', labels=wrap(METHODS, batch('scenes')))
#     vis.show()


# for angle in ANGLE_POINTS:
#     paths = [f"~/phi/sin_lin_net/{SEED}/{method} ill_32.0 angle_{angle}_000000" for method in METHODS]
#     vis.plot_scalars(paths, x='time', names=['x_dist', 'y_l2'], smooth=64, reduce='scenes', log_scale='y',
#                      titles=f'angle = {angle}$', labels=wrap(METHODS, batch('scenes')))
#     vis.show()





# Plot loss landscape for one example

# math.seed(0)
# BATCH = batch(batch=50)
# VECTOR = channel(vector=2)
# ill_conditioning = vis.control(1.)
# angle = vis.control(15., (0, 90))
# global_scale = vis.control(10.)
# target_x = vis.control(.3, (-1, 1))
# target_y = vis.control(-.5, (-1, 1))
#
#
# def process(vec):
#     vec = math.rotate_vector(vec, math.degrees(angle))
#     vec *= (global_scale / ill_conditioning, global_scale * ill_conditioning)
#     x, y = vec.vector.unstack()
#     x, y = math.sin(x), y
#     return math.stack([x, y], channel('vector'))
#
#
# def eval_loss(x):
#     return math.l2_loss(process(x) - (target_x, target_y))
#
#
# pylab.figure(figsize=(1.8, 1.4))
# grid_points = (math.meshgrid(batch, x=64, y=64) / 64 - .5) * 2 * (12, 6) / global_scale
# grid_points *= (ill_conditioning, 1/ill_conditioning) if angle <= 45 else (1/ill_conditioning, ill_conditioning)
# pylab.contour(grid_points.vector[0].numpy('y,x'), grid_points.vector[1].numpy('y,x'), eval_loss(grid_points).numpy('y,x') ** 0.2, cmap='Greys')
# pylab.tight_layout()
# pylab.savefig(f"plots/sin/loss-landscape {target_x} {target_y} angle_{angle} ill_{ill_conditioning}.pdf", transparent=True)
# pylab.show()

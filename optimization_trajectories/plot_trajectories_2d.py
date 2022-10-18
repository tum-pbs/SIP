import pylab
import numpy as np

pylab.rc('font', family='Arial', weight='normal', size=8)
# pylab.style.use('dark_background')
cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']


def path(method, x=(3, 3), steps=10, learning_rate=0.01):
    x = np.array(x)
    path = [x]
    for _ in range(steps):
        if method == 'gradient descent':
            x = x - learning_rate * np.array([2 * x[0], 4 * x[1] ** 3])
        elif method == 'quasi-Newton':
            x = x - learning_rate * np.array([x[0], x[1] / 3])
        elif method == 'inverse gradient':
            x = x - learning_rate * np.array([x[0], x[1] / 2])
        elif method == 'physical gradient':
            def inv(z):
                return np.array([z[0], np.sqrt(z[1])])

            z = np.array([x[0], x[1] ** 2])
            dz = - z * learning_rate
            dx = inv(z + dz) - x
            x = x + dx
        path.append(x)
    path = np.array(path)
    return path[:, 0], path[:, 1]


def z(path):
    return path[0], path[1] ** 2


x1, x2 = np.meshgrid(np.linspace(-0.5, 3.5, 100), np.linspace(-0.5, 3.5, 100))
L_x = x1 ** 2 + x2 ** 4
z1, z2 = np.meshgrid(np.linspace(-0.5, 3.5, 100), np.linspace(-0.5, 9.5, 100))
L_z = z1 ** 2 + z2 ** 2

lr_factor = 0.5

fig, (ax1, ax2) = pylab.subplots(1, 2, figsize=(8.7 / 2.54, 5 / 2.54))
ax1.contour(np.linspace(-0.5, 3.5, 100), np.linspace(-0.5, 3.5, 100), L_x, levels=[2 ** i for i in range(-5, 8)], colors='gray', linewidths=.5)
ax1.plot(*path('quasi-Newton', learning_rate=lr_factor), color=cycle[0], marker='o', label='Newton', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 1
ax1.plot(*path('quasi-Newton', learning_rate=0.1, steps=100), color=cycle[0])  # no-overshoot max: 1
# ax1.plot(*path('physical gradient', learning_rate=lr_factor), color=cycle[0], marker='o', label='PG', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 1
ax1.plot(*path('physical gradient', learning_rate=0.05, steps=100), color='grey', label='PI')  # no-overshoot max: 1
ax1.plot(*path('gradient descent', learning_rate=lr_factor / 36), color=cycle[1], marker='o', label='GD', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 0.03
ax1.plot(*path('gradient descent', learning_rate=0.005, steps=1000), color=cycle[1])
# ax1.plot(*path('inverse gradient', learning_rate=lr_factor), color=cycle[2], marker='o', label='IG', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 1
# ax1.plot(*path('inverse gradient', learning_rate=0.05, steps=100), color=cycle[2])  # no-overshoot max: 1
ax1.set(xlim=(-0.3, 3.2), ylim=(-0.1, 3.5), xlabel='$x_1$', ylabel='$x_2$')

pylab.contour(np.linspace(-0.5, 3.5, 100), np.linspace(-0.5, 9.5, 100), L_z, levels=[2 ** i for i in range(-5, 8)], colors='gray', linewidths=.5)
ax2.plot(*z(path('quasi-Newton', learning_rate=lr_factor)), color=cycle[0], marker='o', label='N', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 1
ax2.plot(*z(path('quasi-Newton', learning_rate=0.1, steps=100)), color=cycle[0])  # no-overshoot max: 1
# ax2.plot(*z(path('physical gradient', learning_rate=lr_factor)), color=cycle[0], marker='o', label='PG', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 1
ax2.plot(*z(path('physical gradient', learning_rate=0.05, steps=100)), color='grey', label='PI')  # no-overshoot max: 1
ax2.plot(*z(path('gradient descent', learning_rate=lr_factor / 36)), color=cycle[1], marker='o', label='GD', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 0.03
ax2.plot(*z(path('gradient descent', learning_rate=0.005, steps=1000)), color=cycle[1])
# ax2.plot(*z(path('inverse gradient', learning_rate=lr_factor)), color=cycle[2], marker='o', label='IG', linestyle='--', markersize=4, linewidth=0.8)  # no-overshoot max: 1
# ax2.plot(*z(path('inverse gradient', learning_rate=0.05, steps=100)), color=cycle[2])  # no-overshoot max: 1
ax2.legend(loc='upper left')
ax2.set(xlim=(-0.3, 3.2), ylim=(-0.1, 9.5), xlabel='$y_1$', ylabel='$y_2$')
pylab.tight_layout()
pylab.savefig('cubic_3.pdf', transparent=True, dpi=200)
pylab.show()
# pylab.close()

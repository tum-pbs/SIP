from phi.flow import *

phi.set_logging_level()

dirs = [[[f"~/phi/heat_net2/2/{method}_lr_{learning_rate}_bs{batch_size}_000000" for batch_size in [4, 32, 128]] for learning_rate in [1e-2, 1e-3, 1e-4]] for method in ['Adam + PG', 'Adam']]
dirs = math.layout(dirs, batch('method,learning_rate,batch_size'))
scenes = Scene.at(dirs)
titles = [[f"$\eta = {learning_rate}$, batch={batch_size}" for batch_size in [4, 32, 128]] for learning_rate in [1e-2, 1e-3, 1e-4]]
titles = math.layout(titles, batch('learning_rate,batch_size'))
# i = {'learning_rate': 0, 'batch_size': 0}
# scenes, titles = scenes[i], titles[i]

labels = math.wrap(["SIP", "Adam"], batch('method'))


vis.plot_scalars(scenes, 'x_l1', reduce='method', down='batch_size', titles=titles, x='time', labels=labels,
                 size=(7, 7), smooth=64, log_scale='y', ylim=(80, 1000))
import pylab
pylab.savefig("plots/heat/heat_hyperparameters.pdf", transparent=True)
vis.show()

# pylab.show()

#
# import pylab
#
# cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']
#
# data = [[30, 25, 50, 20],
#         [40, 23, 51, 17],
#         [35, 22, 45, 19]]
# X = np.arange(4)
# pylab.bar(X + 0.00, data[0], color='b', width=0.25)
# pylab.bar(X + 0.25, data[1], color='g', width=0.25)
# pylab.bar(X + 0.50, data[2], color='r', width=0.25)
# pylab.tight_layout()
# pylab.show()

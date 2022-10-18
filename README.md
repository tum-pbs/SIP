# Scale-invariant Learning by Physics Inversion (SIP)


This repository contains the code for the NeurIPS 2022 paper [Scale-invariant Learning by Physics Inversion](https://openreview.net/pdf?id=F2Gk6Vr3wu).
With the code published here, all experiments from the paper can be reproduced.



## Requirements

Python 3.6 or higher including [`pytorch`](https://pytorch.org/) and [`phiflow`](https://github.com/tum-pbs/PhiFlow).

## Using the Code

The code is ordered by experiment.

| Experiment                    | Figures | Source Code Directory                                  |
|-------------------------------|---------|--------------------------------------------------------|
| 2D Optimization               | 1       | [optimization_trajectories](optimization_trajectories) |
| Inverting the exponential     | 2       | [exp](exp)                                             |
| Experimental Characterization | 5       | [sin_characterization](sin_characterization)           |
| Poisson's equation            | 6a,b    | [poisson](poisson)                                     |
| Heat equation                 | 6c,d    | [heat](heat)                                           |
| Navier-Stokes equations       | 7       | [fluid](fluid)                                         |

[//]: # (| Wavepacket fitting            | 16      | [wavepacket]&#40;wavepacket&#41;                               |)

Inside the directories, you will find `train_*` and `plot_*` files.
No external configuration is required, the settings are adjusted within the Python files.
The `train_*` files train a neural network using the selected method and store checkpoints and learning curves in a subdirectory of `~/phi`.

Once the networks are trained, the `plot_*` files can be used to visualize the results. You need to fill in the correct paths before running them.

## Citing this Work
Please use the below citation:

```
@inproceedings{Holl2022Scale,
        title     = {Scale-invariant Learning by Physics Inversion},
        author    = {Philipp Holl and Vladlen Koltun and Nils Thuerey},
        booktitle = {Conference on Neural Information Processing Systems},
        year      = {2022},
}
```
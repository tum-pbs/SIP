import numpy as np

from phi import math
from phi.field import Grid, SoftGeometryMask, CenteredGrid
from phi.geom._box import Cuboid, Box
from phi.math import Tensor, extrapolation, batch, channel


def generate_heat_example(*shape, bounds: Box = None):
    shape = math.merge_shapes(*shape)
    heat_t0 = CenteredGrid(0, extrapolation.PERIODIC, bounds, resolution=shape.spatial)
    bounds = heat_t0.bounds
    component_counts = math.to_int32(4 + 7 * math.random_uniform(shape.batch))
    positions = (math.random_uniform(shape.batch, batch(components=10), channel(vector=shape.spatial.names)) - 0.5) * bounds.size * 0.8 + bounds.size * 0.5
    for i in range(10):
        position = positions.components[i]
        half_size = math.random_uniform(shape.batch, channel(vector=shape.spatial.names)) * 10
        strength = math.random_uniform(shape.batch) * math.to_float(i < component_counts)
        position = math.clip(position, bounds.lower + half_size, bounds.upper - half_size)
        component_box = Cuboid(position, half_size)
        component_mask = SoftGeometryMask(component_box)
        component_mask = component_mask.at(heat_t0)
        heat_t0 += component_mask * strength
    return heat_t0


def apply_damping(kernel, inv_kernel, amp, f_uncertainty, log_kernel):
    signal_prior = 0.5
    expected_amp = 1. * kernel.shape.get_size('x') * inv_kernel  # This can be measured
    signal_likelihood = math.exp(-0.5 * (abs(amp) / expected_amp) ** 2) * signal_prior  # this can be NaN
    signal_likelihood = math.where(math.isfinite(signal_likelihood), signal_likelihood, math.zeros_like(signal_likelihood))
    noise_likelihood = math.exp(-0.5 * (abs(amp) / f_uncertainty) ** 2) * (1 - signal_prior)
    probability_signal = math.divide_no_nan(signal_likelihood, (signal_likelihood + noise_likelihood))
    action = math.where((0.5 >= probability_signal) | (probability_signal >= 0.68), 2 * (probability_signal - 0.5), 0.)  # 1 sigma required to take action
    prob_kernel = math.exp(log_kernel * action)
    return prob_kernel, probability_signal


def inv_diffuse(grid: Grid, amount: float, uncertainty: Grid):
    f_uncertainty: Tensor = math.sqrt(math.sum(uncertainty.values ** 2, dim='x,y'))  # all frequencies have the same uncertainty, 1/N in iFFT
    k_squared: Tensor = math.sum(math.fftfreq(grid.shape, grid.dx) ** 2, 'vector')
    fft_laplace: Tensor = -(2 * np.pi) ** 2 * k_squared
    # --- Compute sharpening kernel with damping ---
    log_kernel = fft_laplace * -amount
    log_kernel_clamped = math.minimum(log_kernel, math.to_float(math.floor(math.log(math.wrap(np.finfo(np.float32).max)))))  # avoid overflow
    raw_kernel = math.exp(log_kernel_clamped)  # inverse diffusion FFT kernel, all values >= 1
    inv_kernel = math.exp(-log_kernel)
    amp = math.fft(grid.values)
    kernel, sig_prob = apply_damping(raw_kernel, inv_kernel, amp, f_uncertainty, log_kernel)
    # --- Apply and compute uncertainty ---
    data = math.real(math.ifft(amp * math.to_complex(kernel)))
    uncertainty = math.sqrt(math.sum(((f_uncertainty * kernel) ** 2))) / grid.shape.get_size('x')  # 1/N normalization in iFFT
    uncertainty = grid * 0 + uncertainty
    return grid.with_values(data), uncertainty, abs(amp), raw_kernel, kernel, sig_prob

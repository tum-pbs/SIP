from phi.torch.flow import *
from phi.field._field_math import discretize
from phi.vis._vis import record


TORCH.set_default_device('GPU')
math.seed(0)

DOMAIN = dict(x=64, y=64, extrapolation=extrapolation.PERIODIC, bounds=Box[0:128, 0:128])
TIME = 2
STEPS = 8
DT = TIME / STEPS
BATCH = batch(batch=64)


def match_loss(actual_marker: Grid, target: Grid):
    return field.frequency_loss(actual_marker - target, n=1, frequency_falloff=80, ignore_mean=True)


def physics_step(marker: Grid,
                 velocity: StaggeredGrid,  # divergence-free
                 pressure_guess: Grid):
    """ Energy-conserving inviscid incompressible Navier-Stokes with passive marker advection """
    initial_energy = field.mean(velocity ** 2)
    velocity = advect.mac_cormack(velocity, velocity, DT)  # first so that the returned velocity matches the marker advection
    velocity, pressure_guess = fluid.make_incompressible(velocity, solve=Solve('CG', 0, 1e-4, x0=pressure_guess))
    energy = field.mean(velocity ** 2)
    velocity *= math.where(energy == 0, energy, math.sqrt(initial_energy / energy))  # avoid NaN
    marker = advect.mac_cormack(marker, velocity, DT)
    return marker, velocity, pressure_guess


def generate_example():
    with math.NUMPY:
        pos = math.random_uniform(BATCH, channel(vector=2)) * 64 + 32
        noise = CenteredGrid(Noise(BATCH, scale=10), **DOMAIN) * CenteredGrid(geom.Box(pos - 28, pos + 28), **DOMAIN)
        fill_fraction = math.random_uniform() * 0.1 + 0.05
    marker_0 = discretize(noise, fill_fraction) * CenteredGrid(geom.Box(pos - 28, pos + 28), **DOMAIN)
    uniform_velocity = CenteredGrid(64 - field.center_of_mass(marker_0), **DOMAIN) / TIME
    vorticity = (math.random_uniform(BATCH) - 0.5) * 4
    swirl = CenteredGrid(field.AngularVelocity([64, 64], strength=vorticity, falloff=lambda d: math.exp(-0.5 * math.vec_squared(d) / 20 ** 2)), **DOMAIN)
    fluctuations = CenteredGrid(Noise(BATCH, channel(vector=2)), **DOMAIN) * 0.1
    velocity = uniform_velocity + swirl + fluctuations
    pressure = CenteredGrid(0, **DOMAIN)
    marker_t = marker_0
    fwd = record(marker_t, velocity)
    print("Ref:", end=" ")
    for _ in fwd.range(frames=STEPS):
        marker_t, velocity, pressure = physics_step(marker_t, velocity, pressure)
    print()
    return marker_0, marker_t, fwd.rec.marker_t, fwd.rec.velocity


def eval_physics_loss(v0: Grid, marker_keys: Grid):
    """ Run forward simulation and compute nn_loss. """
    marker = marker_keys.keyframe[0]
    velocity, _ = fluid.make_incompressible(StaggeredGrid(v0, **DOMAIN))
    pressure = CenteredGrid(0, **DOMAIN)
    fwd = record(marker, velocity)
    for _ in fwd.range(frames=STEPS):
        marker, velocity, pressure = physics_step(marker, velocity, pressure)
    key_loss = match_loss(marker, marker_keys.keyframe[-1])
    return key_loss, fwd.rec.marker, fwd.rec.velocity


physics_gradient = field.functional_gradient(eval_physics_loss, get_output=True)


def estimate_v0(v0: Grid, marker_keys: Grid):
    initial_vorticity = math.mean(math.cross_product(-v0.values, v0.points - 64), v0.shape.spatial) / 120.5
    """ Run forward and reverse simulation to compute update. """
    # Forward
    marker = marker_keys.keyframe[0]
    velocity, *_ = fluid.make_incompressible(StaggeredGrid(v0, **DOMAIN))
    pressure = CenteredGrid(0, **DOMAIN)
    coms = [field.center_of_mass(marker)]
    fwd = record(marker, velocity)
    for _ in fwd.range(frames=STEPS):
        marker, velocity, pressure = physics_step(marker, velocity, pressure)
        coms.append(field.center_of_mass(marker))
    # Delta
    key_loss = match_loss(marker, marker_keys.keyframe[-1])
    uniform_velocity = CenteredGrid(64 - field.center_of_mass(marker_keys.keyframe[0]), **DOMAIN) / TIME
    rev_com = field.center_of_mass(marker_keys.keyframe[-1])
    swirl_pos = 64 + (64 - field.center_of_mass(marker_keys.keyframe[0]))
    d_vorticities = [math.cross_product(coms[-1] - rev_com, rev_com - swirl_pos)]
    # Reverse
    marker = marker_keys.keyframe[1]
    velocity = -velocity
    pressure = CenteredGrid(0, **DOMAIN)
    rev = record(marker, velocity)
    for i in rev.range(frames=STEPS):
        marker, velocity, pressure = physics_step(marker, velocity, pressure)
        rev_com = field.center_of_mass(marker)
        swirl_pos = 64 + (64 - field.center_of_mass(marker_keys.keyframe[0])) / TIME * (STEPS - 1 - i)
        d_vorticities.append(math.cross_product(coms[-2-i] - rev_com, rev_com - swirl_pos))
    vorticity_delta = math.mean(d_vorticities, dim='0') * 1e-3
    adjusted_swirl = CenteredGrid(field.AngularVelocity([64, 64], strength=initial_vorticity + vorticity_delta, falloff=lambda d: math.exp(-0.5 * math.vec_squared(d) / 20 ** 2)), **DOMAIN)
    corrected_v0 = uniform_velocity + adjusted_swirl
    return corrected_v0, key_loss, fwd.rec.marker, fwd.rec.velocity, rev.rec.marker.frames[::-1], rev.rec.velocity.frames[::-1]

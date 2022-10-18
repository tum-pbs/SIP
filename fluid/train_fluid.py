from phi.torch.flow import *
from phi.field._field_math import discretize
from phi.vis._vis import record


TORCH.set_default_device('GPU')
math.seed(0)

DOMAIN = dict(x=64, y=64, extrapolation=extrapolation.PERIODIC, bounds=Box(x=128, y=128))
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
        pos = math.random_uniform(BATCH, channel(vector='x,y')) * 64 + 32
        noise = CenteredGrid(Noise(BATCH, scale=10), **DOMAIN) * CenteredGrid(Box(pos - 28, pos + 28), **DOMAIN)
        fill_fraction = math.random_uniform() * 0.1 + 0.05
    marker_0 = discretize(noise, fill_fraction) * CenteredGrid(geom.Box(pos - 28, pos + 28), **DOMAIN)
    uniform_velocity = CenteredGrid(64 - field.center_of_mass(marker_0), **DOMAIN) / TIME
    vorticity = (math.random_uniform(BATCH) - 0.5) * 4
    swirl = CenteredGrid(field.AngularVelocity(vec(x=64, y=64), strength=vorticity, falloff=lambda d: math.exp(-0.5 * math.vec_squared(d) / 20 ** 2)), **DOMAIN)
    fluctuations = CenteredGrid(Noise(BATCH, channel(vector='x,y')), **DOMAIN) * 0.1
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
    adjusted_swirl = CenteredGrid(field.AngularVelocity(vec(x=64, y=64), strength=initial_vorticity + vorticity_delta, falloff=lambda d: math.exp(-0.5 * math.vec_squared(d) / 20 ** 2)), **DOMAIN)
    corrected_v0 = uniform_velocity + adjusted_swirl
    return corrected_v0, key_loss, fwd.rec.marker, fwd.rec.velocity, rev.rec.marker.frames[::-1], rev.rec.velocity.frames[::-1]


net = u_net(2, 2, levels=5, filters=16)
print(f"Parameter count: {parameter_count(net)}")

learning_rate = vis.control(0.005)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
method = 'PG'
viewer = view('markers, pred_v0_no_grad, delta_n, delta, uniform_velocity, correction, delta_n', scene=True, select='frames,batch', namespace=globals())
viewer.info(f"Training method: {method}")
torch.save(net.state_dict(), viewer.scene.subpath('net0.pth'))
math.seed(0)


def reset():  # called by UI
    net.load_state_dict(torch.load(viewer.scene.subpath('net0.pth')))
    math.seed(0)


for opt_step in viewer.range():
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    optimizer.zero_grad()
    m0, mt, gt_markers, gt_velocities = generate_example()
    train_marker_keys = field.stack([m0, mt], channel('keyframe'))
    # Predict
    prediction = math.native_call(net, train_marker_keys.values)
    pred_v0 = CenteredGrid(prediction, **DOMAIN)
    pred_v0_no_grad = field.stop_gradient(pred_v0)
    # PG
    if method == 'PG':
        correction, physics_loss, fm, velocities, rm, rv = estimate_v0(pred_v0_no_grad, train_marker_keys)
        nn_loss = field.l2_loss(pred_v0 - correction)
        markers = fm - train_marker_keys.keyframe[-1]
        delta = correction - pred_v0_no_grad
    # GD
    elif method == 'GD':
        physics_loss, markers, velocities = eval_physics_loss(pred_v0, train_marker_keys)
        nn_loss = physics_loss  # optimize network using physics nn_loss
    # GD Loss indirect
    # physics_loss, lk1, lk2, lf, m1, m2, forces_grad = physics_gradient(pred_v0_no_grad, train_marker_keys)
    # correction = field.stop_gradient(pred_forces - forces_grad)
    # nn_loss = field.l2_loss(pred_forces - correction)
    # GD accumulate
    elif method == 'GD_accumulate':
        physics_lr = 4e-2
        try:
            physics_loss, markers, velocities, v0_grad = physics_gradient(pred_v0_no_grad, train_marker_keys)
            v0_grad0 = v0_grad
            correction = pred_v0_no_grad - physics_lr * v0_grad
            for _ in range(7):
                correction -= physics_lr * physics_gradient(correction, train_marker_keys)[-1]  # TODO memory leak
            total_neg_grad = correction - pred_v0_no_grad
            # angle = -total_neg_grad, forces_grad0
            norm_g0 = math.sqrt(field.mean(field.vec_squared(v0_grad0)))
            norm_g = math.sqrt(field.mean(field.vec_squared(total_neg_grad)))
            total_neg_grad *= norm_g0 / norm_g
            correction = pred_v0_no_grad + total_neg_grad
            nn_loss = field.l2_loss(pred_v0 - correction)
        except ConvergenceException as c_exc:
            viewer.info(str(c_exc))
            continue
    # BFGS
    elif method == 'BFGS':
        physics_loss, markers, velocities = eval_physics_loss(pred_v0_no_grad, train_marker_keys)  # stop_gradient() avoids memory leak since custom_gradient not needed
        correction = field.minimize(lambda v0: eval_physics_loss(v0, train_marker_keys)[0],
                                    Solve('L-BFGS-B', 0, 1e-5, max_iterations=16, x0=pred_v0_no_grad, suppress=[NotConverged]))
        nn_loss = field.l2_loss(pred_v0 - field.stop_gradient(correction))
    else:
        raise ValueError(method)
    # Update
    v_dist = field.vec_abs(velocities - gt_velocities)
    v0_dist = v_dist.frames[0]
    viewer.log_scalars(loss=physics_loss.mean, nn_loss=nn_loss.mean, gt_v_l1=field.l1_loss(v_dist).mean, gt_v0_l1=field.l1_loss(v0_dist).mean)
    nn_loss.mean.backward()
    optimizer.step()

    # m0 = m1 = m2 = k1 = k2 = prediction = pred_forces = pred_forces0 = train_marker_keys = physics_loss = lk1 = lk2 = lf = nn_loss = noise = correction = fill_fraction = None
    # count_tensors_in_memory()

    if opt_step % 10 == 0:
        torch.save(net.state_dict(), viewer.scene.subpath('net.pth'))

    if opt_step % 100 == 0:
        torch.save(net.state_dict(), viewer.scene.subpath(f'net_{opt_step}.pth'))

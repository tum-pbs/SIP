from phi.flow import math
import torch


@math.jit_compile
def newton_minimization(grad: math.Tensor, hessian: math.Tensor, max_loss_change=None) -> math.Tensor:
    """
    Computes a Newton update step that always points towards a minimum of the function.

    Args:
        grad: Gradient vector.
        hessian: Hessian matrix.
        max_loss_change: (Optional) If set, adds a limit to the gradient magnitude, approximately restricting by how much the loss can decrease per step.
            This affects directions in which the objective curvature is small but the gradient is large.
            The limit is computed as `max_loss_change / grad` in principal component space.
            If not set, large updates along these dimensions can occur which may result in divergence.

    Returns:
        Newton direction as `Tensor` matching `grad`.
    """
    assert hessian.shape.non_batch.rank == 2, "Not yet implemented"
    batch = hessian.shape.batch
    h_native = math.reshaped_native(hessian, [*batch, *hessian.shape.non_batch])
    # Transform to eigen space
    S, V = torch.linalg.eigh(h_native)  # identity = V.v * V_.v
    S = math.wrap(S, batch, math.channel('vector'))  # scales like 1/global_scale**2
    V = math.reshaped_tensor(V, [*batch, grad.shape.non_batch, math.channel('reduce')])
    V_ = math.reshaped_tensor(V, [*batch, math.channel('reduce'), grad.shape.non_batch])
    grad_eig = grad.vector * V.reduce
    newton_eig = - math.divide_no_nan(grad_eig, S) # scales like 1/global_scale
    S_avg = abs(math.prod(S, 'vector')) ** (1/S.vector.size)
    # S_rel = abs(S / S_avg)
    if max_loss_change is not None:
        inv_grad_eig = abs(math.divide_no_nan(max_loss_change, grad_eig))  # limit step by 1/grad when S is small
        # limit_fac = 1 / (1 + S_rel)  # 1 for small relative curvatures
        # limit = limit_fac * inv_grad_eig + (1 - limit_fac) * abs(newton_eig)
        limit = math.where(S > S_avg, abs(newton_eig), inv_grad_eig)
        newton_eig = math.minimum(abs(newton_eig), limit) * math.sign(-grad_eig)
    else:
        newton_eig = math.sign(-grad_eig) * abs(newton_eig)  # sign flip to descending direction
    # Transform back to parameter space
    newton_min = newton_eig.vector * V_.reduce
    return newton_min


# def newton_minimization(grad: math.Tensor, hessian: math.Tensor, max_loss_change=None) -> math.Tensor:
#     """
#     Computes a Newton update step that always points towards a minimum of the function.
#
#     Args:
#         grad: Gradient vector.
#         hessian: Hessian matrix.
#         max_loss_change: (Optional) If set, adds a limit to the gradient magnitude, approximately restricting by how much the loss can decrease per step.
#             This affects directions in which the objective curvature is small but the gradient is large.
#             The limit is computed as `max_loss_change / grad` in principal component space.
#             If not set, large updates along these dimensions can occur which may result in divergence.
#
#     Returns:
#         Newton direction as `Tensor` matching `grad`.
#     """
#     assert hessian.shape.non_batch.rank == 2, "Not yet implemented"
#     batch = hessian.shape.batch
#     h_native = math.reshaped_native(hessian, [*batch, *hessian.shape.non_batch])
#     grad_native = math.reshaped_native(grad, [*batch, grad.shape.non_batch])
#     newton_any = - torch.linalg.solve(h_native, grad_native)
#     newton_any = math.reshaped_tensor(newton_any, [*grad.shape.batch, grad.shape.non_batch])
#     # Transform to eigen space
#     S, V = torch.linalg.eigh(h_native)  # identity = V.v * V_.v
#     S = math.wrap(S, batch, math.channel('basis'))
#     V = math.reshaped_tensor(V, [*batch, grad.shape.non_batch, math.channel('reduce')])
#     V_ = math.reshaped_tensor(V, [*batch, math.channel('reduce'), grad.shape.non_batch])
#     grad_princ = grad.vector * V.reduce
#     newton_princ = newton_any.vector * V.reduce
#     if max_loss_change is not None:
#         inv_grad_princ = abs(math.divide_no_nan(max_loss_change, grad_princ))  # limit step by 1/grad
#         newton_princ = math.minimum(abs(newton_princ), inv_grad_princ) * math.sign(-grad_princ)
#     else:
#         newton_princ = math.sign(-grad_princ) * abs(newton_princ)  # sign flip to descending direction
#     # Transform back to parameter space
#     newton_min = newton_princ.vector * V_.reduce
#     return newton_min


def damped_newton_minimization(grad: math.Tensor, hessian: math.Tensor, damping=0.1, max_loss_change=None) -> math.Tensor:
    """
    Computes a Newton update step that always points towards a minimum of the function.

    Args:
        grad: Gradient vector.
        hessian: Hessian matrix.
        max_loss_change: (Optional) If set, adds a limit to the gradient magnitude, approximately restricting by how much the loss can decrease per step.
            This affects directions in which the objective curvature is small but the gradient is large.
            The limit is computed as `max_loss_change / grad` in principal component space.
            If not set, large updates along these dimensions can occur which may result in divergence.

    Returns:
        Newton direction as `Tensor` matching `grad`.
    """
    assert hessian.shape.non_batch.rank == 2, "Not yet implemented"
    batch = hessian.shape.batch
    h_native = math.reshaped_native(hessian, [*batch, *hessian.shape.non_batch])
    grad_native = math.reshaped_native(grad, [*batch, grad.shape.non_batch])
    identity = torch.diag(torch.ones(grad.shape.non_batch.volume) * damping)
    newton_any = - torch.linalg.solve(h_native + identity, grad_native)
    newton_any = math.reshaped_tensor(newton_any, [*grad.shape.batch, grad.shape.non_batch])
    # Transform to eigen space
    S, V = torch.linalg.eigh(h_native)  # identity = V.v * V_.v
    S = math.wrap(S, batch, math.channel('basis'))
    V = math.reshaped_tensor(V, [*batch, grad.shape.non_batch, math.channel('reduce')])
    V_ = math.reshaped_tensor(V, [*batch, math.channel('reduce'), grad.shape.non_batch])
    grad_princ = grad.vector * V.reduce  # scales like global_scale
    newton_princ = newton_any.vector * V.reduce  # scales like 1/global_scale
    if max_loss_change is not None:
        inv_grad_princ = abs(math.divide_no_nan(max_loss_change, grad_princ))  # limit step by 1/grad
        newton_princ = math.minimum(abs(newton_princ), inv_grad_princ) * math.sign(-grad_princ)
    else:
        newton_princ *= math.sign(-grad_princ) * math.sign(newton_princ)  # sign flip to descending direction
    # Transform back to parameter space
    newton_min = newton_princ.vector * V_.reduce
    return newton_min


def hessian_eigen(grad: math.Tensor, hessian: math.Tensor):
    assert hessian.shape.non_batch.rank == 2, "Not yet implemented"
    batch = hessian.shape.batch
    h_native = math.reshaped_native(hessian, [*batch, *hessian.shape.non_batch])
    # Transform to eigen space
    S, V = torch.linalg.eigh(h_native)  # identity = V.v * V_.v
    S = math.wrap(S, batch, math.channel('basis'))
    V = math.reshaped_tensor(V, [*batch, *math.channel('vector,reduce')])
    V_ = math.reshaped_tensor(V, [*batch, *math.channel('reduce,vector')])
    unit = math.tensor([(1, 0), (0, 1)], math.channel(basis='x,y'), math.channel(vector=2)).vector * V_.reduce
    grad_eig = grad.vector * V.reduce  # scales like global_scale
    ig_eig = math.divide_no_nan(1., grad_eig)
    ig = ig_eig.vector * V_.reduce
    return unit, unit * S, ig

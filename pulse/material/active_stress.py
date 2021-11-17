import dolfin

try:
    from dolfin_adjoint import Constant
except ImportError:
    from dolfin import Constant


def Wactive_transversally(Ta, C, f0, eta=0.0):
    """
    Return active strain energy when activation is only
    working along the fibers, with a possible transverse
    component defined by eta

    Arguments
    ---------
    Ta : dolfin.Function or dolfin.Constant
        A scalar function representng the mangnitude of the
        active stress in the reference configuration (firt Pioala)
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfin.Function
        A vector function representng the direction of the
        active stress
    eta : float
        Amount of active stress in the transverse direction
        (relative to f0)
    """

    I4f = dolfin.inner(C * f0, f0)
    I1 = dolfin.tr(C)
    return Constant(0.5) * Ta * ((I4f - 1) + eta * ((I1 - 3) - (I4f - 1)))


def Wactive_orthotropic(Ta, C, f0, s0, n0):
    """Return active strain energy for an orthotropic
    active stress


    Arguments
    ---------
    Ta : dolfin.Function or dolfin.Constant
        A vector function representng the mangnitude of the
        active stress in the reference configuration (firt Pioala).
        Ta = (Ta_f0, Ta_s0, Ta_n0)
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfin.Function
        A vector function representng the direction of the
        first component
    s0 : dolfin.Function
        A vector function representng the direction of the
        second component
    n0 : dolfin.Function
        A vector function representng the direction of the
        third component
    """
    I4f = dolfin.inner(C * f0, f0)
    I4s = dolfin.inner(C * s0, s0)
    I4n = dolfin.inner(C * n0, n0)

    I4 = dolfin.as_vector([I4f - 1, I4s - 1, I4n - 1])
    return Constant(0.5) * dolfin.inner(Ta, I4)


def Wactive_anisotropic(Ta, C, f0, s0, n0):
    """Return active strain energy for a fully anisotropic
    acitvation.
    Note that the three basis vectors are assumed to be
    orthogonal

    Arguments
    ---------
    Ta : dolfin.Function or dolfin.Constant
        A full tensor function representng the active stress tensor
        of the active stress in the reference configuration
        (firt Pioala).
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfin.Function
        A vector function representng the direction of the
        first component
    s0 : dolfin.Function
        A vector function representng the direction of the
        second component
    n0 : dolfin.Function
        A vector function representng the direction of the
        third component
    """
    I4f = dolfin.inner(C * f0, f0)
    I4s = dolfin.inner(C * s0, s0)
    I4n = dolfin.inner(C * n0, n0)

    I8fs = dolfin.inner(C * f0, s0)
    I8fn = dolfin.inner(C * f0, n0)
    I8sn = dolfin.inner(C * s0, n0)

    A = dolfin.as_matrix(
        (
            (0.5 * (I4f - 1), I8fs, I8fn),
            (I8fs, 0.5 * (I4s - 1), I8sn),
            (I8fn, I8sn, 0.5 * (I4n - 1)),
        ),
    )

    return dolfin.inner(Ta, A)

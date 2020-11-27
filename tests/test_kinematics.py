import dolfin as df
import pytest

from pulse import kinematics


@pytest.fixture
def F_2D_Real():
    mesh = df.UnitSquareMesh(2, 2)
    V = df.TensorFunctionSpace(mesh, "R", 0)
    F = df.Function(V)
    F.assign(df.Constant([[1.1, 0], [0.0, 1 / 1.1]]))
    return F


@pytest.fixture
def F_2D_CG1():
    mesh = df.UnitSquareMesh(2, 2)
    V = df.TensorFunctionSpace(mesh, "CG", 1)
    F = df.Function(V)
    F.assign(df.Constant([[1.1, 0], [0.0, 1 / 1.1]]))
    return F


@pytest.fixture
def F_3D_Real():
    mesh = df.UnitCubeMesh(2, 2, 2)
    V = df.TensorFunctionSpace(mesh, "R", 0)
    F = df.Function(V)
    F.assign(df.Constant([[1.1, 0, 0], [0.0, 1 / 1.1, 0], [0, 0, 1]]))
    return F


@pytest.fixture
def F_3D_CG1():
    mesh = df.UnitCubeMesh(2, 2, 2)
    V = df.TensorFunctionSpace(mesh, "CG", 1)
    F = df.Function(V)
    F.assign(df.Constant([[1.1, 0, 0], [0.0, 1 / 1.1, 0], [0, 0, 1]]))
    return F


@pytest.fixture
def invariants():
    inv = kinematics.Invariants()
    return inv


@pytest.fixture
def F_dict(F_2D_Real, F_2D_CG1, F_3D_Real, F_3D_CG1):
    return {
        "F_2D_Real": F_2D_Real,
        "F_2D_CG1": F_2D_CG1,
        "F_3D_Real": F_3D_Real,
        "F_3D_CG1": F_3D_CG1,
    }


@pytest.mark.parametrize(
    "F_str, dim", [("F_2D_Real", 2), ("F_2D_CG1", 2), ("F_3D_Real", 3), ("F_3D_CG1", 3)]
)
def test_SecondOderIdetity(F_dict, F_str, dim):
    F = F_dict[F_str]
    Id = kinematics.SecondOrderIdentity(F)
    assert Id == df.Identity(dim)


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_EngineeringStrain(F_dict, F_str):
    F = F_dict[F_str]
    E = kinematics.EngineeringStrain(F)
    assert abs(df.assemble((E[0, 0]) * df.dx) - (1.1 - 1)) < 1e-12
    assert abs(df.assemble((E[1, 1]) * df.dx) - (1 / 1.1 - 1)) < 1e-12
    assert abs(df.assemble((E[1, 0]) * df.dx)) < 1e-12
    assert abs(df.assemble((E[0, 1]) * df.dx)) < 1e-12


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_GreenLagrangeStrain(F_dict, F_str):
    F = F_dict[F_str]
    E = kinematics.GreenLagrangeStrain(F)
    assert abs(df.assemble((E[0, 0]) * df.dx) - (0.5 * (1.1 ** 2 - 1))) < 1e-12
    assert abs(df.assemble((E[1, 1]) * df.dx) - (0.5 * ((1 / 1.1) ** 2 - 1))) < 1e-12
    assert abs(df.assemble((E[1, 0]) * df.dx)) < 1e-12
    assert abs(df.assemble((E[0, 1]) * df.dx)) < 1e-12


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I1(invariants, F_dict, F_str):
    F = F_dict[F_str]
    extra = 0 if "2D" in F_str else 1.0
    # Trace of F**2
    assert (
        abs(
            df.assemble(invariants._I1(F) * df.dx) - (1.1 ** 2 + (1 / 1.1) ** 2 + extra)
        )
        < 1e-12
    )


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I2(invariants, F_dict, F_str):
    F = F_dict[F_str]
    extra = 0 if "2D" in F_str else 1.0

    assert (
        abs(
            df.assemble(invariants._I2(F) * df.dx)
            - 0.5
            * (
                (1.1 ** 2 + (1 / 1.1) ** 2 + extra) ** 2
                - (1.1 ** 4 + (1 / 1.1) ** 4 + extra)
            )
        )
        < 1e-12
    )


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I3(invariants, F_dict, F_str):
    F = F_dict[F_str]
    assert abs(df.assemble(invariants._I3(F) * df.dx) - 1.1 * (1 / 1.1)) < 1e-12


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I4x(invariants, F_dict, F_str):
    F = F_dict[F_str]

    if "2D" in F_str:
        x = df.as_vector([1, 0])
    else:
        x = df.as_vector([1, 0, 0])

    assert abs(df.assemble(invariants._I4(F, x) * df.dx) - 1.1 ** 2) < 1e-12


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I4y(invariants, F_dict, F_str):
    F = F_dict[F_str]

    if "2D" in F_str:
        y = df.as_vector([0, 1])
    else:
        y = df.as_vector([0, 1, 0])

    assert abs(df.assemble(invariants._I4(F, y) * df.dx) - (1 / 1.1) ** 2) < 1e-12


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I5x(invariants, F_dict, F_str):
    F = F_dict[F_str]

    if "2D" in F_str:
        x = df.as_vector([1, 0])
    else:
        x = df.as_vector([1, 0, 0])

    assert abs(df.assemble(invariants._I5(F, x) * df.dx) - 1.1 ** 4) < 1e-12


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I5y(invariants, F_dict, F_str):
    F = F_dict[F_str]

    if "2D" in F_str:
        y = df.as_vector([0, 1])
    else:
        y = df.as_vector([0, 1, 0])

    assert abs(df.assemble(invariants._I5(F, y) * df.dx) - (1 / 1.1) ** 4) < 1e-12


def test_I6():
    """I6 is identical to I4"""
    pass


def test_I7():
    """I7 is identical to I5"""
    pass


@pytest.mark.parametrize("F_str", ["F_2D_Real", "F_2D_CG1", "F_3D_Real", "F_3D_CG1"])
def test_I8xy(invariants, F_dict, F_str):
    F = F_dict[F_str]

    if "2D" in F_str:
        x = df.as_vector([1, 0])
        y = df.as_vector([0, 1])
    else:
        x = df.as_vector([1, 0, 0])
        y = df.as_vector([0, 1, 0])

    assert (
        abs(
            df.assemble(invariants._I8(F, x, y) * df.dx)
            - (1.1 ** 2 * 0 + 0 * (1 / 1.1))
        )
        < 1e-12
    )

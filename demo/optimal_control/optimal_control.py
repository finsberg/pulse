"""
To run this you need to install dolfin-adjoint and cyipopt
"""
import dolfin as df
import dolfin_adjoint as da
import matplotlib.pyplot as plt

import pulse


def cost_function(u_model, u_data):
    norm = lambda f: da.assemble(df.inner(f, f) * df.dx)
    return norm(u_model - u_data)


def create_forward_problem(
    mesh,
    activation,
    active_value=0.0,
    active_model="active_stress",
    T_ref=1.0,
):

    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    left = df.CompiledSubDomain("on_boundary && near(x[0], 0)")
    left_marker = 1
    left.mark(ffun, left_marker)

    # Collect the functions containing the markers
    marker_functions = pulse.MarkerFunctions(ffun=ffun)

    def dirichlet_bc(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        return da.DirichletBC(V, da.Constant((0.0, 0.0, 0.0)), left)

    bcs = pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
    )

    f0 = df.as_vector([1, 0, 0])
    microstructure = pulse.Microstructure(f0=f0)

    geometry = pulse.Geometry(
        mesh=mesh,
        marker_functions=marker_functions,
        microstructure=microstructure,
    )

    material_parameters = dict(
        a=2.28,
        a_f=1.686,
        b=9.726,
        b_f=15.779,
        a_s=0.0,
        b_s=0.0,
        a_fs=0.0,
        b_fs=0.0,
    )
    material = pulse.HolzapfelOgden(
        active_model=active_model,
        parameters=material_parameters,
        activation=activation,
        T_ref=T_ref,
    )

    problem = pulse.MechanicsProblem(geometry, material, bcs)
    problem.solve()

    if active_value > 0.0:
        pulse.iterate.iterate(problem, activation, active_value)

    return problem


def main():
    N = 5
    mesh = da.UnitCubeMesh(N, N, N)

    W = df.FunctionSpace(mesh, "CG", 1)
    T_ref = 10.0

    active = da.Function(W)
    problem = create_forward_problem(mesh, active, active_value=0.3, T_ref=T_ref)
    u, _ = problem.state.split()

    V = df.VectorFunctionSpace(mesh, "CG", 2)
    u_synthetic = da.project(u, V)

    active_ctrl = da.Function(W)
    problem = create_forward_problem(mesh, active_ctrl, active_value=0.0, T_ref=T_ref)
    u_model, _ = problem.state.split()

    J = cost_function(
        u_model,
        u_synthetic,
    )

    cost_func_values = []
    control_values = []

    def eval_cb(j, m):
        """Callback function"""
        cost_func_values.append(j)
        control_values.append(m)

    reduced_functional = da.ReducedFunctional(
        J,
        da.Control(active_ctrl),
        eval_cb_post=eval_cb,
    )

    problem = da.MinimizationProblem(reduced_functional, bounds=[(0, 1)])

    parameters = {
        "limited_memory_initialization": "scalar2",
        "maximum_iterations": 10,
    }
    solver = da.IPOPTSolver(problem, parameters=parameters)

    optimal_control = solver.solve()

    control_error = [df.assemble((c - active) ** 2 * df.dx) for c in control_values]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].semilogy(cost_func_values)
    ax[0].set_xlabel("#iterations")
    ax[0].set_ylabel("cost function")

    ax[1].semilogy(control_error)
    ax[1].set_xlabel("#iterations")
    ax[1].set_ylabel(r"$\int (\gamma - \gamma^*) \mathrm{d}x$")
    fig.tight_layout()
    fig.savefig("results")

    mean = optimal_control.vector().get_local().mean()
    std = optimal_control.vector().get_local().std()
    print(f"Optimal control is {mean} +/- {std}")


if __name__ == "__main__":
    main()

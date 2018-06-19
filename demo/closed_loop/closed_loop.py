import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import dolfin
import pulse

from problem import Problem
from force import ca_transient

# pulse.parameters['log_level'] = dolfin.WARNING

geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths['simple_ellipsoid'])
# Scale mesh to fit Windkessel parameters
geometry.mesh.coordinates()[:] *= 2.4

activation = dolfin.Constant(0.0)
matparams = pulse.Guccione.default_parameters()
material = pulse.Guccione(activation=activation,
                          parameters=matparams)


# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [pulse.RobinBC(value=dolfin.Constant(base_spring),
                          marker=geometry.markers["BASE"][0])]


# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(V.sub(0),
                            dolfin.Constant(0.0),
                            geometry.ffun, geometry.markers["BASE"][0])
    return bc


dirichlet_bc = [fix_basal_plane]


# Collect boundary conditions
bcs = pulse.BoundaryConditions(dirichlet=dirichlet_bc,
                               robin=robin_bc)

# Create the problem
problem = Problem(geometry, material, bcs)
problem.solve()

# Unloaded volume
pressures = [0.0]
volumes = [geometry.cavity_volume()]

# Passive inflation to end-diastole
ED_vol = 53.0
vol = problem.volume
pulse.iterate.iterate(problem, vol, ED_vol)


u, p, pinn = problem.state.split(deepcopy=True)
vol = geometry.cavity_volume(u=u)
problem.volume = vol
print("Inlation phase done")
print("Pressure = {:.3f}, Volume = {:2f}".format(float(pinn) * 1000, vol))
pressures.append(float(pinn) * 1000)
volumes.append(vol)

# plt.ion()


def update_plot(volumes, pressures):
    # ax.clear()
    fig, ax = plt.subplots()
    ax.set_ylabel("Pressure (Pa)")
    ax.set_xlabel("Volume (ml)")
    ax.plot(volumes, pressures)
    # fig.canvas.draw()
    plt.show()
    plt.close()


t = 0.0
cycle_lenght = 200 #ms
dt = 3


# Isovolumic contraction
# pulse.iterate.iterate(problem, activation, 0.2)

# u, p, pinn = problem.state.split(deepcopy=True)
# vol = geometry.cavity_volume(u=u)
# print("Isovolumic relaxation phase done")
# print("Pressure = {:.3f}, Volume = {:2f}".format(float(pinn), vol))


# Ejection (Windkessel model)

# Aorta compliance (reduce)
Cao = 10.0 / 1000.0
# Venous compliace
Cven = 400.0 / 1000.0
# Dead volume
Vart0 = 510
Vven0 = 2800
# Aortic resistance
Rao = 10 * 1000.0
Rven = 2.0 * 1000.0
# Peripheral resistance (increase)
Rper = 10 * 1000.0

V_ven = 3660
V_art = 640


while t < cycle_lenght:

    u, p, pinn = problem.state.split(deepcopy=True)

    V_cav = geometry.cavity_volume(u=u)
    PLV = float(pinn) * 1000

    # Change dt for final iteration
    if t + dt > cycle_lenght:
        dt = cycle_lenght - t

    Part = 1.0 / Cao * (V_art - Vart0)
    Pven = 1.0 / Cven * (V_ven - Vven0)

    print(("\n\nTime = {:.2f}\n"
           "LV pressure = {:.2f}\n"
           "Arterial Pressure = {:.2f}\n"
           "LV volum = {:.2f}\n").format(t, PLV, Part, V_cav))

    # Increment time
    t = t + dt

    # Flux trough aortic valve
    if(PLV <= Part):
        Qao = 0.0
    else:
        Qao = (1.0 / Rao) * (PLV - Part)

    # Flux trough mitral valve
    if(PLV >= Pven):
        Qmv = 0.0
    else:
        Qmv = (1.0 / Rven) * (Pven - PLV)

    Qper = 1.0 / Rper*(Part - Pven)

    V_cav = V_cav + dt * (Qmv - Qao)
    V_art = V_art + dt * (Qao - Qper)
    V_ven = V_ven + dt * (Qper - Qmv)

    # Update cavity volume
    problem.volume = V_cav
    pulse.iterate.iterate(problem, activation, ca_transient(t))

    pressures.append(PLV)
    volumes.append(V_cav)

    update_plot(volumes, pressures)

    # # Adapt time step
    # if len(states) == 1:
    #     dt *= 1.7
    # else:
    #     dt *= 0.5
          
    # dt = min(dt, 10)


from IPython import embed; embed()
exit()

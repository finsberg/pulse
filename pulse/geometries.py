import math
import sys
import typing
from pathlib import Path
from textwrap import dedent

import dolfin
from dijitso.signatures import hashit

try:
    import meshio

    has_meshio = True
except ImportError:
    has_meshio = False

try:
    import gmsh

    has_gmsh = True
except ImportError:
    has_gmsh = False

try:
    import ldrb

    has_ldrb = True
except ImportError:
    has_ldrb = False

try:
    from dolfin_adjoint import Mesh, interpolate, project
except ImportError:
    from dolfin import Mesh, interpolate, project

from .geometry import MarkerFunctions, HeartGeometry, Microstructure

cachedir = Path.home().joinpath(".cache").joinpath("pulse")
cachedir.mkdir(exist_ok=True, parents=True)


def create_mesh(mesh, cell_type, prune_z=True):
    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


def read_meshfunction(fname, obj):
    with dolfin.XDMFFile(Path(fname).as_posix()) as f:
        f.read(obj, "name_to_read")


def gmsh2dolfin(msh_file):

    msh = meshio.gmsh.read(msh_file)

    vertex_mesh = create_mesh(msh, "vertex")
    line_mesh = create_mesh(msh, "line")
    triangle_mesh = create_mesh(msh, "triangle")
    tetra_mesh = create_mesh(msh, "tetra")

    vertex_mesh_name = Path("vertex_mesh.xdmf")
    meshio.write(vertex_mesh_name, vertex_mesh)

    line_mesh_name = Path("line_mesh.xdmf")
    meshio.write(line_mesh_name, line_mesh)

    triangle_mesh_name = Path("triangle_mesh.xdmf")
    meshio.write(triangle_mesh_name, triangle_mesh)

    tetra_mesh_name = Path("mesh.xdmf")
    meshio.write(
        tetra_mesh_name,
        tetra_mesh,
    )

    mesh = Mesh()

    with dolfin.XDMFFile(tetra_mesh_name.as_posix()) as infile:
        infile.read(mesh)

    cfun = dolfin.MeshFunction("size_t", mesh, 3)
    read_meshfunction(tetra_mesh_name, cfun)
    tetra_mesh_name.unlink()
    tetra_mesh_name.with_suffix(".h5").unlink()

    ffun_val = dolfin.MeshValueCollection("size_t", mesh, 2)
    read_meshfunction(triangle_mesh_name, ffun_val)
    ffun = dolfin.MeshFunction("size_t", mesh, ffun_val)
    for value in ffun_val.values():
        mesh.domains().set_marker(value, 2)
    ffun.array()[ffun.array() == max(ffun.array())] = 0
    triangle_mesh_name.unlink()
    triangle_mesh_name.with_suffix(".h5").unlink()

    efun_val = dolfin.MeshValueCollection("size_t", mesh, 1)
    read_meshfunction(line_mesh_name, efun_val)
    efun = dolfin.MeshFunction("size_t", mesh, efun_val)
    efun.array()[efun.array() == max(efun.array())] = 0
    line_mesh_name.unlink()
    line_mesh_name.with_suffix(".h5").unlink()

    vfun_val = dolfin.MeshValueCollection("size_t", mesh, 0)
    read_meshfunction(vertex_mesh_name, vfun_val)
    vfun = dolfin.MeshFunction("size_t", mesh, vfun_val)
    vfun.array()[vfun.array() == max(vfun.array())] = 0
    vertex_mesh_name.unlink()
    vertex_mesh_name.with_suffix(".h5").unlink()

    markers = msh.field_data
    marker_functions = MarkerFunctions(vfun=vfun, efun=efun, ffun=ffun, cfun=cfun)

    geo = HeartGeometry(
        mesh=mesh,
        markers=markers,
        marker_functions=marker_functions,
    )
    return geo


def check_gmsh():
    check_meshio()
    if not has_gmsh:
        raise ImportError(
            (
                "Cannot create mesh using gmsh. "
                "Please install gmsh first with 'python -m pip install gmsh'"
            ),
        )


def check_meshio():
    if not has_meshio:
        raise ImportError(
            (
                "Cannot create mesh using gmsh. "
                "Please install meshio first with 'python -m pip install meshio'"
            ),
        )


def check_ldrb():
    if not has_ldrb:
        raise ImportError(
            (
                "Cannot create microstructure using ldrb. "
                "Please install ldrb first with 'python -m pip install ldrb'"
            ),
        )


def create_benchmark_ellipsoid_mesh_gmsh(
    mesh_name,
    r_short_endo=7.0,
    r_short_epi=10.0,
    r_long_endo=17.0,
    r_long_epi=20.0,
    quota_base=-5.0,
    psize=3.0,
    ndiv=1.0,
    mesh_size_factor=1.0,
):
    check_gmsh()
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    # gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)

    mu_base = math.acos(quota_base / r_long_endo)
    psize_ref = psize / ndiv

    def ellipsoid_point(mu, theta, r_long, r_short, psize):
        return gmsh.model.geo.addPoint(
            r_long * math.cos(mu),
            r_short * math.sin(mu) * math.cos(theta),
            r_short * math.sin(mu) * math.sin(theta),
            psize,
        )

    center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)

    apex_endo = ellipsoid_point(
        mu=0.0,
        theta=0.0,
        r_short=r_short_endo,
        r_long=r_long_endo,
        psize=psize_ref / 2.0,
    )

    base_endo = ellipsoid_point(
        mu=mu_base,
        theta=0.0,
        r_short=r_short_endo,
        r_long=r_long_endo,
        psize=psize_ref,
    )

    apex_epi = ellipsoid_point(
        mu=0.0,
        theta=0.0,
        r_short=r_short_epi,
        r_long=r_long_epi,
        psize=psize_ref / 2.0,
    )

    base_epi = ellipsoid_point(
        mu=math.acos(r_long_endo / r_long_epi * math.cos(mu_base)),
        theta=0.0,
        r_short=r_short_epi,
        r_long=r_long_epi,
        psize=psize_ref,
    )

    apex = gmsh.model.geo.addLine(apex_endo, apex_epi)
    base = gmsh.model.geo.addLine(base_endo, base_epi)
    endo = gmsh.model.geo.add_ellipse_arc(apex_endo, center, apex_endo, base_endo)
    epi = gmsh.model.geo.add_ellipse_arc(apex_epi, center, apex_epi, base_epi)

    ll1 = gmsh.model.geo.addCurveLoop([apex, epi, -base, -endo])

    s1 = gmsh.model.geo.addPlaneSurface([ll1])

    sendoringlist = []
    sepiringlist = []
    sendolist = []
    sepilist = []
    sbaselist = []
    vlist = []

    out = [(2, s1)]
    for _ in range(4):
        out = gmsh.model.geo.revolve(
            [out[0]],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            math.pi / 2,
        )

        sendolist.append(out[4][1])
        sepilist.append(out[2][1])
        sbaselist.append(out[3][1])
        vlist.append(out[1][1])

        gmsh.model.geo.synchronize()
        bnd = gmsh.model.getBoundary([out[0]])

        sendoringlist.append(bnd[1][1])
        sepiringlist.append(bnd[3][1])

    phys_apex_endo = gmsh.model.addPhysicalGroup(0, [apex_endo])
    gmsh.model.setPhysicalName(0, phys_apex_endo, "APEX_ENDO")

    phys_apex_epi = gmsh.model.addPhysicalGroup(0, [apex_epi])
    gmsh.model.setPhysicalName(0, phys_apex_epi, "APEX_EPI")

    phys_epiring = gmsh.model.addPhysicalGroup(1, sepiringlist)
    gmsh.model.setPhysicalName(1, phys_epiring, "EPIRING")

    phys_endoring = gmsh.model.addPhysicalGroup(1, sendoringlist)
    gmsh.model.setPhysicalName(1, phys_endoring, "ENDORING")

    phys_base = gmsh.model.addPhysicalGroup(2, sbaselist)
    gmsh.model.setPhysicalName(2, phys_base, "BASE")

    phys_endo = gmsh.model.addPhysicalGroup(2, sendolist)
    gmsh.model.setPhysicalName(2, phys_endo, "ENDO")

    phys_epi = gmsh.model.addPhysicalGroup(2, sepilist)
    gmsh.model.setPhysicalName(2, phys_epi, "EPI")

    phys_myo = gmsh.model.addPhysicalGroup(3, vlist)
    gmsh.model.setPhysicalName(3, phys_myo, "MYOCARDIUM")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(Path(mesh_name).as_posix())

    gmsh.finalize()


class EllipsoidGeometry(HeartGeometry):
    """
    Truncated ellipsoidal geometry, defined through the coordinates:

    X1 = Rl(t) cos(mu)
    X2 = Rs(t) sin(mu) cos(theta)
    X3 = Rs(t) sin(mu) sin(theta)

    for t in [0, 1], mu in [0, mu_base] and theta in [0, 2pi).
    """

    @classmethod
    def from_parameters(cls, params=None):
        params = params or {}
        parameters = cls.default_parameters()
        parameters.update(params)

        msh_name = Path("test.msh")
        create_benchmark_ellipsoid_mesh_gmsh(
            msh_name,
            r_short_endo=parameters["r_short_endo"],
            r_short_epi=parameters["r_short_epi"],
            r_long_endo=parameters["r_long_endo"],
            r_long_epi=parameters["r_long_epi"],
            quota_base=parameters["quota_base"],
            psize=parameters["mesh_generation"]["psize"],
            ndiv=parameters["mesh_generation"]["ndiv"],
        )

        geo: HeartGeometry = gmsh2dolfin(msh_name)
        msh_name.unlink()
        obj = cls(
            mesh=geo.mesh,
            marker_functions=geo.marker_functions,
            markers=geo.markers,
        )
        obj._parameters = parameters

        # microstructure
        mspace = obj._parameters["microstructure"]["function_space"]

        dolfin.info("Creating microstructure")
        # coordinate mapping

        cart2coords_code = obj._compile_cart2coords_code()
        cart2coords = dolfin.compile_cpp_code(cart2coords_code)
        cart2coords_expr = dolfin.CompiledExpression(
            cart2coords.SimpleEllipsoidCart2Coords(),
            degree=1,
        )

        # local coordinate base
        localbase_code = obj._compile_localbase_code()
        localbase = dolfin.compile_cpp_code(localbase_code)
        localbase_expr = dolfin.CompiledExpression(
            localbase.SimpleEllipsoidLocalCoords(cart2coords_expr.cpp_object()),
            degree=1,
        )

        # function space
        family, degree = mspace.split("_")
        degree = int(degree)
        V = dolfin.TensorFunctionSpace(obj.mesh, family, degree, shape=(3, 3))
        # microstructure expression
        alpha_endo = obj._parameters["microstructure"]["alpha_endo"]
        alpha_epi = obj._parameters["microstructure"]["alpha_epi"]

        microstructure_code = obj._compile_microstructure_code()
        microstructure = dolfin.compile_cpp_code(microstructure_code)
        microstructure_expr = dolfin.CompiledExpression(
            microstructure.EllipsoidMicrostructure(
                cart2coords_expr.cpp_object(),
                localbase_expr.cpp_object(),
                alpha_epi,
                alpha_endo,
            ),
            degree=1,
        )

        # interpolation
        W = dolfin.VectorFunctionSpace(obj.mesh, family, degree)
        microinterp = interpolate(microstructure_expr, V)
        s0 = project(microinterp[0, :], W)
        n0 = project(microinterp[1, :], W)
        f0 = project(microinterp[2, :], W)
        obj.microstructure = Microstructure(f0=f0, s0=s0, n0=n0)
        obj.update_xshift()

        return obj

    @staticmethod
    def default_parameters():
        p = {
            "mesh_generation": {
                "psize": 3.0,
                "ndiv": 1,
            },
            "microstructure": {
                "function_space": "DG_1",
                "alpha_endo": +90.0,
                "alpha_epi": -90.0,
            },
            "r_short_endo": 7.0,
            "r_short_epi": 10.0,
            "r_long_endo": 17.0,
            "r_long_epi": 20.0,
            "quota_base": -5.0,
        }
        return p

    def _compile_cart2coords_code(self):
        code = dedent(
            f"""\
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        namespace py = pybind11;
        #include <boost/math/tools/roots.hpp>
        #include <dolfin/function/Expression.h>
        #include <dolfin/function/Function.h>
        using boost::math::tools::newton_raphson_iterate;

        class SimpleEllipsoidCart2Coords : public dolfin::Expression
        {{
        public :

            std::shared_ptr<dolfin::Function> coords;

            SimpleEllipsoidCart2Coords() : dolfin::Expression(3)
            {{}}

            void eval(dolfin::Array<double>& values,
                      const dolfin::Array<double>& raw_x,
                      const ufc::cell& cell) const
            {{
                // coordinate mapping
                const std::size_t value_size = 3;
                dolfin::Array<double> x_tmp(value_size);

                if (this->coords)
                    coords->eval(x_tmp, raw_x, cell);
                else
                    std::copy(raw_x.data(), raw_x.data() + value_size, x_tmp.data());

                dolfin::Array<double> x(3);
                x[0] = x_tmp[0];
                x[1] = x_tmp[1];
                x[2] = x_tmp[2];

                // constants
                const double r_short_endo = {self._parameters['r_short_endo']};
                const double r_short_epi  = {self._parameters['r_short_epi']};
                const double r_long_endo  = {self._parameters['r_long_endo']};
                const double r_long_epi   = {self._parameters['r_long_epi']};

                // to find the transmural position we have to solve a
                // 4th order equation. It is easier to apply bisection
                // in the interval of interest [0, 1]
                auto fun = [&](double t)
                {{
                    double rs = r_short_endo + (r_short_epi - r_short_endo) * t;
                    double rl = r_long_endo + (r_long_epi - r_long_endo) * t;
                    double a2 = x[1]*x[1] + x[2]*x[2];
                    double b2 = x[0]*x[0];
                    double rs2 = rs*rs;
                    double rl2 = rl*rl;
                    double drs = (r_short_epi - r_short_endo) * t;
                    double drl = (r_long_epi - r_long_endo) * t;

                    double f  = a2 * rl2 + b2 * rs2 - rs2 * rl2;
                    double df = 2.0 * (a2 * rl * drl + b2 * rs * drs
                                - rs * drs * rl2 - rs2 * rl * drl);

                    return boost::math::make_tuple(f, df);
                }};

                int digits = std::numeric_limits<double>::digits;
                double t = newton_raphson_iterate(fun, 0.5, -0.0001, 1.0, digits);
                values[0] = t;

                double r_short = r_short_endo * (1-t) + r_short_epi * t;
                double r_long  = r_long_endo  * (1-t) + r_long_epi  * t;

                double a = std::sqrt(x[1]*x[1] + x[2]*x[2]) / r_short;
                double b = x[0] / r_long;

                // mu
                values[1] = std::atan2(a, b);

                // theta
                values[2] = (values[1] < DOLFIN_EPS)
                          ? 0.0
                          : M_PI - std::atan2(x[2], -x[1]);
            }}
        }};


        PYBIND11_MODULE(SIGNATURE, m)
        {{
            py::class_<SimpleEllipsoidCart2Coords, std::shared_ptr<SimpleEllipsoidCart2Coords>, dolfin::Expression>
                (m, "SimpleEllipsoidCart2Coords")
                .def(py::init<>());

        }}
        """,
        )
        return code

    def _compile_localbase_code(self):
        code = dedent(
            f"""\
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        namespace py = pybind11;
        #include <boost/math/tools/roots.hpp>
        #include <dolfin/function/Expression.h>
        #include <dolfin/function/Function.h>

        #include <Eigen/Dense>

        class SimpleEllipsoidLocalCoords : public dolfin::Expression
        {{
        public :

            typedef Eigen::Vector3d vec_type;
            typedef Eigen::Matrix3d mat_type;
            std::shared_ptr<dolfin::Expression> cart2coords;

            SimpleEllipsoidLocalCoords(std::shared_ptr<dolfin::Expression> cart2coords)
                : dolfin::Expression(3, 3), cart2coords(cart2coords)
            {{}}

            void eval(dolfin::Array<double>& values,
                      const dolfin::Array<double>& raw_x,
                      const ufc::cell& cell) const
            {{
                // check if coordinates are ok
                assert(this->cart2coords);

                // first find (lambda, mu, theta) from (x0, x1, x2)
                // axisymmetric case has theta = 0
                dolfin::Array<double> coords(3);
                this->cart2coords->eval(coords, raw_x, cell);

                double t = coords[0];
                double mu = coords[1];
                double theta = coords[2];

                // (e_1, e_2, e_3) = G (e_t, e_mu, e_theta)
                const double r_short_endo = {self._parameters['r_short_endo']};
                const double r_short_epi  = {self._parameters['r_short_epi']};
                const double r_long_endo  = {self._parameters['r_long_endo']};
                const double r_long_epi   = {self._parameters['r_long_epi']};

                double rs = r_short_endo + (r_short_epi - r_short_endo) * t;
                double rl = r_long_endo + (r_long_epi - r_long_endo) * t;
                double drs = r_short_epi - r_short_endo;
                double drl = r_long_epi - r_long_endo;

                double sin_m = std::sin(mu);
                double cos_m = std::cos(mu);
                double sin_t = std::sin(theta);
                double cos_t = std::cos(theta);

                mat_type base;
                base << drl*cos_m,       -rl*sin_m,        0.0,
                        drs*sin_m*cos_t,  rs*cos_m*cos_t, -rs*sin_m*sin_t,
                        drs*sin_m*sin_t,  rs*cos_m*sin_t,  rs*sin_m*cos_t;
                if (mu < DOLFIN_EPS)
                {{
                    // apex, e_mu and e_theta not defined
                    // --> random, but orthonormal
                    base << 1, 0, 0,
                            0, 1, 0,
                            0, 0, 1;
                }}
                base = base.colwise().normalized();

                // in general this base is not orthonormal, unless
                //   d/dt ( rs^2(t) - rl^2(t) ) = 0
                bool enforce_orthonormal_base = true;
                if (enforce_orthonormal_base)
                {{
                    base.col(0) = base.col(1).cross(base.col(2));
                }}

                Eigen::Map<mat_type>(values.data()) = base;
            }}
        }};

        PYBIND11_MODULE(SIGNATURE, m)
        {{
            py::class_<SimpleEllipsoidLocalCoords, std::shared_ptr<SimpleEllipsoidLocalCoords>, dolfin::Expression>
                (m, "SimpleEllipsoidLocalCoords")
                .def(py::init<std::shared_ptr<dolfin::Expression> >())
                .def_readwrite("cart2coords", &SimpleEllipsoidLocalCoords::cart2coords);
        }}
        """,
        )

        return code

    def _compile_microstructure_code(self):
        """
        C++ code for analytic fiber and sheet.
        """
        code = dedent(
            """\
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        namespace py = pybind11;
        #include <boost/math/tools/roots.hpp>
        #include <dolfin/function/Expression.h>
        #include <dolfin/function/Function.h>
        #include <Eigen/Dense>

        class EllipsoidMicrostructure : public dolfin::Expression
        {
        public :

            typedef Eigen::Vector3d vec_type;
            typedef Eigen::Matrix3d mat_type;

            std::shared_ptr<dolfin::Expression> cart2coords;
            std::shared_ptr<dolfin::Expression> localbase;

            double alpha_epi, alpha_endo;

            EllipsoidMicrostructure(
                std::shared_ptr<dolfin::Expression> cart2coords,
                std::shared_ptr<dolfin::Expression> localbase,
                double alpha_epi, double alpha_endo
            ) : dolfin::Expression(3, 3), cart2coords(cart2coords), localbase(localbase), alpha_epi(alpha_epi), alpha_endo(0.0)
            {

            }

            void eval(dolfin::Array<double>& values,
                      const dolfin::Array<double>& raw_x,
                      const ufc::cell& cell) const
            {
                // check if coordinates are ok
                assert(this->localbase);
                assert(this->cart2coords);

                // first find (lambda, mu, theta) from (x0, x1, x2)
                dolfin::Array<double> coords(3);
                this->cart2coords->eval(coords, raw_x, cell);

                // then evaluate the local basis
                dolfin::Array<double> base(9);
                this->localbase->eval(base, raw_x, cell);

                // transmural position
                double pos = 0.0;
                pos = coords[0];

                // angles
                double alpha = (alpha_epi - alpha_endo) * pos + alpha_endo;
                alpha = alpha / 180.0 * M_PI;

                // Each column is a basis vector
                // --> [ e_lambda, e_mu, e_theta ]
                mat_type S = Eigen::Map<mat_type>(base.data());

                // Rotation around e_lambda of angle alpha
                Eigen::AngleAxisd rot1(alpha, S.col(0));
                S = rot1 * S;
                // --> [ n0, s0, f0 ]

                // Return the values
                Eigen::Map<mat_type>(values.data()) = S;
            }
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {{
            py::class_<EllipsoidMicrostructure, std::shared_ptr<EllipsoidMicrostructure>, dolfin::Expression>
                (m, "EllipsoidMicrostructure")
                .def(py::init<std::shared_ptr<dolfin::Expression>, std::shared_ptr<dolfin::Expression>, double, double >())
                .def_readwrite("cart2coords", &EllipsoidMicrostructure::cart2coords)
                .def_readwrite("localbase", &EllipsoidMicrostructure::localbase)
                .def_readwrite("alpha_epi", &EllipsoidMicrostructure::alpha_epi)
                .def_readwrite("alpha_endo", &EllipsoidMicrostructure::alpha_endo);
        }}
        """,
        )

        return code


def benchmark_ellipsoid_geometry(
    params: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    """Create an ellipsoidal geometry from the bechmark parper

    Parameters
    ----------
    params : typing.Dict[str, typing.Any], optional
        Parameters for generating the geometry,
        see `EllipsoidGeometry.default_parameters`

    Returns
    -------
    geometry.HeartGeometry
        The gemometry
    """
    params = params or {}
    signature = hashit(repr(params))
    path = cachedir.joinpath(f"benchmark_ellipsoid_geometry_{signature}.h5")

    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters(params)
        geo.save(path)
    return HeartGeometry.from_file(path)


def prolate_ellipsoid_geometry(
    r_short_endo: float = 7.0,
    r_short_epi: float = 10.0,
    r_long_endo: float = 17.0,
    r_long_epi: float = 20.0,
    quota_base: float = 0.0,
    psize: float = 3.0,
    ndiv: float = 1.0,
    mesh_size_factor: float = 1.0,
    fiber_params: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    fiber_params = fiber_params or {}
    fiber_params = fiber_params.copy()
    signature = hashit(
        repr(
            dict(
                r_short_endo=r_short_endo,
                r_short_epi=r_short_epi,
                r_long_endo=r_long_endo,
                r_long_epi=r_long_epi,
                quota_base=quota_base,
                psize=psize,
                ndiv=ndiv,
                mesh_size_factor=mesh_size_factor,
                fiber_params=fiber_params,
            ),
        ),
    )
    path = cachedir.joinpath(f"prolate_ellipsoid_geometry_{signature}.h5")

    if not path.is_file():
        check_ldrb()
        msh_name = Path("test.msh")
        create_benchmark_ellipsoid_mesh_gmsh(
            msh_name,
            r_short_endo=r_short_endo,
            r_short_epi=r_short_epi,
            r_long_endo=r_long_endo,
            r_long_epi=r_long_epi,
            quota_base=quota_base,
            psize=psize,
            ndiv=ndiv,
            mesh_size_factor=mesh_size_factor,
        )

        geo: HeartGeometry = gmsh2dolfin(msh_name)
        msh_name.unlink()
        ldrb_markers = {
            "base": geo.markers["BASE"][0],
            "epi": geo.markers["EPI"][0],
            "lv": geo.markers["ENDO"][0],
        }

        fiber_space = fiber_params.pop("fiber_space", "CG_1")

        fiber_sheet_system = ldrb.dolfin_ldrb(
            geo.mesh, fiber_space, geo.ffun, ldrb_markers, **fiber_params
        )
        geo.microstructure = Microstructure(
            f0=fiber_sheet_system.fiber,
            s0=fiber_sheet_system.sheet,
            n0=fiber_sheet_system.sheet_normal,
        )
        geo.save(path)
    return HeartGeometry.from_file(path)

#!/usr/bin/env python
from collections import namedtuple
import dolfin

from .geometry_utils import load_geometry_from_h5, move

from . import numpy_mpi
# from . import config
from .dolfin_utils import compute_meshvolume, map_vector_field, get_cavity_volume
from .utils import set_default_none, make_logger
logger = make_logger(__name__, 10)

Marker = namedtuple('Marker', ['name', 'value', 'dimension'])

# f0 - fibers, s0 - sheets, n0 - cross-sheets
Microstructure = namedtuple('Microstructure', ['f0', 's0', 'n0'])
# Set defaults none to allow for different types of anisotropy
set_default_none(Microstructure)

# l0 - longitudinal, c0 - circumferential, r0 - radial
CRLBasis = namedtuple('CRLBasis', ['c0', 'r0', 'l0'])
# These are only needed in the RegionalStrainTarget
set_default_none(CRLBasis)


# vfun - vertex function
# efun - edge function
# ffun - facet function
# cfun - cell function
MarkerFunctions = namedtuple('MarkerFunctions', ['vfun', 'efun',
                                                 'ffun', 'cfun'])
# Note the this might not allways make sense for meshes
# of dimension < 3, also some of these function might not
# be relevant, therefore set defaults to None
set_default_none(MarkerFunctions)


class Geometry(object):
    """
    Base class for geometry

    Geometry can be intanciated directly

    Example
    -------

        ..code-block: python

            # To be implemented

    You can load create and instace be loading a geometry from a file

    Example
    -------

        ..code-block: python

            # Geometry is stored in a file "geometry.h5"
            geo = Geometry.from_file("geometry.h5")

    """
    def __init__(self, mesh,
                 markers=None,
                 marker_functions=None,
                 microstructure=None,
                 crl_basis=None):

        self.mesh = mesh

        self.markers = markers or {}
        self.marker_functions = marker_functions or MarkerFunctions()
        self.microstructure = microstructure or Microstructure
        self.crl_basis = crl_basis or CRLBasis()

    def __repr__(self):
        args = []
        for attr in ('f0', 's0', 'n0', 'vfun',
                     'ffun', 'cfun', 'c0', 'r0', 'l0'):
            if getattr(self, attr) is not None:
                args.append(attr)

        return ('{self.__class__.__name__}'
                '({self.mesh}), '
                '{args})').format(self=self,
                                  args=", ".join(args))

    @classmethod
    def from_file(cls, h5name, h5group='', comm=None):

        comm = comm if comm is not None else dolfin.mpi_comm_world()

        return cls(**cls.load_from_file(h5name, h5group, comm))

    def copy(self, u=None):
        """
        Return a copy of this geometry and attach a moved
        mesh according to the provided displacement.
        If no displacement is provided this will just
        return a copy of the original geometry.
        """
        new_mesh = dolfin.Mesh(self.mesh)

        if u is not None:
            U = move(new_mesh, u, -1.0)
        else:
            U = u

        marker_functions_ = {}
        for dim, fun in ((0, 'vfun'), (1, 'efun'),
                         (2, 'ffun'), (3, 'cfun')):
            f_old = getattr(self, fun)
            if f_old is None:
                continue
            f = dolfin.MeshFunction("size_t", new_mesh,
                                    dim, new_mesh.domains())
            f.set_values(f_old.array())
            marker_functions_[fun] = f
        marker_functions = MarkerFunctions(**marker_functions_)

        microstructure_ = {}
        for field in ('f0', 's0', 'n0'):

            v0 = getattr(self, field)
            if v0 is None:
                continue
            v = map_vector_field(v0, new_mesh, U)

            microstructure_[field] = v
        microstructure = Microstructure(**microstructure_)

        crl_basis_ = {}
        for basis in ('c0', 'r0', 'l0'):

            v0 = getattr(self, basis)
            if v0 is None:
                continue
            v = map_vector_field(v0, new_mesh, U)
            crl_basis_[basis] = v
        crl_basis = CRLBasis(**crl_basis_)

        return Geometry(mesh=new_mesh,
                        markers=self.markers,
                        marker_functions=marker_functions,
                        microstructure=microstructure,
                        crl_basis=crl_basis)

    @staticmethod
    def load_from_file(h5name, h5group, comm):

        logger.debug("Load geometry from file {}".format(h5name))

        geo = load_geometry_from_h5(h5name, h5group,
                                    include_sheets=True, comm=comm)

        kwargs = {
            'mesh': geo.mesh,
            'markers': geo.markers,
            'marker_functions': MarkerFunctions(vfun=geo.vfun,
                                                ffun=geo.ffun,
                                                cfun=geo.cfun),
            'microstructure': Microstructure(f0=geo.f0,
                                             s0=geo.s0,
                                             n0=geo.n0),
            'crl_basis': CRLBasis(c0=geo.circumferential,
                                  r0=geo.radial,
                                  l0=geo.longitudinal)
        }

        return kwargs

    def save(self, h5name):
        pass

    def dim(self):
        return self.mesh.geometry.dim()

    def crl_basis_list(self):
        """Return a list of the CRL basis in the order
        c0, r0, l0. Basis elements that are none will not
        be included
        """
        if not hasattr(self, '_crl_basis_list'):
            self._crl_basis_list = []
            for l in ['c0', 'r0', 'l0']:

                e = getattr(self.crl_basis, l)
                if e is not None:
                    self._crl_basis_list.append(e)

        return self._crl_basis_list

    @property
    def nbasis(self):
        return len(self.crl_basis_list())

    @property
    def dx(self):
        """Return the volume measure using self.mesh as domain and
        self.cfun as subdomain_data
        """
        return dolfin.dx(domain=self.mesh,
                         subdomain_data=self.cfun)

    @property
    def ds(self):
        """Return the surface measure of exterior facets using
        self.mesh as domain and self.ffun as subdomain_data
        """
        return dolfin.ds(domain=self.mesh,
                         subdomain_data=self.ffun)

    @property
    def facet_normal(self):
        return dolfin.FacetNormal(self.mesh)

    @property
    def nregions(self):
        return len(self.regions)

    @property
    def regions(self):
        """Return a set of the unique values of the cell function (cfun)
        """
        if not hasattr(self, '_regions'):
            try:
                regions \
                    = set(numpy_mpi
                          .gather_broadcast(self.dx
                                            .subdomain_data().array())
                          .astype(int))
                self._regions = {int(i) for i in regions}
            except AttributeError:
                # self.dx.subdomain_data() is None
                logger.warning('No regions found')
                self._regions = {0}

        return self._regions

    @property
    def meshvols(self):
        """Return a list of the volume of each subdomain defined
        by the cell function (cfun)
        """
        if not hasattr(self, '_meshvols'):
            self._meshvols = [compute_meshvolume(dx=self.dx,
                                                 subdomain_id=int(i))
                              for i in self.regions]
        return self._meshvols

    @property
    def meshvol(self):
        """Return the volume of the whole mesh
        """
        if not hasattr(self, '_meshvol'):
            self._meshvol = compute_meshvolume(domain=self.mesh)

        return self._meshvol

    @property
    def vfun(self):
        """Vertex Function
        """
        return self.marker_functions.vfun

    @property
    def efun(self):
        """Edge Function
        """
        return self.marker_functions.efun

    @property
    def ffun(self):
        """Facet Function
        """
        return self.marker_functions.ffun

    @property
    def cfun(self):
        """Cell Function
        """
        return self.marker_functions.cfun

    @property
    def f0(self):
        """Fibers
        """
        return self.microstructure.f0

    @property
    def s0(self):
        """Sheets
        """
        return self.microstructure.s0

    @property
    def n0(self):
        """Cross-Sheets
        """
        return self.microstructure.n0

    @property
    def l0(self):
        """Longitudinal
        """
        return self.crl_basis.l0

    @property
    def c0(self):
        """Circumferential
        """
        return self.crl_basis.c0

    @property
    def r0(self):
        """Radial
        """
        return self.crl_basis.r0


class HeartGeometry(Geometry):
    def __init__(self, *args, **kwargs):
        super(HeartGeometry, self).__init__(*args, **kwargs)

    @property
    def is_biv(self):
        return 'ENDO_RV' in self.markers

    def cavity_volume(self, chamber="lv", u=None):

        return get_cavity_volume(self, chamber=chamber, u=u)
                 

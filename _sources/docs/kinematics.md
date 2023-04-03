# Kinematics
We represent the heart as a continuum body $\mathfrak{B}$ embedded in
$\mathbb{R}^3$. A configuration of $\mathfrak{B}$ is a mapping $\chi:
\mathfrak{B} \rightarrow \mathbb{R}^3$.
We denote the *reference configuration* of the heart by $\Omega
\equiv \chi_0(\mathfrak{B})$, and the *current configuration* by $\omega
\equiv \chi(\mathfrak{B})$. The mapping $\varphi :  \Omega
\rightarrow \omega$, given by the composition $\varphi = \chi
\circ \chi_0^{-1}$, is a smooth, orientation preserving (positive
determinant) and invertible map. We denote the coordinates in the
reference configuration by $\mathbf{X} \in \Omega$, and the coordinates in the current
configuration by $\mathbf{x} \in \omega$. The coordinates $\mathbf{X}$ and $\mathbf{x}$ are
commonly referred to as material and spatial points respectively, and
are related through the mapping $\varphi$, by $\mathbf{x} = \varphi(\mathbf{X})$.
For time-dependent problems it is common to make  the time-dependence
explicitly by writing $\mathbf{x} = \varphi(\mathbf{X}, t)$. In the following
we will only focus on the mapping between two configurations and
therefore no time-dependence is needed. The *deformation gradient* is a
rank-2 tensor, defined as the partial derivative of $\varphi$  with
respect to the material coordinates:

```{math}
:label: deformation_gradient
\begin{align}
  \mathbf{F} = \nabla_{\mathbf{X}} \varphi= \nabla \mathbf{x}.
\end{align}
```

Here we also introduce the notation $\nabla$, which means derivative
with respect to reference coordinates.
The deformation gradient maps vectors in the reference configuration to
vectors in the current configuration, and belongs to the space of
linear transformations from $\mathbb{R}^3$ to $\mathbb{R}^3$ with
strictly positive determinant, which we denote by
$\mathrm{Lin}^+$. Another important quantity is the
*displacement* field

```{math}
:label: displacement
\begin{align}
  \mathbf{u} = \mathbf{x}-\mathbf{X},
\end{align}
```
which relates positions in the reference configuration to positions
in the current configuration. From {eq}`deformation_gradient` we
see that

```{math}
\begin{align}
  \mathbf{F} = \nabla \mathbf{x} = \nabla \mathbf{u} + \nabla \mathbf{X} = \nabla \mathbf{u} + \mathbf{I}.
\end{align}
```

Some other useful quantities are the *right Cauchy-Green* deformation
tensor $\mathbf{C} = \mathbf{F}^T\mathbf{F}$, the *left Cauchy-Green* deformation tensor
$\mathbf{B} = \mathbf{F}\mathbf{F}^T$, the *Green-Lagrange* strain tensor
$\mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I})$, and the determinant of the
deformation gradient $J = \det \mathbf{F}$.

An important concept in mechanics is the concept of stress, which is
defined as force per area
$\left[\frac{\mathrm{N}}{\mathrm{m}^2}\right]$. When working with
different configurations one needs to be careful with which forces and
which areas we are talking about. {numref}`stress_tensor`
shows how forces and areas are related for the most important stress
tensors used in this thesis. Note that the explicit form of the stress
tensor requires a constitutive law for the material at hand. This will
be discussed in more detail in {ref}`section:constitutive_relations`.


```{list-table} Showing different stress tensors used in this thesis, and how they relate forces to areas trough different configurations.
:header-rows: 1
:name: stress_tensor

* - Stress tensor
  - Forces
  - Area
* - Second Piola-Kirchhoff ($\mathbf{S}$)
  - Reference configuration
  - Reference configuration
* - First Piola-Kirchhoff ($\mathbf{P}$)
  - Current configuration
  - Reference configuration
* - Cauchy ($\sigma$)
  - Current configuration
  - Current configuration
```


## Code

In pulse there is a separate module for kinematics that contains the most relevant quantities, e.g

```python
import dolfin
import pulse

# Some mesh
mesh = dolfin.UnitCubeMesh(3, 3, 3)

# Space for displacement
V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
u = dolfin.Function(V)

F = pulse.kinematics.DeformationGradient(u)
E = pulse.kinematics.GreenLagrangeStrain(F)
```

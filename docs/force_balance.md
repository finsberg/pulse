# Force-balance equation

We will now collect all the terms that are involved in the force balance for
the cardiac mechanics problem. Considering the myocardium as an incompressible,
hyperelastic material we obtain the following strong form in the
Lagrangian formulation

```{math}
:label: force_balance_strong
\begin{align}
  \begin{split}
  \nabla \cdot \mathbf{P} &= 0 \\
  J - 1 &= 0,
  \end{split}
\end{align}
```
completed with appropriate boundary conditions. To solve this
numerically using the finite element method, we need to derive the weak
variational form of this equation.


## Variational formulation

There are many ways to arrive at the variational formulation of the
force-balance equations for the cardiac mechanics problem.  One way is
to consider the strong form in {eq}`force_balance_strong` and use
the standard approach in the finite element method to multiply by
test function in a suitable space, and perform integration by
parts. Within the fields of continuum mechanics it is common to
refer to this approach as the *principle of virtual work*, which states
that the virtual work of all forces applied to a mechanical system
vanishes in equilibrium. Within this framework, test functions are
referred to as virtual variations.
Another approach, which we will use here, derives the variational form
by utilizing a fundamental principle in physics
called the *principle of stationary potential energy*, or
*minimum total potential energy principle*. This principle states that a
physical system is at equilibrium when the total potential energy is
minimized, and any infinitesimal changes from this state should not add
any energy to the system.
In order to make use of this principle we first need to sum up all the
potential energy in the system. Here we separate between internal and
external energy. Internal energy is energy that is stored within the
material, for instance when you stretch a rubber band you increase its
internal energy. External energy represent the contribution from all
external forces such as gravity and traction forces.

For an incompressible, hyperelastic material the total potential
energy in the system is given by
```{math}
\begin{align}
  \Pi(\mathbf{u}, p) &= \Pi_{\mathrm{int}}(\mathbf{u},p) + \Pi_{\mathrm{ext}}(\mathbf{u}). \\
  \Pi_{\mathrm{int}}(\mathbf{u},p) &= \int_{\Omega} \left[ p(J - 1) +  \Psi(\mathbf{F}) \right] \mathrm{d}V\\
  \Pi_{\mathrm{ext}}(\mathbf{u}) &= - \int_{\Omega} \mathbf{B} \cdot \mathbf{u} \mathrm{d} V - \int_{\partial \Omega_N} \mathbf{T} \cdot \mathbf{u} \mathrm{d}S
\end{align}
```
Here $\mathbf{B}$ represents body forces acting on a volume element in
the reference domain, e.g  gravity, and $\mathbf{T} = \mathbf{P}
\mathbf{N}$ represents first Piola-Kirchhoff traction force acting on
the Neumann boundary $\partial \Omega_N$. According to the
*Principle of stationary potential energy* we have
```{math}
:label: minimum_potential_energy
\begin{align}
  D_{\delta \mathbf{u}} \Pi(\mathbf{u}, p) = 0,  && \text{and} && D_{\delta p} \Pi(\mathbf{u}, p) = 0.
\end{align}
```
Here $\delta \mathbf{u}$ and $\delta p$ are virtual variations in the
space for the displacement and hydrostatic pressure respectively, and
```{math}
\begin{align}
  D_{\mathbf{v}} \Phi(\mathbf{x}) = \frac{\mathrm{d}}{\mathrm{d}\varepsilon} \Phi(\mathbf{x} + \varepsilon \mathbf{v})\big|_{\varepsilon = 0}
\end{align}
```
is the directional derivative of $\Phi$ at $\mathbf{x}$ is the
direction $\mathbf{v}$. This operator is also known as the G\^ateaux
operator. The virtual variations $\delta \mathbf{u}$ and $\delta p$
represents an arbitrary direction with infinitesimal magnitude. We have
```{math}
\begin{align}
  0 = D_{\delta p} \Pi(\mathbf{u}, p)
  = \int_{\Omega}  \delta p(J(\mathbf{u}) - 1) \mathrm{d}V,
\end{align}
```
and
\begin{align*}
  \begin{split}
  0 = D_{\delta \mathbf{u}} \Pi(\mathbf{u}, p)
  =&  \int_{\Omega}  \left[ pJ \mathbf{F}^{-T} + \mathbf{P} \right] : \nabla \delta \mathbf{u} \mathrm{d}V - \int_{\Omega} \mathbf{B} \cdot \delta \mathbf{u} \mathrm{d} V
  \end{split}
\end{align*}
Note that the traction forces are now incorporated into the stress
tensors after application of the divergence theorem. These equations
are also commonly referred to as the Euler-Lagrange equations. Here $\mathbf{u}  \in V =
\left[H_D^1(\Omega)\right]^3$, with $H_D^1(\Omega) = \{ \mathbf{v}:
\int_{\Omega} \left[ |\mathbf{v}|^2 +  |\nabla \mathbf{v}|^2
\right]\mathrm{dV} < \infty \wedge \mathbf{v}\big|_{\partial \Omega_D}
= 0\}$ and $p \in Q = L^2(\Omega)$, with $\partial \Omega_D$
representing the Dirichlet boundary. In summary, the Euler-Lagrange
equations written in a mixed form reads : *Find $(\mathbf{u}, p)\in V
  \times Q$ such that*
```{math}
:label: intro_variational_form
  \begin{align}
  \begin{pmatrix}
    D_{\delta p} \Pi(\mathbf{u}, p)\\
    D_{\delta \mathbf{u}} \Pi(\mathbf{u}, p)
  \end{pmatrix}
  = \mathbf{0}.  && \forall \; (\delta \mathbf{u}, p) \in V \times Q.
\end{align}
```


## Discretization of the force balance equations

Equation {eq}`force_balance_strong` is only possible to solve
analytically for very simplified problems. Therefore we need to employ
numerical methods to solve the problem. One such method
is the finite element method (FEM). When using the finite element method, we often refer to such
approximation as a Galerkin approximation. This is based on
approximating the solution by linear combinations of  basis functions in a finite dimensional
subspace of the true solution. If $V$ and $Q$ are two suitable Hilbert
spaces for the displacement $\mathbf{u}$ and the hydrostatic pressure $p$
respectively, we now select some finite dimensional subspaces $V_h
\subset V$ and $Q_h \subset Q$, which are spanned by a finite number
of basis functions.


For incompressible problems such as {eq}`force_balance_strong`, we cannot choose the
approximation spaces $V_h, Q_h$ at random. A known numerical issue
that arises for such saddle-point problems is *locking*, which can be
seen if the material do not deform even if forces are applied. The
problem is solved by requiring the finite element approximation to
satisfy the discrete inf-sup condition {cite}`le1982existence`. There
exist several mixed elements that satisfies this condition
{cite}`chapelle1993inf`. The elements used in this thesis are the
Taylor-Hood finite elements {cite}`taylor1973numerical`. Let the domain of
interest be denoted by $\Omega$, and let $\mathcal{T}_h$ be a
triangulation of that domain in the sense that $\bigcup_{T \in
  \mathcal{T}_h} T = \overline{\Omega}$. Let $\mathcal{P}_k
(T)$ be the linear space of all polynomials of degree $\leq k$ defined
on $T$. Then for $k \geq 2$, the Taylor-Hood finite element spaces are the spaces
```{math}
\begin{align}
  V_h = \{  \phi \in C(\Omega) \;  | \; \phi \big| T  \in \mathcal{P}_k (T) , T \in \mathcal{T}_h \},\\
  Q_h = \{  \phi \in C(\Omega) \;  | \; \phi \big| T  \in \mathcal{P}_{k-1} (T) , T \in \mathcal{T}_h \},
\end{align}
```
where $C(\Omega)$ denotes the space of continuous function on
$\Omega$. In this thesis we have exclusively used these elements with
$k = 2$.

```{note}
  The basis functions that span the Taylor-Hood
  finite element spaces are also known as the
  Lagrangian basis functions. These basis functions, of
  degree $n$, are polynomials of degree $n$ on each element, but only
  continuous at the nodes (i.e not continuously
  differentiable). Consequently, differentiating a function that is
  expressed as a linear combination of the Lagrangian basis functions,
  will not be continuous at the nodes, and therefore caution has to
  be made when evaluating features that depends on the derivative of
  such functions. Examples of such features are stress and
  strain with depends on the deformation gradient which again depends
  on the derivative of the displacement. One way to deal with this
  issue is to 1) use other types of elements that are continuously
  differentiable everywhere,  such a the cubic Hermite elements or 2) evaluate the features at the
  Gaussian quadrature points where there is no problem with continuity.
```

## References

```{bibliography}
:filter: docname in docnames
```

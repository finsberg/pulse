(section:hyperelasticity)=
# Hyperelasticity

Even though experimental studies have indicated visco-elastic behavior
of the myocardium {cite}`dokos2002shear, gultekin2016orthotropic` a
common assumption is to consider quasi-static behavior, meaning that
the inertial term in {eq}`chauch_momentum_eq` is negligible and
static equilibrium is achieved at all points in the cardiac cycle. Therefore
it is also possible to model the myocardium as a hyperelastic
material,which is a type of elastic material.
This means that we postulate the existence of
a strain-energy density function $\Psi:\mathrm{Lin}^+ \rightarrow
\mathbb{R}^+$, and that stress is given by the relation
```{math}
:label:
\begin{align}
\mathbf{P} = \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}}.
\end{align}
```
Since stress has unit Pa, we see that the strain-energy density
function is defined as energy per unit reference volume, and has units
$\frac{\text{Joule}}{m^3}$.
The strain-energy density function relates  the amount of
energy that is stored within the material in response to a given
strain. Hence, the stresses in a hyperelastic material with a given
strain-energy density function, depend only on the strain, and not the
path for which the material deforms. On the contrary, if the model had
been visco-elastic we would expect to see hysteresis in the
stress/strain curve, but this is not possible for a hyperelastic
material.

```{note}
  The second law of thermodynamics states that the total entropy
  production in a thermodynamic process can never be negative. Elastic
  materials define a special class of materials in which the entropy
  production is zero. Within this thermodynamic framework the
  strain-energy density function coincides (up to a constant) with the
  Helmholtz free energy density.
```

(section:strain_energy_req)=
## General requirements for the strain-energy density function

Some general requirements must hold for the strain-energy function.
First of all, we require that the reference state is stress free and
that the stored energy increases monotonically with the deformation.
Formally this can be stated simply as
```{math}
\begin{align*}
  \Psi(\mathbf{I}) = 0 \; && \text{and} &&\; \Psi(\mathbf{F}) \geq 0.
\end{align*}
```
Moreover, expanding or compressing a body to zero volume would
require an infinite amount of energy, i.e
```{math}
\begin{align*}
  \Psi(\mathbf{F}) \rightarrow \infty \; && \text{as} &&\; \det \mathbf{F} \rightarrow& 0 \\
  \Psi(\mathbf{F}) \rightarrow \infty \; && \text{as} &&\; \det \mathbf{F} \rightarrow& \infty
\end{align*}
```
We say that the strain energy should be objective, meaning that the
stored energy in the material should be invariant with respect to
change of observer. Formally we must have: \emph{given any positive symmetric
rank-2 tensor $\mathbf{C} \in \mathrm{Sym}$:}
```{math}
\begin{align}
  \Psi(\mathbf{C}) = \Psi(\mathbf{Q}\mathbf{C}\mathbf{Q}^T), \; \forall \mathbf{Q} \in \mathcal{G} \subseteq \mathrm{Orth}.
\end{align}
```
Here $\mathrm{Orth}$ is the group of all positive orthogonal matrices.
If $\mathcal{G} = \mathrm{Orth}$ we say that the material is
isotropic, and otherwise we say that the material is anisotropic.
This brings us to another important issue, which is related to the
choice of coordinate-system. Having to deal with different
coordinate-systems, and mapping quantities from one coordinate-system
to another can results in complicated computations. Therefore it would be beneficial if we
could work with quantities which do not depend on the choice of
coordinate-system. Such quantities are called invariants.
If the material is isotropic, the representation theorem for
invariants {cite}`wang1970new` states that $\Psi$ can be expressed in terms of the
principle invariants of $\mathbf{C}$, that is $\Psi = \Psi(I_1, I_2,
I_3)$. The principle invariants $I_i, i=1,2,3$ are the coefficients in
the characteristic polynomial of $\mathbf{C}$, and is given by
```{math}
:label:
\begin{align}
  I_1 = \mathrm{tr} \; \mathbf{C},  && I_2 = \frac{1}{2}\left[ I_1^2 - \mathrm{tr} \;(\mathbf{C}^2)\right] && \text{and} && I_3 = \det \mathbf{C}.
\end{align}
```
In the case when the material constitutes a transversely isotropic
behavior, that is, the material has a preferred direction $\mathbf{a}_0$,
which in the case of the myocardium could be the direction of fiber
muscle fibers, we have
```{math}
\begin{align*}
  \mathcal{G} = \{ \mathbf{Q} \in \mathrm{Orth}: \mathbf{Q}(\mathbf{a}_0\otimes\mathbf{a}_0)\mathbf{Q}^T = \mathbf{a}_0\otimes\mathbf{a}_0 \},
\end{align*}
```
with $\otimes$ being the outer product. In this case the strain-energy
density function can still be expressed through invariants. However,
we need to include the so called quasi-invariants, which are defined
as stretches in the local microstructural coordinate-system. The
transversely isotropic invariants are given by
```{math}
\begin{align*}
  I_{4\mathbf{a}_0 } = \mathbf{a}_0 \cdot (\mathbf{C} \mathbf{a}_0) && \text{and} && I_5 = \mathbf{a}_0 \cdot (\mathbf{C}^2 \mathbf{a}_0).
\end{align*}
```
Some of the invariants do have a physical interpretation. For instance, $I_3$
is related to the volume ratio of material during deformation, while
$I_{4\mathbf{a}_0 } $ is related to the stretch along the direction
$\mathbf{a}_0 $. Indeed the \emph{stretch} ratio in the direction
$\mathbf{a}_0$ is given by $\lambda_{\mathbf{a}_0} = | \mathbf{F} \mathbf{a}_0
|$ and we see that $I_{4\mathbf{a}_0 }  =  \mathbf{a}_0 \cdot (\mathbf{C}
\mathbf{a}_0) = \mathbf{F} \mathbf{a}_0 \cdot (\mathbf{F} \mathbf{a}_0) =
\lambda_{\mathbf{a}_0}^2$. For more details about invariants see e.g
{cite}`holzapfel2009constitutive,liu1982representations`.


The theory of global existence of unique solutions for elastic problems
was originally based convexity of the free energy function.
An energy function $\Psi: \mathrm{Lin}^+ \rightarrow
\mathbb{R}^+$ is strictly \emph{convex} if for each $\mathbf{F} \in
\mathrm{Lin}^+$ and $\mathbf{H} \neq \mathbf{0}$ with $\det (\mathbf{F} +
(1-\lambda)\mathbf{H}) > 0$, we have
```{math}
:label: strain_convex
\begin{align}
  \Psi(\lambda \mathbf{F} + (1-\lambda) \mathbf{H})
  < \lambda \Psi(\mathbf{F})
  + (1-\lambda) \Psi(\mathbf{F} + \mathbf{H}), && \lambda \in (0,1).
\end{align}
```
If the response $\mathbf{P}$ is differentiable, then condition
{eq}`strain_convex` is equivalent of saying that the response is
positive definite,
```{math}
:label: response_posdev

\begin{align}
  \mathbf{H} : \frac{\partial \mathbf{P}}{\partial \mathbf{F}} : \mathbf{H} > 0,
  && \mathbf{F} \in
     \mathrm{Lin}^+, \mathbf{H} \neq \mathbf{0}.
\end{align}
```

However, from a physical point of view this requirement is too strict
{cite}`ball1976convexity`. A slightly weaker requirement is the strong
ellipticity condition which states that {eq}`response_posdev`
should hold for any $\mathbf{H}$ of rank-one, and is analogous to
say that the strain energy function is rank-one convex.


## References

```{bibliography}
:filter: docname in docnames
```

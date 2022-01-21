(section:incompressibility)=
# Incompressibility

The myocardium contains small blood vessels that supply the
myocardial cells with oxygen. When the myocardium contracts, this
perfused blood is squeezed out, resulting in an overall loss of
2-4\% volume {cite}`yin1996compressibility`. A material that change its
volume in response to applied loads is referred to as compressible. When the
volume is preserved we say that the material is incompressible.
Since 2-4\% is very little, a common assumption in cardiac mechanical modeling,
which has also been made in the work conducted in this thesis, is to assume
the myocardium to be incompressible. The reason for this choice is
purely numerical.

For an incompressible material, only isochoric motions are
possible. This means that the volume of the material does not change during
any deformation, and hence we have the constraint

```{math}
:label: incompressible_cons

\begin{align}
  J = \det(\mathbf{F}) = 1.
\end{align}
```

The constraint {eq}`incompressible_cons` can be imposed by
considering the modified strain energy function

```{math}
:label: incomp_strain_energy

\begin{align}
  \Psi = \Psi(\mathbf{F}) + p(J-1),
\end{align}
```
where $p$ is a scalar which serves as a Lagrange multiplier, but which
can be identified as the hydrostatic pressure. If we differentiate
{eq}`incomp_strain_energy` with respect to $\mathbf{F}$ we get the First
Piola-Kirchhoff stress tensor for an incompressible material
```{math}
\begin{align}
  \mathbf{P} = \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}} + J p \mathbf{F}^{-T}.
\end{align}
```
Likewise the Cauchy stress tensor is given by
```{math}
:label: cauchy_incomp
\begin{align}
  \sigma = J^{-1} \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}}\mathbf{F}^{T} + p \mathbf{I}.
\end{align}
```


```{note}
  The sign of $p$ is determined by whether you add or subtract the term
  $ p(J-1)$ to the total strain energy in
  {eq}`incomp_strain_energy`. For all practical purposes, it
  does not matter if you add or subtract the term as long as you are
  consistent.
```


## Uncoupling of volumetric and isochoric response
The total strain energy function in {eq}`incomp_strain_energy`
can be written as a sum of isochoric and volumetric components. Let
```{math}
\begin{align}
  \mathbf{F} =  \mathbf{F}_{\mathrm{vol}} \mathbf{F}_{\mathrm{iso}},
\end{align}
```
then $ \mathbf{F}_{\mathrm{vol}} =
J^{1/3}\mathbf{I}$ and $\mathbf{F}_{\mathrm{iso}} = J^{-1/3}\mathbf{F}$. For
compressible materials (i.e with $J \neq 1$) it is important to consider
only deviatoric strains in the strain-energy density function, so that
$\Psi = \Psi_{\mathrm{iso}}(\mathbf{F}_{\mathrm{iso}}) +
\Psi_{\mathrm{vol}}(J)$. For incompressible material ($J = 1$), we
have $\mathbf{F}_{\mathrm{vol}} = \mathbf{I}$ so that such a decomposition seems
unnecessary. However, a similar decomposition has shown to be
numerically beneficial {cite}`weiss1996finite`. Note that, in this case, a similar
decoupling of the stress tensors has to be done.

In the kinematics module you can specify whether if want the isochoric version

```python
F_iso = pulse.kinematics.DeformationGradient(u, isochoric=True)
```

## References

```{bibliography}
:filter: docname in docnames
```

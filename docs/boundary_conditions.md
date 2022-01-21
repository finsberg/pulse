# Boundary Conditions

Choosing the correct boundary conditions for the model is essential,
and the choice should mimic what is observed in reality. To
physiologically constrain the ventricle in a correct way is difficult,
and different approaches has been proposed.
The boundary condition at the endocardium is typically modeled as a
Neumann boundary condition, representing the endocardial blood
pressure. For the left ventricle we have
```{math}
\begin{align}
  \sigma \mathbf{n} = -p_{\mathrm{lv}} \mathbf{n}, \;  \mathbf{x} \in  \partial \omega_{ \text{endo LV}},
\end{align}
```
and for the right ventricle, *lv* is substituted with *rv*.
This condition has a negative sign because the unit normal
$\mathbf{N}$ is pointing out of the domain, while the pressure is
acting into the domain.
Note that this condition is imposed on the current configuration, and
to utilize the Lagrangian formulation we can pull back this condition
to the reference configuration to obtain
```{math}
\begin{align}
  \mathbf{P}\mathbf{N} &= -p_{\mathrm{lv}} J\mathbf{F}^{-T} \cdot \mathbf{N}, \;  \mathbf{X} \in \partial \Omega_{\text{endo LV}}
\end{align}
```
Likewise, it is common to enforce a
Neumann boundary condition on the epicardium,
```{math}
\begin{align}
\mathbf{P}\mathbf{N}  &= -p_{\mathrm{epi}}  J\mathbf{F}^{-T} \cdot \mathbf{N}, \;  \mathbf{X} \in \partial \Omega_{\text{epi}}.
\end{align}
```
However, the pressure $p_{\mathrm{epi}}$ is often set to zero as a
simplification.

There exist a variety of boundary conditions at the base.
It is common to constrain the longitudinal motion of
base, even though it is observed in cardiac images that the apex tend
to be more fixed than the base. A recent study shows that taking into
account the base movement is important to capture the correct
geometrical shape {cite}`palit2016passive`. However, this has not been
done in the studies in this thesis.
Fixing the longitudinal motion at the base is enforced through a
Dirichlet boundary condition,
```{math}
\begin{align}
  u_1 = 0,  \;  \mathbf{X} \in \partial \Omega_{\text{base}},
\end{align}
```
where $u_1$ is the longitudinal component of the displacement $\mathbf{u} =
(u_1, u_2, u_3)$. To apply this type of condition, it is easiest if
the base is flat and located at a prescribed location, for example in
the $x= 0$ plane. Constraining the longitudinal motion of the base
alone is not enough since the ventricle is free to move in the basal
plane. In order to anchor the geometry it is possible to fix the
movement of the base in all directions
```{math}
\begin{align}
  \mathbf{u} = \mathbf{0},  \;  \mathbf{X} \in \partial \Omega_{\text{base}},
\end{align}
```
or fixing the endocardial or epicardial ring
```{math}
\begin{align}
  \mathbf{u} &= \mathbf{0},  \;  \mathbf{X} \in \Gamma_{\mathrm{endo}} \\
  \mathbf{u} &= \mathbf{0},  \;  \mathbf{X} \in \Gamma_{\mathrm{epi}}.
\end{align}
```

Another approach which is used in this thesis is to impose a Robin
type boundary condition at the base
```{math}
\begin{align}
  \mathbf{P} \mathbf{N} + k \mathbf{u} = \mathbf{0},  \;  \mathbf{X} \in \partial \Omega_{\text{base}},
\end{align}
```
or at the epicardium to mimic the pericardium
```{math}
\begin{align}
  \mathbf{P} \mathbf{N} + k \mathbf{u} = \mathbf{0},  \;  \mathbf{X} \in \partial \Omega_{\text{epi}}.
\end{align}
```
Here $k$ can be seen as the stiffness of a spring that limits the
movement. The limiting cases, $k = 0$ and $k \rightarrow
\infty$ represent free and fixed boundary respectively.
More complex boundary conditions to mimic the pericardium are also
possible {cite}`fritz2014simulation`, but not considered in this thesis.
An overview of the location of the different boundaries for the
bi-ventricular geometry is illustrated in {numref}`boundaries`.


```{figure} figures/boundaries.png
---
name: boundaries
---
Illustration of the different boundaries in a bi-ventricular domain.
```

## References

```{bibliography}
:filter: docname in docnames
```

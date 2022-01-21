# Balance laws and transformations
In this section we will cover some basic transformations used to
derive the fore-balance equations for the mechanics of the heart.

## Transformations between reference and current configuration
By definition, the reference configuration $\Omega$, and current
configuration $\omega$, are related via the deformation map $\varphi$ in the
sense that a point $\mathfrak{p} \in \mathfrak{B}$ with reference
coordinates $\mathbf{X}$ and current coordinates $\mathbf{x}$ satisfies $\mathbf{x} =
\varphi(\mathbf{X})$. Likewise a vector in the reference configuration is
related to a vector in the current configuration  via the
deformation gradient $\mathbf{F}$; if $\mathrm{d}\mathbf{X}$ is a vector in the
reference configuration it will transform to the vector
$\mathrm{d}\mathbf{x}$ in the current configuration, and $\mathrm{d}\mathbf{x} =
\mathbf{F} \mathrm{d}\mathbf{X}$. From this relation we also derive that the
transformation of an infinitesimal volume element in the reference
configuration, $\mathrm{d}V$ is related to an infinitesimal volume
element in the current configuration, $\mathrm{d}v$  via the determinant of the
deformation gradient,
```{math}
:label: volume_element

\begin{align}
  \mathrm{d}v =\mathrm{det} \; (\mathbf{F}) \mathrm{d}V.
\end{align}
```

Another important transformation is the transformation of normal
vectors. By noting that we can write {eq}`volume_element` using
surface elements
```{math}
\begin{align*}
  \mathrm{d}s \mathbf{n} \mathrm{d}\mathbf{x}  &= \mathrm{d}v = \mathrm{det} \; (\mathbf{F}) \mathrm{d}V = \mathrm{det} \; (\mathbf{F}) \mathrm{d}S  \mathbf{N} \mathrm{d}\mathbf{X}\\
  &\implies \left( \mathrm{d}s \mathbf{n} \mathbf{F}  - \mathrm{d}S \mathrm{det} \; (\mathbf{F}) \mathbf{N} \right) \mathrm{d}\mathbf{X} = 0\\
  &\implies \left( \mathrm{d}s \mathbf{F}^T \mathbf{n}  - \mathrm{d}S \mathrm{det} \; (\mathbf{F}) \mathbf{N} \right) \mathrm{d}\mathbf{X} = 0,\\
\end{align*}
```
we get *Nanson's formula*
```{math}
:label:
\begin{align}
  \mathrm{d}s \mathbf{n}  =  \mathrm{det} \; (\mathbf{F}) \mathbf{F}^{-T} \mathrm{d}S \mathbf{N},
\end{align}
```

which relates the normal vector in the current configuration to the
normal vector in the reference configuration.


## Conservation of linear momentum
Newton's seconds law states that the change in linear momentum equals
the net impulse acting on it. For a continuum material with constant
mass density $\rho$ this implies that
```{math}
:label: cons_lin_mom
\begin{align}
  \int_{\omega} \rho \dot{\mathbf{v}} \mathrm{d}v = \mathbf{f},
  && \mathbf{f} = \int_{\partial \omega} \mathbf{t} \mathrm{d}s
     + \int_{\omega} \mathbf{b} \mathrm{d}v,
\end{align}
```

where $\mathbf{v}$ is the spatial velocity field, $\mathbf{t}$ is
the traction acting on the boundary, and $\mathbf{b}$ is the body
force. From \emph{Cauchy's stress theorem} we know that there exists a
second order tensor $\sigma$, known as the Cauchy stress tensor that is
related to the traction vector by $\mathbf{t} = \sigma \mathbf{n}$,
where $\mathbf{n}$ is the unit normal in the current configuration.
Using the divergence theorem we get
```{math}
\begin{align*}
  \int_{\partial \omega} \mathbf{t} \mathrm{d}s
  = \int_{\partial \omega} \sigma \mathbf{n} \mathrm{d}s
  = \int_{\omega} \nabla \cdot \sigma \mathrm{d}v,
\end{align*}
```
and by collecting the terms from {eq}`cons_lin_mom` we arrive at
Cauchy's momentum equation
```{math}
:label: chauch_momentum_eq
\begin{align}
  \nabla \cdot \sigma + \mathbf{b} =  \rho \dot{\mathbf{v}}.
\end{align}
```

The contribution from the body force ($\mathbf{b}$)  and inertial term
($\rho \dot{\mathbf{v}}$) can be considered negligible compared to the stresses
{cite}`costa1996three, tallarida1970left, moskowitz1981effects`, which is
why the force balance equations is typically only stated as
```{math}
:label:
\begin{align}
  \nabla \cdot \sigma = \mathbf{0}.
  \label{momentum_simple_current}
\end{align}
```

Note that we have formulated the balance law in the current
configuration. An equivalent statement can be formulated in terms of
the reference configuration
```{math}
:label:
\begin{align}
  \nabla \cdot \mathbf{P} = \mathbf{0},
  \label{momentum_simple_reference}
\end{align}
```

where $\mathbf{P}$ is the first Piola-Kirchhoff stress tensor. Note that the
operator $\nabla \cdot$ acting on the Cauchy stress tensor represents
differentiation with respect to coordinates in the current
configuration, while when acting on the first Piola-Kirchhoff stress
tensor represent differentiation with respect to coordinates in the
reference configuration.

## Conservation of angular momentum
Just like linear momentum, the angular momentum is also a conserved
quantity. We will not go through the derivation, but state that as a
consequence, the Cauchy stress tensor is symmetric
```{math}
:label:
\begin{align}
  \sigma = \sigma^T.
\end{align}
```


## References

```{bibliography}
:filter: docname in docnames
```

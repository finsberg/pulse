(section:constitutive_relations)=
# Constitutive relations
We have now covered a mechanical framework which holds any for
material in general. What differentiate the mechanics of soft
living tissue, like the myocardium, from other materials is the
constitutive relations which describes the response of a material to
applied load. Such constitutive relations often comes from
experimental observations, both observations of anatomical structure
but also from experiments done on tissue slabs.

We have already covered the theory of hyperelasticity and incompressibility in Section
{ref}`section:hyperelasticity` and {ref}`section:incompressibility` respectively
which are types of constitutive relations. In this section we will
cover constitutive relations which only apply to soft living tissue
such as the myocardium. In particular, we will consider a complete
constitutive model of the mechanical behavior of the myocardium that
accounts for both the passive and the active response of the myocardium.


## Modeling of the passive myocardium


The passive response of the myocardium has been investigated through
uni-axial, bi-axial and shear deformation experiments {cite}`dokos2002shear`.

In 2009 Holzapfel and Ogden proposed an orthotropic constitutive model
of the passive myocardium {cite}`holzapfel2009constitutive` which is
based on the experiments done in {cite}`dokos2002shear`, and is the
model used in this thesis. Other constitutive models for the passive
myocardium exists
{cite}`costa2001modelling,guccione1991passive,nash2000computational`
but is not considered here. The model
assumes a local orthonormal coordinate system with the fiber axis
$\mathbf{f}_0$, sheet axis $\mathbf{s}_0$ and sheet-normal axis $\mathbf{n}_0$.

From this coordinate system we define the invariants
```{math}
\begin{align}
  \begin{split}
    I_1 &= \mathrm{tr} \; (\mathbf{C}) ,\\
    I_{4\mathbf{f}_0} &= \mathbf{f}_0 \cdot (\mathbf{C} \mathbf{f}_0),\\
    I_{4\mathbf{s}_0} &= \mathbf{s}_0 \cdot (\mathbf{C} \mathbf{s}_0),\\
    I_{8\mathbf{f}_0\mathbf{s}_0} &=  \mathbf{s}_0 \cdot (\mathbf{C} \mathbf{f}_0),
  \end{split}
\end{align}
```
Here $I_{4\mathbf{f}_0} $ and $I_{4\mathbf{s}_0}$ are the stretches along the
fiber, sheet axis respectively and $I_{8\mathbf{f}_0\mathbf{s}_0}$ is
related to the angle between the fiber and sheets in the current
configuration given that they are orthogonal in the reference
configuration. Note that since $(\mathbf{f}_0, \mathbf{s}_0, \mathbf{n}_0)$ is an orthonormal
system, we have the relation $I_1 = I_{4\mathbf{f}_0} + I_{4\mathbf{s}_0} +I_{4\mathbf{n}_0}$,
and so $I_{4\mathbf{n}_0}$ is redundant. The orthotropic Holzapfel and Ogden
model reads
```{math}
:label: holzapfel_full
\begin{align}
  \begin{split}
  \Psi(I_1, I_{4\mathbf{f}_0},  I_{4\mathbf{s}_0},  I_{8\mathbf{f}_0\mathbf{s}_0}) =& \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)\\
  +& \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right) \\
  +& \frac{a_s}{2 b_s} \left( e^{ b_s (I_{4\mathbf{s}_0} - 1)_+^2} -1 \right)\\
  +& \frac{a_{fs}}{2 b_{fs}} \left( e^{ b_{fs} I_{8\mathbf{f}_0\mathbf{s}_0}^2} -1 \right).
\end{split}
\end{align}
```
Here $( x )_+ = \frac{1}{2} \left( x + |x| \right)$, so that the
the terms involving $I_{4\mathbf{f}_0}$ and $I_{4\mathbf{s}_0}$ only contribute to the
stored energy during elongation. From {eq}`holzapfel_full` we see
that it is easy to identify the physical meaning of each term. For
example the first term represents the isotropic contribution which is
the overall stiffness in the extracellular matrix while the second
term accounts for the extra stiffness along the fibers when they are
elongated. It is also straight forward to prove that the strain-energy
function is convex, and that the requirements for existence and
uniqueness discussed in {ref}`section:strain_energy_req` are
fulfilled.

In this thesis we have used a transversely isotropic version of
{eq}`holzapfel_full` which is obtained by setting $a_{fs} =
b_{fs}= a_s = b_s = 0$, i.e

```{math}
:label: holzapel_trans
\begin{align}
  \begin{split}
  \Psi(I_1, I_{4\mathbf{f}_0}) = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
  + \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right).
  \end{split}
\end{align}
```
If we further set $a_f = b_f = b = 0$ so that in $a$ is the
only nonzero parameter, then the Holzapfel-Ogden model reduces (after a
series expansion of the exponential and a limiting argument) to

```{math}
\begin{align}
  \Psi(I_1)  = \frac{a}{2} \left( I_1 - 3 \right),
\end{align}
```
which is the model of a Neo Hookean material. The Cauchy stress can be derived
analytically from {eq}`holzapfel_full`, by using the chain rule and
{eq}`cauchy_incomp`,

```{math}
\begin{align}
  \begin{split}
    J\sigma
    =& \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}}\mathbf{F}^{T} + p \mathbf{I}
    = \sum_{i \in \left\{ 1, 4\mathbf{f}_0,  4\mathbf{s}_0,  8\mathbf{f}_0\mathbf{s}_0 \right\} }
    \psi_i \frac{\partial I_i}{\partial \mathbf{F}}\mathbf{F}^{T} + p \mathbf{I} \\
    =& p \mathbf{I} + a \left( e^{ b (I_1 - 3)}  -1 \right) \mathbf{B}
    + 2 a_f (I_{4\mathbf{f}_0} - 1)_+  e^{ b_f (I_{4\mathbf{f}_0} - 1)^2} \mathbf{f} \otimes \mathbf{f} \\
    &+ 2 a_f (I_{4\mathbf{s}_0} - 1)_+  e^{ b_f (I_{4\mathbf{s}_0} - 1)^2} \mathbf{s} \otimes \mathbf{s}
    + a_{fs} I_{8\mathbf{f}_0\mathbf{s}_0}  e^{ b_{fs} I_{8\mathbf{f}_0\mathbf{s}_0}^2} \left( \mathbf{f} \otimes \mathbf{s}  +  \mathbf{s} \otimes \mathbf{f} \right),
  \end{split}
\end{align}
```
where $\mathbf{B} = \mathbf{F} \mathbf{F}^{T}$ is the left Cauchy-Green tensor,
$\mathbf{f} = \mathbf{F} \mathbf{f}_0$ and $\mathbf{s} = \mathbf{F} \mathbf{s}_0$.

## Modeling of the active contraction

One feature that separates the myocardium from other hyperelastic
materials such as rubber, is its ability to actively generate force
without external loads. This active component of the model can be
incorporated using two fundamentally different approaches known as the
*active stress* and *active strain* formulation.


```{figure} ./figures/Hill_muscle_model.png
---
name: hill_muscle_model
width: 300px
---
The classical three-element Hill muscle model with one contractile element and two non-linear springs, one arranged in series and one parallel.
```


The *active stress* approach is based on the classical three element
Hill model illustrated in {numref}`hill_muscle_model`, where the
active contribution naturally decomposes the total stress into a sum
of passive and active stresses
{cite}`nash2004electromechanical`. Hence, in the active stress
formulation {cite}`hunter1998modelling` one assumes that the total
Cauchy stress $\sigma$ can be written as an additive sum of one
passive contribution $\sigma_p$ and one active contribution $\sigma_a$,

```{math}
\begin{align}
  \sigma = \sigma_p + \sigma_a
\end{align}
```
The passive contribution is determined by the material model used

```{math}
\begin{align}
 \sigma_p = \frac{1}{J} \frac{\partial \Psi(\mathbf{F})}{\partial \mathbf{F}} \mathbf{F}^{T},
\end{align}
```
while the active contribution is given by

```{math}
\begin{align}
  \sigma_a = \sigma_{ff} \mathbf{f} \otimes \mathbf{f} +
  \sigma_{ss} \mathbf{s} \otimes \mathbf{s} +
  \sigma_{nn} \mathbf{n} \otimes \mathbf{n},
\end{align}
```
and the different constants $\sigma_{ff}, \sigma_{ss}$, and
$\sigma_{nn}$, which are the active stress in the fiber, sheet and
sheet-normal direction respectively, are typically coupled to the
electrophysiology and calcium dynamics.
There are experimental evidence that the active stresses in the
transverse direction of the fibers ($\sigma_{ss}$, and $\sigma_{nn}$),
are non-negligible {cite}`lin1998multiaxial`, and one approach is to assume
a uniform transverse activation in which the total active tension
can be written as

```{math}
:label: intro_active_stress
\begin{align}
  \sigma_a = T_a \left[\mathbf{f} \otimes \mathbf{f} +
   \eta\left( \mathbf{s} \otimes \mathbf{s} +
  \ \mathbf{n} \otimes \mathbf{n} \right)\right],
\end{align}
```
where $\eta$ represent the amount of transverse activation and $T_a
\in \mathbb{R}$ is the magnitude of the active tension.
In the limiting case ($\eta = 0.0$), the active tension acts purely
along the fibers and {eq}`intro_active_stress` reduces to

```{math}
\begin{align}
  \sigma_a = T_a \mathbf{f} \otimes \mathbf{f}.
\end{align}
```
Note that, by observing  that

```{math}
\begin{align*}
  \frac{\partial I_{4\mathbf{a}_0}}{\partial \mathbf{F}}
  = \frac{\partial (\mathbf{a}_0  \cdot \mathbf{C} \mathbf{a}_0 )}{\partial \mathbf{F}}
  = 2 \mathbf{a} \otimes \mathbf{a}_0 \implies
  \mathbf{a} \otimes  \mathbf{a}= \frac{1}{2} \frac{\partial I_{4\mathbf{a}_0}}{\partial \mathbf{F}} \mathbf{F}^{T}
\end{align*}
```
and that $I_1 =  I_{4\mathbf{f}_0} +  I_{4\mathbf{s}_0} +  I_{4\mathbf{n}_0}$,
we can instead decompose the strain-energy function into a passive and active
parts {cite}`pathmanathan2010cardiac`, $\Psi= \Psi_p + \Psi_a$, with

```{math}
\begin{align}
\Psi_a = \frac{T_a}{2J} \left(( I_{4\mathbf{f}_0} - 1)  + \eta \left[ (I_1 - 3) -
    (I_{4\mathbf{f}_0} - 1)\right] \right),
\end{align}
```
so that $J \sigma_a  = \frac{\partial \Psi_a}{\partial \mathbf{F}}
\mathbf{F}^{T}$.

The *active strain* formulation is a relatively new way of modeling the
active contraction in the heart and was first introduced in
{cite}`taber2000modeling`. This formulation is based on a
multiplicative decomposition of the deformation gradient,

```{math}
:label: active_strain
\begin{equation}
 \mathbf{F} = \mathbf{F}_e \mathbf{F}_a.
\end{equation}
```
The active part $\mathbf{F}_a$, is an inelastic process driven by the
biochemistry and can be seen as the actual distortion of the
microstructure. The elastic part $\mathbf{F}_e$ is responsible for preserving
compatibility of the tissue and stores all the energy in the
deformations. As a consequence, the strain energy function is a
function of the elastic deformation gradient only. The
decoupling can be illustrated by considering two sarcomeres connected in
series as shown in {numref}`actstrain`.

```{figure} ./figures/actstrain.png
---
name: actstrain
width: 300px
---
Illustration of the active strain formulation. During the active deformation, the sarcomeres shortens as if they were all detached. The elastic deformation ensures compatibility of the tissue.
```

The general form of the active deformation gradient for a
material with an orthotropic active response is given by

```{math}
:label: active_strain_Fa_general
\begin{equation}
  \mathbf{F}_a =  \mathbf{I}
  - \gamma_f \mathbf{f}_0 \otimes \mathbf{f}_0
  - \gamma_s \mathbf{s}_0 \otimes \mathbf{s}_0
  - \gamma_n \mathbf{n}_0 \otimes \mathbf{n}_0
\end{equation}
```

We add the constraint $\mathrm{det} \; (\mathbf{F}_a) = 1$, meaning that the active
deformation is volume preserving. Further we assume that the activation is
transversely isotropic, so that the sheet and sheet-normal axis is
treated in the same way. It is then straight forward to verify that
$\gamma_n = \gamma_s =1- (1-\gamma_f)^{-1/2}$, and we have

```{math}
:label: intro_active_strain_Fa_gjerald
\begin{equation}
  \mathbf{F}_a = (1 - \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  + \frac{1}{\sqrt{1 - \gamma}} (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0),
\end{equation}
```
where we set $\gamma = \gamma_f$ for convenience.


While the motivation behind the active stress formulation is purely
physiological and based on the classical Hill model shown in
{numref}`hill_muscle_model`, the motivation behind the active strain
formulation is more driven by ensuring mathematical robustness. In
particular it has been shown {cite}`ambrosi2012active` that with the
active strain formulation, properties such as frame invariance and
rank-one ellipticity is inherited from the strain energy function. In
contrast, rank-one ellipticity is not guaranteed for the active stress
formulation.

For a more extensive comparison of the active stress and active strain
approach we refer to {cite}`ambrosi2012active,giantesio2017comparison`,
and for an overview of other methods to model the active contraction
we refer to {cite}`goriely2017five`.

## References

```{bibliography}
:filter: docname in docnames
```

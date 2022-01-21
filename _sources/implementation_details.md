# Implementation details
The cardiac mechanics solver developed during the work of this thesis
is implemented using the finite element framework FEniCS. Here we
briefly explain the main components of FEniCS as well as some
numerical considerations made when implementing the solver.


## The FEniCS Project

The FEniCS project is an open-source computing platform for solving
partial differential equations (PDEs) using the finite element method (FEM).
Solving PDEs using FEM involves many implementation details that can
be tedious to implement yourselves. The idea behind FEniCS is to
automate code generation so that the user can spend more time on doing
research and less time on implementation of assembly matrices. At the
core of FEniCS is DOLFIN {cite}`logg2012dolfin`, which is a C++/Python
library, and works as the main interface in FEniCS. In this thesis
only the Python interface has been used, in which C++ code is
automatically generated using SWIG. This allows for simplicity through
the Python scripting language and the speed of the C++ language.
The domain specific language used to represent weak formulations is
called the Unified Form Language (UFL) {cite}`alnaes2014unified`, and
allows for e.g automatic differentiation of forms and expressions. The
FEniCS form compiler (FFC) {cite}`logg2012ffc` compiles code written in
UFL to Unified Form-assembly Code (UFC) {cite}`alnaes2012ufc` which are
optimized C++ code. The Python interface also makes use of the Instant
module which allows for just-in-time (JIT) compilation of C++
code. The compiled code is also stored in a cache so that compilation
of a form only happens once. Also, the relatively new UFL Analyser and
Compiler System (UFLACS) allows for fast compilation of complex forms
such as variational formulations that include the Holzapfel Ogden
material model {eq}`holzapfel_full`.

For more information about FEniCS, the reader is referred to the
official web page (https://fenicsproject.org) or any of the
cited literature.

## Numerical considerations

The solution of non-linear problems such as the one described here are
typically solved using methods like Newton's method. The convergence of
such methods depends on the initial guess, and if the
initial guess is too far from the true solution, the solver might diverge.
Moreover, if the initial guess is close to the true solution the
convergence rate is in general quadratic.

Let us consider a typical numerical problem of inflating the ventricular geometry from a
stress-free configuration to end-diastole. This involves increasing
the pressure, or the boundary traction on the endocardium, from zero
to the end-diastolic pressure. A strategy know as the *incremental load* technique is usually a good approach. In this strategy you select some incremental step-size (for instance $0.4$ kPa), and increase the pressure linearly until the target pressure is
reached. If the solver diverges you decrease the step-size (for instance by a factor of 0.5) until convergence is reached, and continue to step up the pressure with the new step-size. This is very robust, but definitely a slow approach. Since many of the
constitutive models for myocardium consist of an exponential relationship between the stress and strain (so called Fung-type relation), the amount of stress needed to displace a material will be higher if the material is a state with high strain compared to a state
of low strain. Therefore, in the low strain state, the Newtons solver might perform fewer iterations to reach convergence when the load is increased. As a result, one could improve the incremental load technique by adapting the step size if the number of newton iterations are below a certain threshold (for instance $8$ iterations).

An even more clever strategy uses a technique from bifurcation and chaos theory and is known as numerical continuation {cite}`allgower2003introduction`.  Suppose we want to
solve the non-linear problem $F(\mathbf{u}, \lambda)=0$ with state variable
$\mathbf{u}$ and parameter $\lambda$. For instance $\mathbf{u}$ could be the
displacement and $\lambda$ could be the endocardial pressure.
The idea behind numerical continuation is that given a solution pair
$(\mathbf{u}_0, \lambda_0)$ there exist (under conditions stated by the
implicit function theorem) a solution curve $\mathbf{u}(\lambda)$ such that
$F(\mathbf{u}(\lambda), \lambda)=0$ and $\mathbf{u}(\lambda_0) = \mathbf{u}_0$.
To explicitly find such a curve is not always easy but a simple
approximation can be found by linear extrapolation: Given two pairs
$(\mathbf{u}_0, \lambda_0)$ and $(\mathbf{u}_1, \lambda_1)$, and a new target
parameter $\lambda_2$, a possible solution is
```{math}
\begin{align}
  \mathbf{u}_2 =  (1-\delta)\mathbf{u}_0 + \delta \mathbf{u}_1 && \delta = \frac{\lambda_2 - \lambda_0}{\lambda_1 - \lambda_0}.
\end{align}
```
Choosing $\mathbf{u}_2$ as initial guess for the non-linear solver has been
successfully performed by others in non-linear cardiac mechanics
problems {cite}`pezzuto2013mechanics`, and this approach is also used
in this thesis.


## References

```{bibliography}
:filter: docname in docnames
```

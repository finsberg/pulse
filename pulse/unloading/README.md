# Unloading #

Ventricular geometries obtained from medical images are geometries subjected to physical loads such 
as blood pressure. The unloaded reference geometry therefore needs to be estimated.

This repository contains code for estimating the unloaded reference geometry 
using three iterative methods one is the fixed-point method described in [1], the other is the method called the raghavan-method is described in [2], and the third is a hybrid version of the two, starting with the fixed-point method and switching to raghavan when the fixed-point method fails. In general the fixed-point method is better but tend to fail for thinned wall BiV geometries, in which the raghavan-method might be a good alternative. The implementation of these methods are found in `unloader.py`

# Coupled unloading/materal parameter estimation #
Finding the unloaded geometry is one thing, but the unloaded geometry also depends on the material parameters, which is also unknown. In `optimize_material.py` we combine the unloading with the parameter estimation by iteratively unload and estimate material parameters. This method is similar to the one described in [3]. Note this requires that you also have pulse_adjoint. 


# References #
[1] Bols, Joris, et al. "A computational method to assess the in vivo stresses and unloaded configuration of patient-specific blood vessels." Journal of computational and Applied mathematics 246 (2013): 10-17.

[2] Raghavan, M. L., Baoshun Ma, and Mark F. Fillinger. "Non-invasive determination of zero-pressure geometry of arterial aneurysms." Annals of biomedical engineering 34.9 (2006): 1414-1419.

[3] Nikou, Amir, et al. "Effects of using the unloaded configuration in predicting the in vivo diastolic properties of the heart." Computer methods in biomechanics and biomedical engineering 19.16 (2016): 1714-1720.

# License #
UNLOADING is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
at your option) any later version.

UNLOADING is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with UNLOADING. If not, see <http://www.gnu.org/licenses/>.
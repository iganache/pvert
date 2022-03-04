# pvert
A set of Python + MATLAB codes for modeling radar backscatter and emission from one and two layer geologic media, with or without distributed heterogeneities. Backscattering and emission from homogenous media with rough interfaces are computed using the Improved Integral Equation Method (Fung et al., 2002). EM wave propagation and scattering within a homogenous background medium with discrete, spheroidal scatterers are described using vector radiative transfer (VRT) relations. A first order iterative approach is then used to solve the VRT equations (Tsang, 1985). Scattering and extinction due to heterogeneities are computed using the T-matrix approach (Mischenko et al., 2000). We use the [pytmatrix] (https://pypi.org/project/pytmatrix/) module to implement a T-matrix solver. 


See [usage instructions](https://github.com/iganache/pvert/wiki) for more information on model functionalities and execution.

Requires [NumPy](https://numpy.org/doc/stable/contents.html), [SciPy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), [pytmatrix](https://pypi.org/project/pytmatrix/), [MATLAB](https://www.mathworks.com/products/matlab.html), and [MATLAB engine for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

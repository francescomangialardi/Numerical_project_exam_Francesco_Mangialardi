# Lepage's analysis
*Project by Francesco Mangialardi, realised for the 'Theoretical and Numerical Aspects of Nuclear Physics'
exam at University of Bologna, a.y. 2022-2023*

Modules, libraries and related methods required for the execution of the project's main script:
- sys
- math
- numpy
- pyplot from matplotlib
- gamma from scipy.special
- curve_fit, root, brentq from scipy.optimize


This project partially reproduces the [analysis of P.Lepage](https://arxiv.org/abs/nucl-th/9706029) concerning the construction of effective theories in non-relativistic quantum mechanics. The files in this project are organised as follows.

## The `Classes.py` file
It is the smaller of all files, and contains the definition of the only class implemented in the project, the `Grid` class. A `Grid` object contains all the necessary details about the discretization of the radial axis, such as its left and right limits, and the array containing all the points in which it is discretized. Practically every step of the analysis carried out in this project, like the computation of the bound state energies or the scattering phase shifts for a given potential, implies first the creation of a suitable instance of object of this class. As a matter of fact, all the main routines contained in the `Functions.py` file require in input, or create inside their body, a `Grid` object.


## The `Potentials.py` file
This file contains the definition of all potentials used in the project. Each of the potentials is defined as a Python function whose arguments are the value of the radial coordinate at which one wants to evaluate the potential, and the values of the parameters that enter in the definition of the potential function, if there are any. The return value of each of these functions is the value of the potential at the radial point given in input.

If one wants to modify the project and use a type of potential not present in this file, for example to generate the synthetic data that serve as base for the whole effective theory analysis, then he/she should first add the chosen potential in the `Potentials.py` file, respecting the above implementation prescriptions. 

## The `Functions.py` file
It contains all the routines necessary for the analysis. Among these, the most important are:

- the `solve_for_bound_state` routine, which, given in input a choice of number of nodes for the radial wavefunction and a discretized version of the potential (plus other operational parameters), solves the schrodinger problem for S-wave bound states by using Numerov's method to compute the (discretized) wavefunction and the Cooley's energy correction formula to refine the bound state energy estimate. For the computation of the wavefunction, the matching method is implemented, shooting both from the origin and from an estimated right boundary of the radial axis where we impose the wavefunction to be zero. The output of the `solve_for_bound_state` routine are the estimated energy level, an estimate (to first order) of its error coming from Cooley's formula, and the normalised wavefunction.

- the `compute_phase_shift` routine, which, given in input an energy value for a S-wave scattering state and a discretized version of the potential (plus other operational parameters), computes the phase shift associated to the scattering state chosen. The phase shift is computed by evolving, again through Numerov's method, the radial wavefunction of the scattering state from the origin of the radial axis up to a point where the effect of the potential is negligible, where the radial wavefunction $\chi_{k}(r)$ can be approximated by [^1] $$ \chi_{k}(r)\simeq 2(sin(kr)-\eta \ln(2kr) +\delta_{k})$$
where $k$ is the wavenumber (related to the scattering energy $E$ as $k=\sqrt{2mE}$) and $\eta=-m\alpha/k$ is the so-called Sommerfeld parameter. To estimate the phase shift $\delta_{k}$, a fit of the computed wavefunction to this functional form is performed, using the Scipy method `curve_fit`. The output of the `compute_phase_shift` routine can be chosen to be either the full outcome of the fit analysis, i.e. the estimated value of the fit parameters (the phase shift and an overall rescaling factor) and the covariance matrix of the fit, or only one or both the fit parameters estimates.

- the `optimize_ET1_with_energy` routine, which is used to find an optimal value for the unique parameter c of the effective theory's potential at order 2 of the cutoff. The optimal value for c is found by minimizing the relative discrepancy between a given bound state energy (used as reference) and its estimate found using the effective theory potential and the `solve_for_bound_state` routine. Minimization is done by first bracketing the root and then using the Scipy root finding method `brentq`. The output of the `optimize_ET1_with_energy` routine is the best guess for the c parameter.

- the `optimize_ET2_with_energy` routine, which is used to find an optimal value for the two parameters c and d of the effective theory's potential (for S-wave states) at order 4 of the cutoff. The optimal value for the two parameters is found by minimizing the relative discrepancy between two given bound state energies (used as reference) and their estimate found using the effective theory potential and the `solve_for_bound_state` routine. Minimization in this two dimensional parameter space is done using the Scipy root finding method `root`, starting from an initial guess in input. The output of the `optimize_ET2_with_energy` routine is a numpy array containing the best guess for the c and d parameters.

The routines `compute_phase_shift`, `optimize_ET1_with_energy`, `optimize_ET2_with_energy` have an optional input parameter (with default value `False`) that, if set to `True`, toggles on the printing on screen of some plots which are useful to verify that the fitting or root finding routines contained into them have worked properly.

All other routines in the `Functions.py` file are auxiliary to the four described here in detail. Among these other minor routines, a couple worth mentioning are the `normalise_wavefunction`, that uses Simpson method to integrate a given wavefunction and then rescales it to produce in output its normalised version, the `compute_energy_difference`, that is the one which explicitly computes the energy correction using Cooley's formula, and the couple of functions `numerov_forward_shooting, numerov_backward_shooting`, used to recursively find, for a given energy, the value of the corresponding discretized radial wavefunction along all the radial axis up to a stopping point, starting either from the origin or the right boundary of the axis.

[^1]: the formula holds in presence of a Coulombic interaction + a generic short-range radial potential.

## The `Tester_Bound_States.py` and the `Tester_Phase_Shifts.py` files
These two files do not strictly contribute to reproduce the analysis done by P.Lepage, but serve as a check that the routines implemented in the project work properly. The check is done by using the routines to numerically solve models for which also an analytical solution is known, and comparing the two type of results. In particular, the `Tester_Bound_States.py` is used to verify the accuracy of the method `solve_for_bound_state` (the one that solves the bound state problem) by applying it to the exactly solvable cases of the Coulomb potential and of the 3D isotropic harmonic oscillator, while the `Tester_Phase_Shifts.py` puts to test the method `compute_phase_shift` on the Coulomb potential model only.

## The `Lepage_Analysis.py` file
This file is the one to be executed to reproduce the first part of the analysis done by P.Lepage in the paper [``How to renormlize the Schrodinger equation''](https://arxiv.org/abs/nucl-th/9706029). The analysis is divided in few steps, that we briefly recap:

1. Choose a short range potential (here the selected one is a radial square well) to be added to the Coulomb one, then in this setting compute S-wave bound state energies and phase shifts, respectively for selected values of the principal quantum number and a set of chosen energies for the scattering states. In this project the bound state energies, together with their eigenfunctions, are computed using the routine `solve_for_bound_state`, while the scattering phase shifts are estimated via the method `compute_phase_shift`. The values of the energies and of the phase shifts is printed out by the program, while the radial bound state eigenfunctions with lowest principal quantum number are plotted in a picture that is saved in the `Figures` folder.
Forgetting that their physical origin is known, from this point forward these data serve as an example of (synthetic) dataset representing a Schrodinger problem with a potential that is of Coulomb type, plus a short range part considered unknown. All the analysis that follows is aimed at the construction of an approximate model for the short range part of the potential which correctly reproduce the dataset.

2. Use a delta function as first guess for the short range part of the potential, and compute the binding energies using the formula coming from first order perturbation theory. The only parameter in the potential is the constant to be put in front of the delta function, that is fixed by requiring the computed lowest binding energy to coincide with the synthetic one. The values of the binding energies found with this first method are printed in output.

3. Use then an effective theory approach to build the potential. First an effective theory at order two of the cutoff value (which is chosen to be equal to the width of the ptential well) is used, and then one at order four. The parameters of the effective potentials are estimated using the two routines `optimize_ET1_with_energy` and `optimize_ET2_with_energy`, choosing the lowest synthetic binding energies as the ones to be reproduced. The choice to use the binding energies data over those of the phase shifts for this estimate, has been dictated by the availability of a much faster routine to compute them (at equal order of magnitude for the energy). Having found an optimal value for the parameters of the two types of effective potential, these forms for the potentials are then used to try to reproduce the synthetic energies and phase shifts. The results are once again printed in output.

4. Reorganize and plot (in figures stored in the `Figures` folder) all the data coming from the analysis done in points (2) and (3). In particular, the following plots are produced:
- A plot containing the graph of all the potentials used in the analysis
- A plot depicting the relative errors (with respect to the synthetic data) of the estimated binding energies in the different approaches. Here are included also the relative errors in case of considering only the Coulomb part of the potential
- An analogous plot of the previous one, but this time for the phase shifts

5. Perform a final analysis on the cutoff dependence of the results obtained using the effective theory at order 2. This is done by repeating the effective's theory analysis at different values of the cutoff, and comparing the relative errors on the estimated binding energies in the various cases. The results are plotted in a graph and saved in the `Figures` folder.


Since the execution of the entire analysis just described takes a long computational time ($\sim 1h$ as order of magnitude; the routines which take longer are the root searching ones `brentq` and `root`), four flag variables have been inserted at the beginning of the script to toggle on and off (independently) the execution of the different parts of the script, corresponding to the part of the analysis explained above in points (2), (3) and (5). An additional flag `enable_control_plots` can be used to switch on the printing on screen of all the plots coming from the routines `compute_phase_shift`, `optimize_ET1_with_energy`, `optimize_ET2_with_energy`, that may be useful to verify their correct functioning.

### Trial run numerical results
For completeness, in the project folder is also inserted a text file `Trial run numerical results.txt` which contains the output that has been printed onto the terminal by the `Lepage_Analysis.py` file in a full trial run performed with the following settings
```
energy_bound = 1e-5 
boundary_condition = 0.
shooting_parameter = pow(10, -5)
mass = 1
alpha = 1
well_height = 1.5
well_width = 1.3
```
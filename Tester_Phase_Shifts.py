
'''
FILE THAT TESTS THE CORRECT FUNCTIONING OF THE ROUTINE TO COMPUTE THE PHASE SHIFT OF A SCATTERING STATE
IN PRESENCE OF A COULOMBIC POTENTIAL PLUS EVENTUALLY A SHORT RANGE PART
WE TEST THE ROUTINE ON THE COULOMB POTENTIAL AND COMPARE IT
WITH THE OUTCOMES EXPECTED FROM THE THEORETICAL RESULTS
'''

import math as m
import numpy as np
from scipy.special import gamma

import Potentials
import Functions
from Classes import Grid

# operational settings
enable_phase_shift_plots = False     # flag to toggle on and off the display of the plots to check that the computation of each phase shift is correct

# initial conditions
boundary_condition = 0.
shooting_parameter = pow(10, -5)

# parameters of Schrodinger equation
mass = 1
alpha = 1


phase_shifts = []
chosen_energies = [1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

print("\n- Phase shifts from theory vs computed values, for the Coulomb potential-")
print("Energy | Expected phase shift | Computed phase shift")

for E in chosen_energies:
    r_max_estimate = pow(10, 3 - m.floor(m.log10(E))) # this is an estimate of a good approximation of r_max = ininity for the energy level considered (the formula comes from testing different values of r_max for various energy levels)
    grid = Grid(10*r_max_estimate, r_max_estimate)
    discretized_potential = [Potentials.Coulomb(r, alpha) for r in grid.r_values]
    shift = Functions.compute_phase_shift(E, grid, boundary_condition, shooting_parameter, mass, alpha, discretized_potential, 2, enable_phase_shift_plots)
    phase_shifts.append(shift)

    wavenumber = np.sqrt(2 * mass * E)
    Sommerfeld_parameter = - mass * alpha / wavenumber
    print(f"{E}:   {np.angle(gamma(complex(1,Sommerfeld_parameter)))} | {shift}")

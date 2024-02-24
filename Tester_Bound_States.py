
'''
FILE THAT TESTS THE CORRECT FUNCTIONING OF THE ROUTINE TO FIND THE BOUND STATE EIGENFUNCTIONS AND EIGENENERGIES.
WE TEST THE ROUTINE ON THE COULOMB AND THE 3D ISOTROPIC HARMONIC OSCILLATOR POTENTIALS AND COMPARE IT
WITH THE OUTCOMES EXPECTED FROM THE ANALYTICAL SOLUTION OF THE MODELS

THIS TESTER FILE PRODUCES TWO FIGURES ON SCREEN TO VISUALLY CHECK THE COMPUTED WAVEFUNCTIONS
'''

import math as m
from matplotlib import pyplot as plt

import Potentials
import Functions
from Classes import Grid


#operational settings
energy_bound = 1e-5 #it is a relative energy bound

# initial conditions
boundary_condition = 0.
shooting_parameter = pow(10, -5)

# parameters of Schrodinger equation
mass = 1
alpha = 1
omega = 1


#----------------- COULOMB POTENTIAL ANALYSIS --------------------

energies_1 = []
wavefunctions_1 = []
chosen_energy_levels_1 = [1, 2, 3, 4, 5, 10, 20, 30] #values of the principal quantum number n chosen to be analysed

print("\n- Energy levels from theory vs computed values, for the Coulomb potential-")

for n in chosen_energy_levels_1:
    r_max_estimate = m.ceil(2*pow(n,2)/(mass*alpha) * 5 * (2-m.erf(n-3))) # radial value of the turning point associated to the energy level (from the theory) with a corrective factor that decreases for increasing n
    grid_1 = Grid(20*r_max_estimate, r_max_estimate)
    discretized_potential = [Potentials.Coulomb(r, alpha) for r in grid_1.r_values]
    state = Functions.solve_for_bound_state(n-1, discretized_potential, grid_1, energy_bound, boundary_condition, shooting_parameter, mass)
    energies_1.append(state[0])
    wavefunctions_1.append(state[2])

    print(f"{n}S: {-pow(alpha, 2) * mass / (2 * pow(n, 2))} | {state[0]}")
    if n <= 4:
        plt.plot(grid_1.r_values, state[2], label=f"{n}S")


plt.axhline(y=0, color='black', linewidth = 0.8)
plt.xlabel("r")
plt.ylabel(r"$\chi_{n0}$")
plt.xlim(left = 0, right = 70) # this has to be set manually to get the optimal display effect
plt.legend()
plt.title(r'Radial energy eigenfunctions with $\ell=0$ for Coulomb potential')
plt.show()


#----------------- 3D ISOTROPIC HARMONIC OSCILLATOR ANALYSIS --------------------

energies_2 = []
wavefunctions_2 = []
chosen_energy_levels_2 = [0, 2, 4, 6, 8, 10, 20, 30] #values of the principal quantum number n chosen to be analysed

print("\n- Energy levels from theory vs computed values, for the 3D isotropic harmonic oscillator-")

for n in chosen_energy_levels_2:
    r_max_estimate = m.ceil(m.sqrt(2*(n+3/2)/(mass*omega)) * 5 ) # five times the radial value of the turning point associated to the energy level (from the theory)
    grid_2 = Grid(20*r_max_estimate, r_max_estimate)
    discretized_potential = [Potentials.Harmonic_Oscillator_3D(r, mass, omega) for r in grid_2.r_values]
    state = Functions.solve_for_bound_state(n/2, discretized_potential, grid_2, energy_bound, boundary_condition, shooting_parameter, mass)
    energies_2.append(state[0])
    wavefunctions_2.append(state[2])

    print(f"{n}S: {omega*(n+3/2)} | {state[0]}")
    if n <= 6:
        plt.plot(grid_2.r_values, state[2], label=f"{n}S")


plt.axhline(y=0, color='black', linewidth = 0.8)
plt.xlabel("r")
plt.ylabel(r"$\chi_{n0}$")
plt.xlim(left = 0, right = 8) # this has to be set manually to get the optimal display effect
plt.legend()
plt.title(r'Radial energy eigenfunctions with $\ell=0$ for the 3D isotropic harmonic oscillator', fontsize= 8.5)
plt.show()

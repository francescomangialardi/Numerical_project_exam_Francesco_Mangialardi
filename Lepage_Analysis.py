
'''
FILE TO BE EXECUTED TO REPRODUCE THE FULL ANALYSIS OF P. LEPAGE CONCERNING THE CONSTRUCTION OF EFFECTIVE THEORIES
IN NON-RELATIVISTIC QUANTUM MECHANICS (reference: https://arxiv.org/abs/nucl-th/9706029)
TO RUN THIS SCRIPT, ONE MUST PROVIDE THE THREE AUXILIARY FILES: Classes.py, Functions.py, Potentials.py

THE SCRIPT MAY BE RUNNED ENTIRELY OR ONLY PARTIALLY; THE PARAMETERS TO CUSTOMIZE THE
CHOICE OF OUTPUT ARE COLLECTED UNDER THE SECTION "operational settings"

Project by Francesco Mangialardi, realised for the 'Theoretical and Numerical Aspects of Nuclear Physics'
exam at University of Bologna, a.y. 2022-2023
'''

import math as m
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

import Potentials
import Functions
from Classes import Grid


# operational settings (to reproduce the full analysis of Lepage set as True the first four parameters)
enable_naive_analysis = True    # flag to toggle on and off the execution of the naive analysis, which approximates the short range potential as a delta function, at first order in perturbation theory
enable_ET1 = True     # flag to toggle on and off the effective theory analysis at order two of the cutoff
enable_ET2 = True      # flag to toggle on and off the effective theory analysis at order four of the cutoff
enable_cutoff_dependence_analysis = True   # flag to toggle on and off the study of the dependence of the effective theory analysis on the value of the cutoff

enable_control_plots = False     # flag to toggle on and off the display on screen of the plots present in the routine to compute the phase shifts and in those to optimize the effective theory parameters. These plots are used to visually check that the program is working properly

energy_bound = 1e-5     # required relative precision for the computation of binding energies


# initial conditions
boundary_condition = 0.
shooting_parameter = pow(10, -5)

# parameters of Schrodinger equation
mass = 1
alpha = 1
well_height = 1.5 #height of square well
well_width = 1.3 #width of square well


# -------------------------------- GENERATE SYNTHETIC DATA -------------------------
print("\n--- SYNTHETIC DATA ---")

# generate binding energies

synthetic_energies = []
chosen_energy_levels = [1, 2, 3, 4, 5, 6, 10, 20, 30] #values of the principal quantum number n chosen to be analysed

print("\n- Energy levels for the synthetic potential -")

for n in chosen_energy_levels:
    r_max_estimate = m.ceil(2*pow(n,2)/(mass*alpha) * 5 * (2-m.erf(n-3))) # as estimate for the rightmost point on the radial axis to consider, we choose the location of the turning point for the Coulomb potential, multiplied by a factor >1 which decreases as n increases
    grid = Grid(20*r_max_estimate, r_max_estimate)
    discretized_potential = [Potentials.Coulomb_and_well(r, alpha, well_width, well_height) for r in grid.r_values]
    state = Functions.solve_for_bound_state(n-1, discretized_potential, grid, energy_bound, boundary_condition, shooting_parameter, mass)
    synthetic_energies.append(state[0])

    print(f"{n}S: {state[0]}")
    if n <= 4:
        plt.plot(grid.r_values, state[2], label=f"{n}S")


plt.axhline(y=0, color='black', linewidth = 0.8)
plt.xlabel("r")
plt.ylabel(r"$\chi_{n0}$")
plt.xlim(left = 0, right = 50) # this has to be set manually to get the optimal display effect
plt.legend()
plt.title(r'Radial energy eigenfunctions with $\ell=0$ for the synthetic potential', fontsize = 11)
plt.savefig('./Figures/Bound_states_eigenfunctions_synthetic_potential.png', bbox_inches='tight')
plt.close()


# generate phase shifts

synthetic_phase_shifts = []
chosen_scattering_energies = [1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

print("\n- Phase shifts for the synthetic potential -")
print("Energy | phase shift")

for E in chosen_scattering_energies:
    r_max_estimate = pow(10, 3 - m.floor(m.log10(E))) # use same guess of Coulomb potential (see file "Tester_Phase_Shifts")
    grid = Grid(10 * r_max_estimate, r_max_estimate)
    discretized_potential = [Potentials.Coulomb_and_well(r, alpha, well_width, well_height) for r in grid.r_values]
    shift = Functions.compute_phase_shift(E, grid, boundary_condition, shooting_parameter, mass, alpha, discretized_potential, 2, enable_control_plots)
    synthetic_phase_shifts.append(shift)
    print(f"{E}:  {shift}")


# -------------------------------- NAIVE APPROXIMATION WITH DELTA FUNCTION -----------------------
if enable_naive_analysis:
    print("\n--- NAIVE ANALYSIS ---")

    # tune the parameter c using first order perturbation theory to reproduce the lowest binding energy
    n_max = max(chosen_energy_levels) # quantum number n of the bound state computed having lowest binding energy
    factor = n_max/(alpha*mass) # we use same formula of Lepage but restore the mass and alpha
    c_naive = m.sqrt(np.pi) * (max(synthetic_energies) * pow(factor, 3) + factor / (2*mass))
    print("\nEstimated value for the constant of the delta function:", c_naive)

    # compute binding energies using first order perturbation theory
    naive_energies = []

    print("\n- Energy levels from the naive approximation -")
    for n in chosen_energy_levels:
        E_naive = -pow(alpha, 2) * mass / (2 * pow(n, 2)) + c_naive/m.sqrt(np.pi) * pow(alpha*mass/n, 3)
        naive_energies.append(E_naive)
        print(f"{n}S: {E_naive}")


# ------------------------------------ EFFECTIVE THEORY ANALYSIS --------------------------------------------
cutoff = well_width        

if enable_ET1:
    print("\n--- EFFECTIVE THEORY ANALYSIS AT ORDER 2 ---")

    # estimate value of c parameter
    c_ET1 = Functions.optimize_ET1_with_energy(cutoff, alpha, mass, energy_bound, boundary_condition, shooting_parameter, synthetic_energies[8], chosen_energy_levels[8], enable_control_plots)
    print("\nEstimated value for the constant of the second order term:", c_ET1)

    # compute energies
    ET1_energies = []

    print("\n- Energy levels from the effective theory at order 2 of the cutoff -")

    for n in chosen_energy_levels:
        r_max_estimate = m.ceil(2*pow(n,2)/(mass*alpha) * 5 * (2-m.erf(n-3)))
        grid = Grid(20*r_max_estimate, r_max_estimate)
        discretized_potential = [Potentials.ET1(r, cutoff, alpha, c_ET1) for r in grid.r_values]
        state = Functions.solve_for_bound_state(n-1, discretized_potential, grid, energy_bound, boundary_condition, shooting_parameter, mass)
        ET1_energies.append(state[0])

        print(f"{n}S: {state[0]}")


    # compute phase shifts
    ET1_phase_shifts = []

    print("\n- Phase shifts from the effective theory at order 2 of the cutoff -")
    print("Energy | phase shift")

    for E in chosen_scattering_energies:
        r_max_estimate = pow(10, 3 - m.floor(m.log10(E))) # use same guess of Coulomb potential (see file "Tester_Phase_Shifts")
        grid = Grid(10 * r_max_estimate, r_max_estimate)
        discretized_potential = [Potentials.ET1(r, cutoff, alpha, c_ET1) for r in grid.r_values]
        shift = Functions.compute_phase_shift(E, grid, boundary_condition, shooting_parameter, mass, alpha, discretized_potential, 2, enable_control_plots)
        ET1_phase_shifts.append(shift)
        
        print(f"{E}:  {shift}")


if enable_ET2:
    print("\n--- EFFECTIVE THEORY ANALYSIS AT ORDER 4 ---")

    # estimate values of parameters c and d
    # here we use as first guess value for c the one obtained by optimizing the effective theory at order 2, to help the convergence of the optimization routine with two parameters
    # if one has chosen not to perform the effective theory analysis at order 2, we use here the optimization routine of the effective theory at order 2 to estimate a guess value for c
    if not enable_ET1:
            c_ET1 = Functions.optimize_ET1_with_energy(cutoff, alpha, mass, energy_bound, boundary_condition, shooting_parameter, synthetic_energies[8], chosen_energy_levels[8], enable_control_plots)
    
    c_ET2, d_ET2 = Functions.optimize_ET2_with_energy(cutoff, alpha, mass, energy_bound, boundary_condition, shooting_parameter, synthetic_energies[8], chosen_energy_levels[8], synthetic_energies[7], chosen_energy_levels[7], [c_ET1, 1], enable_control_plots)

    print("\nEstimated value for the constant of the second order term:", c_ET2)
    print("Estimated value for the constant of the fourth order term:", d_ET2)

    # compute energies
    ET2_energies = []

    print("\n- Energy levels from the effective theory at order 4 of the cutoff -")

    for n in chosen_energy_levels:
        r_max_estimate = m.ceil(2*pow(n,2)/(mass*alpha) * 5 * (2-m.erf(n-3)))
        grid = Grid(20*r_max_estimate, r_max_estimate)
        discretized_potential = [Potentials.ET2(r, cutoff, alpha, c_ET2, d_ET2) for r in grid.r_values]
        state = Functions.solve_for_bound_state(n-1, discretized_potential, grid, energy_bound, boundary_condition, shooting_parameter, mass)
        ET2_energies.append(state[0])

        print(f"{n}S: {state[0]}")


    # compute phase shifts
    ET2_phase_shifts = []

    print("\n- Phase shifts from the effective theory at order 4 of the cutoff -")
    print("Energy | phase shift")

    for E in chosen_scattering_energies:
        r_max_estimate = pow(10, 3 - m.floor(m.log10(E))) # use same guess of Coulomb potential (see file "Tester_Phase_Shifts")
        grid = Grid(10 * r_max_estimate, r_max_estimate)
        discretized_potential = [Potentials.ET2(r, cutoff, alpha, c_ET2, d_ET2) for r in grid.r_values]
        shift = Functions.compute_phase_shift(E, grid, boundary_condition, shooting_parameter, mass, alpha, discretized_potential, 2, enable_control_plots)
        ET2_phase_shifts.append(shift)
        
        print(f"{E}:  {shift}")


# ---------------- PLOTS OF THE RESULTS ---------------------

# plot of the potentials
bohr_radius = 1/(mass*alpha) # we use the bohr radius as length scale for the Coulombic interaction
grid = Grid(m.ceil(100*bohr_radius), 5*bohr_radius)

discretized_Synthetic = [Potentials.Coulomb_and_well(r, alpha, well_width, well_height) for r in grid.r_values] # plot the synthetic potential
plt.plot(grid.r_values, discretized_Synthetic, label="Synthetic potential")

if enable_ET1: # plot the potential of the effective theory at order 2
    discretized_ET1 = [Potentials.ET1(r, cutoff, alpha, c_ET1) for r in grid.r_values]
    plt.plot(grid.r_values, discretized_ET1, label=r"Effective potential at order $o(a^2)$")

if enable_ET2: # plot the potential of the effective theory at order 4
    discretized_ET2 = [Potentials.ET2(r, cutoff, alpha, c_ET2, d_ET2) for r in grid.r_values]
    plt.plot(grid.r_values, discretized_ET2, label=r"Effective potential at order $o(a^4)$")

plt.xlabel("r")
plt.ylabel("V(r)")
plt.ylim(-3*well_height, 0)
plt.legend()
plt.title('Radial potentials')
plt.savefig('./Figures/Radial_potentials.png', bbox_inches='tight')
plt.close()

# energies
coulomb_energies = [-pow(alpha, 2)*mass/(2*pow(n, 2)) for n in chosen_energy_levels]
plt.plot(np.abs(synthetic_energies)[:7], np.abs(np.divide(np.subtract(coulomb_energies,synthetic_energies),synthetic_energies))[:7], label="Coulomb potential", ls='--', marker='o', markerfacecolor='none')

if enable_naive_analysis: # energies from naive analysis
    plt.plot(np.abs(synthetic_energies)[:7], np.abs(np.divide(np.subtract(naive_energies,synthetic_energies),synthetic_energies))[:7], label=r"Coulomb + $\delta$ ($1^{st}$ order PT)", ls='--', marker='o', markerfacecolor='none')

if enable_ET1: # plot the energies coming from the effective theories at order 2
    plt.plot(np.abs(synthetic_energies)[:7], np.abs(np.divide(np.subtract(ET1_energies,synthetic_energies),synthetic_energies))[:7], label="ET at $o(a^2)$", ls='--', marker='o', markerfacecolor='none')

if enable_ET2: # plot the energies coming from the effective theories at order 4
    plt.plot(np.abs(synthetic_energies)[:7], np.abs(np.divide(np.subtract(ET2_energies,synthetic_energies),synthetic_energies))[:7], label="ET at $o(a^4)$", ls='--', marker='o', markerfacecolor='none')

plt.xscale('log'), plt.yscale('log')
plt.xlabel("E")
plt.ylabel(r"$|\Delta E/E|$")
plt.legend()
plt.title('Relative errors of the estimated energy values in the different approaches', fontsize = 9.5)
plt.savefig('./Figures/Errors_in_estimated_energy_values_all_methods.png', bbox_inches='tight')
plt.close()


# phase shifts
coulomb_phase_shifts = [np.angle(gamma(complex(1,-mass*alpha/np.sqrt(2 * mass * E)))) for E in chosen_scattering_energies]
plt.plot(chosen_scattering_energies, np.abs(np.subtract(coulomb_phase_shifts,synthetic_phase_shifts)), label="Coulomb potential", ls='-', marker='o')

if enable_ET1: # plot the phase shifts coming from the effective theory at order 2
    plt.plot(chosen_scattering_energies, np.abs(np.subtract(ET1_phase_shifts,synthetic_phase_shifts)), label="ET at o(a^2)", ls='-', marker='o')

if enable_ET2: # plot the phase shifts coming from the effective theory at order 4
    plt.plot(chosen_scattering_energies, np.abs(np.subtract(ET2_phase_shifts,synthetic_phase_shifts)), label="ET at o(a^4)", ls='-', marker='o')

plt.xscale('log'), plt.yscale('log')
plt.xlabel("E")
plt.ylabel(r"$|\Delta\delta(E)|$")
plt.legend()
plt.title('Relative errors of the estimated phase shifts in the different approaches', fontsize = 9.5)
plt.savefig('./Figures/Errors_in_estimated_phase_shifts_all_methods.png', bbox_inches='tight')
plt.close()



# ------------- ANALYSIS OF THE DEPENDENCE OF THE EFFECTIVE THEORY RESULTS ON THE CUTOFF -----------------
# here we use the effective theory at order o(a^2)

if enable_cutoff_dependence_analysis:
    print('\n--- ANALYSIS OF THE DEPENDENCE OF THE RESULTS ON THE CUTOFF FOR THE o(a^2) EFFECTIVE THEORY---')

    # computing the chosen set of energy levels for different values of the cutoff
    cutoff_values = [10 * well_width, well_width, 0.1*well_width, 0.01*well_width]
    formatted_cutoff_values = [float(format(value, '.3g')) for value in cutoff_values] #formatting the cutoff values to only three significant digits to avoid floating point representation issues
    c_values = []
    sets_of_energies = [] #list that contains the set of computed energy levels for the different values of the cutoff (so it's a list of lists)
    print('\nEstimated values for the constant c of the second order term for different values of the cutoff a')

    for cutoff in formatted_cutoff_values:
        # estimate value of c parameter
        c = Functions.optimize_ET1_with_energy(cutoff, alpha, mass, energy_bound, boundary_condition, shooting_parameter, synthetic_energies[8], chosen_energy_levels[8], enable_control_plots)
        print(f'a = {cutoff}   |   c = {c}')
        c_values.append(c)

        # compute energies
        energies = []

        for n in chosen_energy_levels:
            r_max_estimate = m.ceil(2 * pow(n, 2) / (mass * alpha) * 5 * (2 - m.erf(n - 3)))
            grid = Grid(20 * r_max_estimate, r_max_estimate)
            discretized_potential = [Potentials.ET1(r, cutoff, alpha, c) for r in grid.r_values]
            state = Functions.solve_for_bound_state(n - 1, discretized_potential, grid, energy_bound, boundary_condition, shooting_parameter, mass)
            energies.append(state[0])

        sets_of_energies.append(energies)

    # plotting the results

    grey_palette = np.linspace(0, 0.7, len(formatted_cutoff_values))
    for i in range(len(formatted_cutoff_values)):
        plt.plot(np.abs(synthetic_energies)[:7],np.abs(np.divide(np.subtract(sets_of_energies[i], synthetic_energies), synthetic_energies))[:7], label=f"a = {formatted_cutoff_values[i]}", ls='--', marker='o', markerfacecolor='none', color=f'{grey_palette[i]}')

    plt.xscale('log'), plt.yscale('log')
    plt.xlabel("E")
    plt.ylabel(r"$|\Delta E/E|$")
    plt.legend()
    plt.title(r'Relative errors of the estimated energy values for the $o(a^2)$ effective theory at different values of a', fontsize=8.5)
    plt.savefig('./Figures/Cutoff_dependence_of_errors_in_estimated_energy_values.png', bbox_inches='tight')
    plt.close()

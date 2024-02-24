
'''
FILE THAT CONTAINS THE DEFINITION OF THE ROUTINES NECESSARY TO RUN THE Lepage_Analysis.py SCRIPT
THE MOST RELEVANT ROUTINES ARE:
- the radial Shrodinger equation solver for l=0 bound states
- the routine to compute phase shifts for l=0 scattering states in presence of a Coulombic interaction + a generic radial potential
- the routines to estimate the parameters of the effective theory at order 2 and 4 of the cutoff using binding energies data
ALL OTHER ROUTINES ARE AUXILIARY TO THESE MAIN ONES


Notes:
- when using the rotine to solve for the bound states use always a grid with an even number of points
  (required since we use Simpson method for integration when normalizing the wavefunctions)
'''

import sys

import numpy as np
import math as m
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, root, brentq
from scipy.special import gamma

from Classes import Grid
from Potentials import ET1, ET2

# --------------------- AUXILIARY ROUTINES -------------------------------------------------------------------

# formula to compute the r index to use as stop value for the shooting
# corresponds to the closest point on the right of the rightmost turning point (it may be the turning point itself)
def compute_shooting_limit (discretized_potential, energy):
    shooting_limit = 0
    signs = np.sign(energy-discretized_potential)
    for i in range(len(discretized_potential)-1, 0, -1):
        if (signs[i-1])*(signs[i]) <= 0:
            shooting_limit = i
            break
    return shooting_limit

#numerov shooting formula that shoots forward up to a certain index value for the discretized r
def numerov_forward_shooting (wavefunction, numerov_auxiliary_function, shooting_limit):
    for i in range(len(wavefunction)):
        if i>shooting_limit-2:
            break
        wavefunction[i+2]=(2*wavefunction[i+1]*(1-5*numerov_auxiliary_function[i+1])-wavefunction[i]*(1+numerov_auxiliary_function[i]))/(1+numerov_auxiliary_function[i+2])
    return wavefunction

#numerov shooting formula that shoots backward up to a certain index value for the discretized r
def numerov_backward_shooting (wavefunction, numerov_auxiliary_function, shooting_limit):
    for i in range(len(wavefunction)-1, -1, -1):
        if i<shooting_limit+2:
            break
        wavefunction[i-2] = (2*wavefunction[i-1]*(1-5*numerov_auxiliary_function[i-1])-wavefunction[i]*(1+numerov_auxiliary_function[i]))/(1+numerov_auxiliary_function[i-2])
    return wavefunction

#function that counts the number of nodes of a (discretized) wavefunction
def check_n_nodes (wavefunction, shooting_limit):
    n_nodes = 0
    signs = np.sign(wavefunction)
    for i in range(1, shooting_limit-1): # the range starts from 1 to avoid considering the origin of the radial axis
        if signs[i]*signs[i+1] <= 0 and signs[i]!=0: #the second condition is to avoid double counting a node if it happens precisely on a point of the grid
            n_nodes += 1
    return n_nodes

#compute approximate difference between true energy of a wavefunction and its estimate using Cooley's formula
def compute_energy_difference(wavefunction, numerov_auxiliary_function, mass, grid, shooting_limit):
    coefficient = wavefunction[shooting_limit] / (2 * mass * pow(grid.delta_r, 2) * sum([x**2 for x in wavefunction]))
    energy_difference = coefficient * (2*(1-5*numerov_auxiliary_function[shooting_limit])*wavefunction[shooting_limit]-(1+numerov_auxiliary_function[shooting_limit+1])*wavefunction[shooting_limit+1]-(1+numerov_auxiliary_function[shooting_limit-1])*wavefunction[shooting_limit-1])
    return energy_difference

def normalise_wavefunction(wavefunction, grid): # integration via Simpson method
    integrand_function = [x**2 for x in wavefunction]
    normalisation = 1/3 * grid.delta_r * (integrand_function[0] + integrand_function[grid.number_of_points-1] + 2 * np.sum([integrand_function[i] for i in range(0, grid.number_of_points, 2)]) + 4 * np.sum([integrand_function[i] for i in range(1, grid.number_of_points, 2)]))
    wavefunction = wavefunction / np.sqrt(normalisation)
    return wavefunction


# --------------------- MOST IMPORTANT ROUTINES -----------------------------------------------------------------

# routine that finds the energy eigenfunction and its energy level with estimated error for the bound states
def solve_for_bound_state (n_nodes, discretized_potential, grid, energy_bound, boundary_condition, shooting_parameter, mass):

    E_max = discretized_potential[grid.number_of_points-1]
    E_min = min(discretized_potential)

    skip = False
    max_n_of_iterations = 10000

    wavefunction = []
    energy = 0.
    energy_difference = 0.

    for counter in range(max_n_of_iterations):

        #step 0: create and initialize left wavefunction
        left_wavefunction = [0.] * grid.number_of_points
        left_wavefunction[0] = boundary_condition
        left_wavefunction[1] = shooting_parameter

        #step 1: compute energy estimate
        energy = (E_max + E_min) / 2

        #step 2: shoot forward
        numerov_auxiliary_function = [mass * pow(grid.delta_r, 2) * (energy - V) / 6 for V in discretized_potential]
        shooting_limit = compute_shooting_limit(discretized_potential, energy)

        left_wavefunction = numerov_forward_shooting(left_wavefunction, numerov_auxiliary_function, shooting_limit)

        #step 3: check number of nodes and if necessary correct energy estimate
        computed_n_nodes = check_n_nodes(left_wavefunction, shooting_limit)

        if n_nodes == 0:
        # if the required number of nodes is zero we can't narrow the energy range within we are serching for the bound state energy 
        # as we do for the other cases since immediately computed_n_nodes == n_nodes holds true.
        # To have a good candidate for the energy we then start going up with E_min until the energy estimate is no more in the range of the ground state energy, 
        # then we use that value for E_max, while for E_min we revert back to the value of the step before.
        # At ths point, we lower E_max down until the energy estimate is again in the range where computed_n_nodes=0, and proceed to use Cooley's forumla to better approximate our energy estimate
            if computed_n_nodes == n_nodes and skip == False:
                E_min = energy
                continue
            elif computed_n_nodes > n_nodes:
                if skip == False and E_min != min(discretized_potential):
                    E_min = 2*E_min - E_max
                E_max = energy
                skip = True
                continue

        if n_nodes != 0:
            if computed_n_nodes < n_nodes:
                E_min = energy
                continue
            elif computed_n_nodes > n_nodes:
                E_max = energy
                continue
            elif computed_n_nodes == n_nodes:
                pass


        #step 4: initialize right wavefunction, shoot backwards, impose continuity and combine left&right parts of the wavefunction
        right_wavefunction = [0.] * grid.number_of_points
        right_wavefunction[grid.number_of_points - 1] = boundary_condition
        right_wavefunction[grid.number_of_points - 2] = pow(-1, n_nodes) * shooting_parameter

        right_wavefunction = numerov_backward_shooting(right_wavefunction, numerov_auxiliary_function, shooting_limit)
        right_wavefunction = [x * left_wavefunction[shooting_limit]/right_wavefunction[shooting_limit] for x in right_wavefunction]
        wavefunction = left_wavefunction[:shooting_limit] + right_wavefunction[shooting_limit:]

        #step 5: check if energy bound is satisfied and otherwise correct the energy guess using Cooley's formula
        energy_difference = compute_energy_difference(wavefunction, numerov_auxiliary_function, mass, grid, shooting_limit)

        if abs(energy_difference/energy) <= energy_bound:
            energy += energy_difference
            break
        else: # we change the values of E_min and E_max so that their average corresponds to the updated guess for the energy
            E_min += energy_difference
            E_max += energy_difference

        if counter == max_n_of_iterations - 1:
            print('''The routine to solve for bound states has reached the maximal number of iterations.
            The requested accuracy has not been obtained. Perhaps try to modify the parameters of the problem.''')


    #step 6: normalise the wavefunction
    wavefunction = normalise_wavefunction(wavefunction, grid)

    return [energy, energy_difference, wavefunction]

def compute_phase_shift(energy, grid, boundary_condition, shooting_parameter, mass, alpha, discretized_potential, option = 0, enable_plots = False):

    left_wavefunction = [0.] * grid.number_of_points
    left_wavefunction[0] = boundary_condition
    left_wavefunction[1] = shooting_parameter

    numerov_auxiliary_function = [mass * pow(grid.delta_r, 2) * (energy - V) / 6 for V in discretized_potential]
    shooting_limit = grid.number_of_points - 1
    left_wavefunction = numerov_forward_shooting(left_wavefunction, numerov_auxiliary_function, shooting_limit)

    wavenumber = np.sqrt(2 * mass * energy)
    Sommerfeld_parameter = - mass * alpha / wavenumber

    left_r_bound = grid.r_max - 5 * (2*np.pi)/wavenumber #for the fit, we restrict to the rightmost region of the r range (to simulate the r >> 1 condition), ensuring to retain approximately five oscillations of the wavefunction
    if left_r_bound <= 0:
        sys.exit("Warning from the routine to compute the phase shifts. The r_max selected for the computation is too small. The code will stop.")

    restricted_r_values = [r for r in grid.r_values if r > left_r_bound]
    restricted_wavefunction = left_wavefunction[len(grid.r_values)-len(restricted_r_values):]

    def fit_function(r, phase_shift, constant):
        return constant * np.sin(wavenumber * r - Sommerfeld_parameter * np.log(2*wavenumber*r) + phase_shift)

    parameters, cov = curve_fit(fit_function, restricted_r_values, restricted_wavefunction, [np.angle(gamma(complex(1,Sommerfeld_parameter))),max(restricted_wavefunction)])

    if abs(parameters[0]) > 2*np.pi: #adjust for polidromy of phase
        parameters[0] = m.fmod(parameters[0], 2*np.pi)

    if enable_plots:
        plt.plot([fit_function(r, parameters[0], parameters[1]) for r in restricted_r_values])
        plt.plot(restricted_wavefunction)
        plt.show()

    match option:
        case 0:
            return [parameters, cov]
        case 1:
            return parameters
        case 2:
            return parameters[0]
        case _:
            print("Warning from the routine to compute the phase shifts. The chosen option for the output is not valid")


def optimize_ET1_with_energy(cutoff, alpha, mass, energy_bound, boundary_condition, shooting_parameter, synthetic_energy, quantum_number, enable_plots = False):
    r_max_estimate = m.ceil(2*pow(quantum_number,2)/(mass*alpha) * 5 * (2-m.erf(quantum_number-3)))
    grid = Grid(20*r_max_estimate, r_max_estimate)

    def function_to_minimize(c):
        discretized_potential = [ET1(r, cutoff, alpha, c) for r in grid.r_values]
        state = solve_for_bound_state(quantum_number-1, discretized_potential, grid, energy_bound, boundary_condition, shooting_parameter, mass)
        return (synthetic_energy - state[0])/abs(synthetic_energy)

    #bracket the root (expanding the range of values for c)
    c_min = -1
    c_max = 1
    maxiter = 100
    f1 = function_to_minimize(c_min)
    f2 = function_to_minimize(c_max)
    for i in range(maxiter):
        if f1*f2 < 0. :
            break
        elif abs(f1) < abs(f2):
            c_min += (c_min -c_max)
            f1 = function_to_minimize(c_min)
        else:
            c_max += (c_max -c_min)
            f1 = function_to_minimize(c_max)

        if i == maxiter -1:
            sys.exit("Warning from the routine to optimize the parameter of the effective theory at order 2. The root bracketing operation has failed in the number of iterations required.")

    root, result = brentq(function_to_minimize, c_min, c_max, full_output=True)

    if result.converged == True:
        if enable_plots:
            # plot to verify graphically that the estimated value for c is effectively a root
            c_values = np.linspace(root-1, root+1, 21)
            f_values = [function_to_minimize(c) for c in c_values]
            plt.axvline(x=root, color='gray', linewidth=1), plt.axhline(y=0, color='gray', linewidth=1)
            plt.plot(c_values, f_values, '.-b')
            plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
            plt.ylabel(f'ΔE/|E| ({quantum_number}S)')
            plt.xlabel('c')
            plt.title('Check that the estimated value for c is (locally) the best to reproduce the lowest binding energy', fontsize=8)
            plt.show()
        return root
    else:
        print("Warning from the routine to optimize the parameter of the effective theory at order 2. The root finding algorithm has not succeded.", result.flag)
        sys.exit()


def optimize_ET2_with_energy(cutoff, alpha, mass, energy_bound, boundary_condition, shooting_parameter, synthetic_energy_1, quantum_number_1, synthetic_energy_2, quantum_number_2, guess, enable_plots = False):
    r_max_estimate_1 = m.ceil(2*pow(quantum_number_1,2)/(mass*alpha) * 5 * (2-m.erf(quantum_number_1-3)))
    grid_1 = Grid(20*r_max_estimate_1, r_max_estimate_1)
    r_max_estimate_2 = m.ceil(2*pow(quantum_number_2,2)/(mass*alpha) * 5 * (2-m.erf(quantum_number_2-3)))
    grid_2 = Grid(20*r_max_estimate_2, r_max_estimate_2)

    def constraints(parameters):
        discretized_potential = [ET2(r, cutoff, alpha, parameters[0], parameters[1]) for r in grid_1.r_values]
        state_1 = solve_for_bound_state(quantum_number_1-1, discretized_potential, grid_1, energy_bound, boundary_condition,
                                        shooting_parameter, mass)

        discretized_potential = [ET2(r, cutoff, alpha, parameters[0], parameters[1]) for r in grid_2.r_values]
        state_2 = solve_for_bound_state(quantum_number_2-1, discretized_potential, grid_2, energy_bound, boundary_condition,
                                        shooting_parameter, mass)

        return [(synthetic_energy_1 - state_1[0])/abs(synthetic_energy_1), (synthetic_energy_2 - state_2[0])/abs(synthetic_energy_2)]

    result = root(constraints, guess)

    if result.success == True:
        if enable_plots:
            # plot to verify graphically that the estimated values for c and d are effectively a root
            a = np.arange(result.x[0]-2, result.x[0]+2, 1)
            b = np.arange(result.x[1]-2, result.x[1]+2, 1)

            data1 = np.zeros((len(a), len(b)))
            data2 = np.zeros((len(a), len(b)))
            for i in range(len(a)):
                for j in range(len(b)):
                    params = [a[i], b[j]]
                    sol = constraints(params)
                    data1[i][j] = sol[0]
                    data2[i][j] = sol[1]

            plt.contourf(a, b, data1)
            plt.colorbar()
            plt.axvline(x=result.x[0], color='darkslategrey', linewidth=1), plt.axhline(y=result.x[1], color='darkslategrey', linewidth=1)
            plt.scatter(result.x[0], result.x[1], marker='x', color='darkslategrey')
            plt.xlabel('c'), plt.ylabel('d')
            plt.title(f'Check that the estimated values for c and d are (locally) the best to reproduce E({quantum_number_1}S)', fontsize=8.5)
            plt.show()

            plt.contourf(a, b, data2)
            plt.colorbar()
            plt.axvline(x=result.x[0], color='darkslategrey', linewidth=1), plt.axhline(y=result.x[1], color='darkslategrey', linewidth=1)
            plt.scatter(result.x[0], result.x[1], marker='x', color='darkslategrey')
            plt.xlabel('c'), plt.ylabel('d')
            plt.title(f'Check that the estimated values for c and d are (locally) the best to reproduce E({quantum_number_2}S)', fontsize=8.5)
            plt.show()
        return result.x
    else:
        print("Warning from the routine to optimize the parameters of the effective theory at order 4. The root finding algorithm has not succeded.",
              result.message)
        sys.exit()


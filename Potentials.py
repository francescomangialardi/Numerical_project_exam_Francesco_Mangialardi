
'''
FILE THAT CONTAINS THE DEFINITIONS OF THE RADIAL POTENTIALS RELEVANT FOR THE PROJECT
EACH POTENTIAL IS DEFINED AS A FUNCTION OF r AND OF ITS PARAMETERS
TO USE A NEW TYPE OF POTENTIAL IN THE PROJECT'S ANALYSIS ADD IT HERE RESPECTING THE ABOVE PRESCRIPTION.
'''

import math as m
import numpy as np

# auxiliary functions (used to define the potentials)

def smeared_delta (r, cutoff):
    return m.exp(-r**2/(2*cutoff**2)) / (pow(2*np.pi, 3/2)*pow(cutoff, 3))

def laplacian_smeared_delta (r, cutoff):
    return (-3/pow(cutoff, 2) + r**2/pow(cutoff, 4)) * smeared_delta(r,cutoff)


# ---------- POTENTIALS ------------

def Coulomb (r, alpha): #Coulomb potential
    return -alpha/r

def Harmonic_Oscillator_3D (r, mass, omega): #three dimensional isotropic harmonic oscillator
    return 1/2 * mass * (omega*r)**2

def Coulomb_and_well (r, alpha, well_width, well_height): #Coulomb potential plus a spherical square well
    return -alpha/r - well_height * np.heaviside(well_width - r, 1)

def ET1 (r, cutoff, alpha, c): #Effective theory at order 2 of the cutoff
    return - alpha/r * m.erf(r/(np.sqrt(2)*cutoff)) + c * pow(cutoff, 2) * smeared_delta(r, cutoff)

def ET2 (r, cutoff, alpha, c, d): #Effective theory at order 4 of the cutoff, in the simplified version valid for S states
    return - alpha / r * m.erf(r / (np.sqrt(2) * cutoff)) + c * pow(cutoff, 2) * smeared_delta(r, cutoff) + d * pow(cutoff, 4) * laplacian_smeared_delta(r, cutoff)



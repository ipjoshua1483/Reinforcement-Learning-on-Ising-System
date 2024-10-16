import numpy as np
import matplotlib.pyplot as plt

import time

'''
This is code to initalize an Ising Model based on Stephen Whitelam's maxwell demon paper, with updates
to make it run faster for learning the protocol mainly through changing the energy calculation function. 
'''

def initialize_lattice(N):
    """Initialize a square lattice with all spins down (-1)."""
    lattice = np.ones((N, N), dtype=int) * -1
    return lattice



def calculate_energy(lattice, J, h):
    N = lattice.shape[0]
    
    # Term due to the magnetic field
    energy_field = -h * np.sum(lattice)

    # Term due to the nearest-neighbor coupling, considering periodic boundary conditions
    energy_coupling = -0.5 * J * (
        np.sum(lattice * np.roll(lattice, shift=-1, axis=0)) +
        np.sum(lattice * np.roll(lattice, shift=1, axis=0)) +
        np.sum(lattice * np.roll(lattice, shift=-1, axis=1)) +
        np.sum(lattice * np.roll(lattice, shift=1, axis=1))
    )

    return energy_field + energy_coupling



def calculate_delta_energy(i, j, lattice, J, h):
    N = lattice.shape[0]
    cur_spin = lattice[i, j]

    # Using periodic boundary conditions for neighbors
    neighbors = [
        lattice[(i - 1) % N, j],
        lattice[i, (j - 1) % N],
        lattice[(i + 1) % N, j],
        lattice[i, (j + 1) % N]
    ]

    sum_of_neighbors = np.sum(neighbors)
    
    deltaE = 2 * J * cur_spin * sum_of_neighbors
    deltaE += 2 * h * cur_spin # Adding the magnetic field contribution

    return deltaE, cur_spin



def glauber_step_rand(lattice, J_short, h, beta):
    N = lattice.shape[0]
    i = np.random.randint(N)
    j = np.random.randint(N)

    delta_energy, spin = calculate_delta_energy(i, j, lattice, J_short, h)
    delta_energy = np.clip(delta_energy, a_min=None, a_max=700/beta)

    glauber_prob = 1.0 / (1.0 + np.exp(beta * delta_energy))

    if np.random.random() < glauber_prob:
        lattice[i, j] = -spin

    return lattice

def simulate_ising_model_rand(N, J_short, h, beta, num_sweeps, lattice=None):
    """Simulate magnetization reversal in the Ising model using Glauber Monte Carlo dynamics."""
    if lattice is None:
        lattice = initialize_lattice(N)

    num_steps = N**2 * num_sweeps

    for step in range(int(num_steps)):
        lattice = glauber_step_rand(lattice, J_short, h, beta)

    # Calculate final magnetization and energy
    magnetization = np.mean(lattice)
    energy = calculate_energy(lattice, J_short, h)

    return lattice, energy, magnetization


def glauber_step(lattice, J_short, h, beta, i, j):
    delta_energy, spin = calculate_delta_energy(i, j, lattice, J_short, h)
    #delta_energy = np.clip(delta_energy, a_min=None, a_max=700/beta)
    glauber_prob = 1.0 / (1.0 + np.exp(beta * delta_energy))

    if np.random.random() < glauber_prob:
        lattice[i, j] = -spin
        entprod = beta*delta_energy
    else:
        entprod = 0
        delta_energy = 0

    return lattice, entprod, delta_energy

def linear_idx_to_coordinates(k, N):
    """
    k gives the linear index in [0, N*N-1].

    This function will give the (i, j) coordinates in the 2D array.
    """
    i = k // N
    j = k % N
    return i, j

def simulate_ising_model(N, J_short, h, beta, num_sweeps, lattice=None):
    """Simulate magnetization reversal in the Ising model using Glauber Monte Carlo dynamics."""
    if lattice is None:
        lattice = initialize_lattice(N)
    total_entprod_step = 0
    delta_nrg_arr = 0
    for sweep in range(int(num_sweeps)):
        # First, sweep over grid points with even i-index
        for k in range(N*N):
            if (k % 2) == 0:
                i, j = linear_idx_to_coordinates(k, N)
                lattice, entprod_1, delta_energy = glauber_step(lattice, J_short, h, beta, i, j)
                total_entprod_step += entprod_1
                delta_nrg_arr += delta_energy
            
        for k in range(N*N):
            if (k % 2) == 1:
                i, j = linear_idx_to_coordinates(k, N)
                lattice, entprod_1, delta_nrg = glauber_step(lattice, J_short, h, beta, i, j)
                total_entprod_step += entprod_1
                delta_nrg_arr += delta_nrg

    # Calculate final magnetization and energy
    magnetization = np.mean(lattice)
    energy = calculate_energy(lattice, J_short, h)
    return lattice, energy, magnetization, total_entprod_step, delta_nrg_arr
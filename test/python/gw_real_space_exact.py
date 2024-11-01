from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs_tprf.lattice import inv

import numpy as np
from itertools import product

from triqs_tprf.gw_solver_real_space import GWSolver


def generate_g0_w(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij
    g = inv(g_inv)
    return BlockGf(block_list = [g] * 2, name_list = spin_names, make_copies = False)

def exact_polarization(mesh, t):
    g = Gf(mesh = mesh, target_shape = [2, 2])

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = (-1) ** (i - j) * (1 / (w - 2 * t) - 1 / (w + 2 * t)) / 4
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)

def exact_screened_potential(mesh, t, U):
    g = Gf(mesh = mesh, target_shape = [2, 2])

    h_sqrd = 4 * t ** 2 + 4 * U * t

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = U * int(i == j) + (-1) ** (i - j) * 2 * U ** 2 * t / (complex(w) ** 2 - h_sqrd)
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)


def exact_self_energy(mesh, t, U):
    g = Gf(mesh = mesh, target_shape = [2, 2])

    h_sqrd = 4 * t ** 2 + 4 * U * t
    h = np.sqrt(h_sqrd)

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = U * int(i == j) / 2 + U ** 2 * t * (1 / (w - t - h) + (-1) ** (i - j) / (w + t + h)) / (2 * h)
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)


def exact_g(mesh, t, U, interacting = True):
    u = U * int(interacting)
    g = Gf(mesh = mesh, target_shape = [2, 2])

    h_sqrd = 4 * t ** 2 + 4 * u * t
    h = np.sqrt(h_sqrd)

    A = np.sqrt((2 * t + h + u / 2) ** 2 + 4 * u ** 2 * t / h)
    B = np.sqrt((2 * t + h - u / 2) ** 2 + 4 * u ** 2 * t / h)

    w1p = 0.5 * (u / 2 - h + A)
    w1m = 0.5 * (u / 2 - h - A)
    w2p = 0.5 * (u / 2 + h + B)
    w2m = 0.5 * (u / 2 + h - B)

    indices = (0, 1)
    for i, j in product(indices, indices):
        for w in mesh:
            g[w][i, j] = (-1) ** (i - j) * ((0.25 + (h + 2 * t + u / 2) / (4 * A)) / (w - w1p) + (0.25 - (h + 2 * t + u / 2) / (4 * A)) / (w - w1m)) \
                         + (0.25 + (-h - 2 * t + u/2) / (4 * B)) / (w - w2p) + (0.25 - (-h - 2 * t + u/2) / (4 * B)) / (w - w2m)
 
    return BlockGf(block_list = [g] * 2, name_list = ['up', 'dn'], make_copies = False)



def test_gw_exact():

    """
    Comparing to analytical expressions from:
    Chapter 4: Hubbard Dimer in GW and Beyond, by Pina Romaniello

    In the book:
    Simulating Correlations with Computers - Modeling and Simulation Vol. 11
    E. Pavarini and E. Koch (eds.)
    Forschungszentrum Ju ̈lich, 2021, ISBN 978-3-95806-529-1
    
    https://www.cond-mat.de/events/correl21/manuscripts/correl21.pdf    
    """
    
    U = 4.0
    Vij = np.eye(2) * U

    t = 1.0
    tij = np.array([[0, -t],
                    [-t, 0]])

    beta = 50
    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion', w_max = 20.0, eps = 1e-12, symmetrize = True)
    iw_mesh_b = MeshDLRImFreq(beta = beta, statistic = 'Boson', w_max = 20.0, eps = 1e-12, symmetrize = True)

    g0_w = generate_g0_w(tij, iw_mesh_f, spin_names = ['up', 'dn'])
    gw = GWSolver(g0_w, Vij, hartree = True, fock = True, gw = True, spinless = True)


    # Non-interacting Green's function
    g0_exact = exact_g(iw_mesh_f, t, U, interacting = False)
    np.testing.assert_allclose(g0_exact['up'].data[:], g0_w['up'].data[:])
    np.testing.assert_allclose(g0_exact['dn'].data[:], g0_w['dn'].data[:])

    # Polarization
    P_exact = exact_polarization(iw_mesh_b, t)
    P_gw = gw.P
    np.testing.assert_allclose(P_exact['up'].data[:], P_gw['up'].data[:])
    np.testing.assert_allclose(P_exact['dn'].data[:], P_gw['dn'].data[:])

    # Screened potential
    W_exact = exact_screened_potential(iw_mesh_b, t, U)
    W_gw = gw.W
    np.testing.assert_allclose(W_exact['up'].data[:], W_gw['up'].data[:])
    np.testing.assert_allclose(W_exact['dn'].data[:], W_gw['dn'].data[:])

    # Self-energy
    sigma_exact = exact_self_energy(iw_mesh_f, t, U)
    sigma_gw = gw.sigma
    np.testing.assert_allclose(sigma_exact['up'].data[:], sigma_gw['up'].data[:])
    np.testing.assert_allclose(sigma_exact['dn'].data[:], sigma_gw['dn'].data[:])

    # Interacting Green's function
    g_exact = exact_g(iw_mesh_f, t, U, interacting = True)
    g_gw = gw.g_w
    np.testing.assert_allclose(g_exact['up'].data[:], g_gw['up'].data[:])
    np.testing.assert_allclose(g_exact['dn'].data[:], g_gw['dn'].data[:])

if __name__ == '__main__':
    test_gw_exact()
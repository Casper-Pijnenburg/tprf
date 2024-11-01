from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs_tprf.lattice import inv

from h5 import HDFArchive

import numpy as np
from itertools import product

from triqs_tprf.gw_solver_real_space import GWSolver

from triqs_tprf.utilities import read_TarGZ_HDFArchive

def generate_g0_w(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij.transpose()
    g = inv(g_inv)
    return BlockGf(block_list = [g] * 2, name_list = spin_names, make_copies = False)


def coulomb_matrix(orbitals, U, delta = 1, non_local = True):
    Vij = np.zeros([orbitals] * 2)
    for i in range(orbitals):
        for j in range(orbitals):
            Vij[i, j] = delta * U / np.sqrt(abs(i - j) ** 2 + delta ** 2)

        Vij[i, i] = U
    if non_local:
        return Vij

    return np.diag(Vij.diagonal())

def test_nonlocal():

    """
    Comparing a system with non-local interactions 
    """
    

    U = 1.0
    Vij = coulomb_matrix(2, U, delta = 1, non_local = True)

    t = 1.0
    tij = np.array([[0, -t],
                    [-t, 0]])

    beta = 50
    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion', w_max = 40.0, eps = 1e-12, symmetrize = True)
    g0_w = generate_g0_w(tij, iw_mesh_f)

    gw = GWSolver(g0_w, Vij, maxiter = 50)

    gw_benchmark = read_TarGZ_HDFArchive('gw_real_space_non_local_test_data.tar.gz')

    np.testing.assert_allclose(gw_benchmark['P']['up'].data[:], gw.P['up'].data[:])
    np.testing.assert_allclose(gw_benchmark['P']['dn'].data[:], gw.P['dn'].data[:])

    np.testing.assert_allclose(gw_benchmark['W']['up'].data[:], gw.W['up'].data[:])
    np.testing.assert_allclose(gw_benchmark['W']['dn'].data[:], gw.W['dn'].data[:])

    np.testing.assert_allclose(gw_benchmark['S']['up'].data[:], gw.sigma['up'].data[:])
    np.testing.assert_allclose(gw_benchmark['S']['dn'].data[:], gw.sigma['dn'].data[:])

    np.testing.assert_allclose(gw_benchmark['G']['up'].data[:], gw.g_w['up'].data[:])
    np.testing.assert_allclose(gw_benchmark['G']['dn'].data[:], gw.g_w['dn'].data[:])



    return


if __name__ == '__main__':
    test_nonlocal()
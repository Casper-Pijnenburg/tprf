from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs_tprf.lattice import inv

import numpy as np
from itertools import product

from triqs_tprf.gw_solver_real_space import GWSolver as GWSolverRS
from triqs_tprf.gw_solver import GWSolver as GWSolverMom

from triqs.lattice.tight_binding import TBLattice


def generate_g0_w(tij, mesh, spin_names = ['up', 'dn']):
    g_inv = Gf(mesh = mesh, target_shape = np.shape(tij))
    g_inv << iOmega_n - tij.transpose()
    g = inv(g_inv)
    return BlockGf(block_list = [g] * 2, name_list = spin_names, make_copies = False)

class GWHubbardDimer:

    def __init__(
            self,
            beta=20.0, U=1.5, t=1.0, mu=0.0, nw=1024, maxiter=100,
            self_interaction=False, spinless=False,
            gw_flag=True, hartree_flag=False, fock_flag=False):
        
        wmesh = MeshImFreq(beta, 'Fermion', nw)

        if spinless:
            tb_opts = dict(
                units = [(1, 0, 0)],
                orbital_positions = [(0,0,0)],
                orbital_names = ['0'],
                )
            I = np.eye(1)
            
        else:
            tb_opts = dict(
                units = [(1, 0, 0)],
                orbital_positions = [(0,0,0)] * 2,
                orbital_names = ['up_0', 'do_0'],
                )
            I = np.eye(2)

        H_r = TBLattice(hopping = {
            (+1,): -0.5 * t * I,
            (-1,): -0.5 * t * I,
            }, **tb_opts)

        kmesh = H_r.get_kmesh(n_k=(2, 1, 1))
        self.e_k = H_r.fourier(kmesh)

        if self_interaction:
            
            V_aaaa = np.zeros((2, 2, 2, 2))

            V_aaaa[0, 0, 0, 0] = U
            V_aaaa[1, 1, 1, 1] = U
            
            V_aaaa[1, 1, 0, 0] = U
            V_aaaa[0, 0, 1, 1] = U

            self.V_aaaa = V_aaaa
            
        if spinless:

            V_aaaa = np.zeros((1, 1, 1, 1))
            V_aaaa[0, 0, 0, 0] = U
            self.V_aaaa = V_aaaa
            
        if not spinless and not self_interaction:
            
            from triqs.operators import n, c, c_dag, Operator, dagger
            from triqs_tprf.gw import get_gw_tensor
            
            self.H_int = U * n('up',0) * n('do',0)
            self.fundamental_operators = [c('up', 0), c('do', 0)]
            self.V_aaaa = get_gw_tensor(self.H_int, self.fundamental_operators)
        
        self.V_k = Gf(mesh=kmesh, target_shape=self.V_aaaa.shape)
        self.V_k.data[:] = self.V_aaaa    

        gw = GWSolverMom(self.e_k, self.V_k, wmesh, mu=mu)
        gw.solve_iter(
            maxiter=maxiter,
            gw=gw_flag, hartree=hartree_flag, fock=fock_flag,
            spinless=spinless)
        gw.calc_real_space()
        
        self.gw = gw

        for key, val in gw.__dict__.items():
            setattr(self, key, val)


def test_difference(maxiter, fock_flag, hartree_flag):
    U = 4.0
    Vij = np.eye(2) * U
    t = 1.0
    tij = np.zeros([2] * 2)
    for i in range(2 - 1):
        tij[i, i + 1] = -t
        tij[i + 1, i] = -t

    Vij = np.eye(2) * U

    beta = 50
    iw_mesh_f = MeshDLRImFreq(beta = beta, statistic = 'Fermion', w_max = 40.0, eps = 1e-12, symmetrize = True)

    g0_w = generate_g0_w(tij, iw_mesh_f)

    gw_flag = True
    gw = GWSolverRS(g0_w, Vij, hartree = hartree_flag, fock = fock_flag, gw = gw_flag, spinless = True, maxiter = maxiter)


    gw_momentum = GWHubbardDimer(
        beta = beta, U = U, 
        t = t, nw = 5000, maxiter = maxiter,
        self_interaction = True, spinless = False,
        hartree_flag = hartree_flag, fock_flag = fock_flag, gw_flag = gw_flag)

    dlr = make_gf_dlr(gw.g_w)
    g_rs = make_gf_imfreq(dlr, 5000)

  
    np.testing.assert_allclose(gw_momentum.g_wr[:, Idx(0, 0, 0)][0, 0].data, g_rs['up'].data[:, 0, 0], rtol = 1e-5)
    np.testing.assert_allclose(gw_momentum.g_wr[:, Idx(1, 0, 0)][0, 0].data, g_rs['up'].data[:, 0, 1], rtol = 1e-5)

    np.testing.assert_allclose(gw_momentum.g_wr[:, Idx(0, 0, 0)][0, 0].data, g_rs['dn'].data[:, 0, 0], rtol = 1e-5)
    np.testing.assert_allclose(gw_momentum.g_wr[:, Idx(1, 0, 0)][0, 0].data, g_rs['dn'].data[:, 0, 1], rtol = 1e-5)

    return


def test_momentum():

    """
    Comparing to tprf's momentum-spcae GW solver for different self-energy flags and max. GW cycle iterations
    """
     
    tf = (True, False)
    for maxiter in range(1, 5):
        for hartree, fock in product(tf, tf):
            test_difference(maxiter, fock, hartree)


if __name__ == '__main__':
    test_momentum()


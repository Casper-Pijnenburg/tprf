import numpy as np

from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
from triqs.gf import make_gf_dlr, make_gf_dlr_imtime, make_gf_dlr_imfreq

from triqs_tprf.lattice import polarization
from triqs_tprf.lattice import screened_potential
from triqs_tprf.lattice import dyn_self_energy
from triqs_tprf.lattice import hartree_self_energy
from triqs_tprf.lattice import fock_self_energy
from triqs_tprf.lattice import dyson_mu
from triqs_tprf.lattice import dyson_mu_sigma
from triqs_tprf.lattice import total_density
from triqs_tprf.lattice import inv

from itertools import product
        
class GWSolver():
    def __init__(self, g0_w, V, hartree = True, fock = True, gw = True, spinless = False, N_fix = False, N_tol = 1e-5, maxiter = 1, tol = 1e-5):
    
        self.N_fix, self.N_tol = N_fix, N_tol
        self.maxiter, self.tol = maxiter, tol
        self.g0_w = g0_w
        self.V = V
        self.hartree, self.fock, self.gw, self.spinless = hartree, fock, gw, spinless

        self.blocks = [name for name, g in self.g0_w]
        self.target_shape = self.g0_w[self.blocks[0]][Idx(0)].shape

        self.fmesh = self.g0_w.mesh
        self.bmesh = MeshDLRImFreq(self.fmesh.beta, 'Boson', w_max = self.fmesh.w_max, eps = self.fmesh.eps, symmetrize = True)
 

        self.g0_w, self.mu0 = self.dyson_equation(self.g0_w, 0.0, sigma_w = None, N_fix = self.N_fix)

        self.g_w = self.g0_w.copy()
        self.g_w_old = self.g_w.copy()


        self.sigma = self.g0_w.copy()

        for _ in range(maxiter):
            self.sigma.zero()

            if self.hartree:
                self.sigma += self.hartree_self_energy(self.g_w, self.V)
            if self.fock:
                self.sigma += self.fock_self_energy(self.g_w, self.V)
            if self.gw:
                self.P = self.polarization(self.g_w)
                self.W = self.screened_potential(self.P, self.V)
                self.sigma += self.dyn_self_energy(self.g_w, self.W, self.V)

            self.g_w, self.mu = self.dyson_equation(self.g0_w, 0.0, sigma_w = self.sigma, N_fix = self.N_fix)

            diff = np.max([np.max(np.abs((self.g_w - self.g_w_old)[self.blocks[0]].data)), np.max(np.abs((self.g_w - self.g_w_old)[self.blocks[1]].data))])
            self.g_w_old = self.g_w.copy()

            if diff < self.tol:
                break

            

    def polarization(self, g_w):
        return polarization(g_w, self.bmesh)
        
    def screened_potential(self, P_w, V): 
        V_t = V.copy()

        if not self.spinless:
            np.fill_diagonal(V_t, 0)

        I = np.eye(len(V))

        A = I - V_t * P_w['up']
        B =   -   V * P_w['dn']
        C =   -   V * P_w['up']
        D = I - V_t * P_w['dn']

        A_inv = inv(A)

        S = inv(D - C * A_inv * B)

        P_w['up'] = (A_inv + A_inv * B * S * C * A_inv) * V_t - A_inv * B * S * V;
        P_w['dn'] = -S * C * A_inv * V + S * V_t;
        
        return P_w

    def dyn_self_energy(self, g_w, W, V):
        return dyn_self_energy(g_w, W, V, self.spinless)
    
    def hartree_self_energy(self, g_w, V):
        return hartree_self_energy(g_w, V, self.spinless)
       
    def fock_self_energy(self, g_w, V):
        return fock_self_energy(g_w, V, self.spinless)
    
    def N(self, g_w):
        return total_density(g_w)
        
    def _dyson_dispatch(self, g_w, mu, sigma_w = None):
        if sigma_w is not None:
            return dyson_mu_sigma(g_w, mu, sigma_w)
        return dyson_mu(g_w, mu)
    
    
    def dyson_equation(self, g_w, mu, sigma_w = None, N_fix = False):
        if not N_fix:
            if mu == 0 and sigma_w is None:
                return g_w, mu
            return self._dyson_dispatch(g_w, mu, sigma_w), mu
        
        else:
            
            previous_direction = None

            occupation = self.N(self._dyson_dispatch(g_w, mu, sigma_w))
            # step = abs(occupation - N_fix)
            step = 1.0
            
            while abs(occupation - N_fix) > self.N_tol:

                if occupation - N_fix > 0.0:
                    if previous_direction == 'increment':
                        step /= 2.0
                    previous_direction = 'decrement'
                    mu -= step
                if occupation - N_fix < 0.0:
                    if previous_direction == 'decrement':
                        step /= 2.0
                    previous_direction = 'increment'
                    mu += step
                
                occupation = self.N(self._dyson_dispatch(g_w, mu, sigma_w))
                

            return self._dyson_dispatch(g_w, mu, sigma_w), mu
        

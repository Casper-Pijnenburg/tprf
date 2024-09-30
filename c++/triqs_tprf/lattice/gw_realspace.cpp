#include <iostream>
#include <typeinfo>
#include <nda/nda.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>
#include <iomanip>

#include "common.hpp"
#include "lattice_utility.hpp"
#include "../fourier/fourier.hpp"
#include "../mpi.hpp"
#include <omp.h>



namespace triqs_tprf {


    b_g_Dt_t iw_to_tau(b_g_Dw_cvt g_w) {
        auto iw_mesh = g_w[0].mesh();
        auto tau_mesh = make_adjoint_mesh(iw_mesh);
        int iw_size = iw_mesh.size();
        int tau_size = tau_mesh.size();

        int size = g_w[0].target().shape()[0];

        auto g_tau = make_block_gf<dlr_imtime>(g_w.block_names(), {gf(tau_mesh, g_w[0].target().shape()), gf(tau_mesh, g_w[0].target().shape())});


        #pragma omp parallel for shared(g_tau, g_w)
        for (int i = 0; i < size; ++i) {
            auto g_up = gf(iw_mesh, {1, size});
            auto g_dn = gf(iw_mesh, {1, size});

            for (int w = 0; w < iw_size; ++w){
                g_up[w](0, range::all) = g_w[0][w](i, range::all);
                g_dn[w](0, range::all) = g_w[1][w](i, range::all);
            }

            auto g_up_t = make_gf_dlr_imtime(make_gf_dlr(g_up));
            auto g_dn_t = make_gf_dlr_imtime(make_gf_dlr(g_dn));

            for (int t = 0; t < tau_size; ++t){
                g_tau[0][t](i, range::all) = g_up_t[t](0, range::all);
                g_tau[1][t](i, range::all) = g_dn_t[t](0, range::all);
            }
        }

        return g_tau;
    }


    b_g_Dw_t tau_to_iw(b_g_Dt_cvt g_t) {
        auto tau_mesh = g_t[0].mesh();
        auto iw_mesh = make_adjoint_mesh(tau_mesh);
        int tau_size = tau_mesh.size();
        int iw_size = iw_mesh.size();
        int size = g_t[0].target().shape()[0];

        auto g_w = make_block_gf<dlr_imfreq>(g_t.block_names(), {gf(iw_mesh, g_t[0].target().shape()), gf(iw_mesh, g_t[0].target().shape())});


        #pragma omp parallel for shared(g_t, g_w)
        for (int i = 0; i < size; ++i) {
            auto g_up = gf(tau_mesh, {1, size});
            auto g_dn = gf(tau_mesh, {1, size});

            for (int t = 0; t < tau_size; ++t){
                g_up[t](0, range::all) = g_t[0][t](i, range::all);
                g_dn[t](0, range::all) = g_t[1][t](i, range::all);
            }

            auto g_up_t = make_gf_dlr_imfreq(make_gf_dlr(g_up));
            auto g_dn_t = make_gf_dlr_imfreq(make_gf_dlr(g_dn));

            for (int w = 0; w < iw_size; ++w){
                g_w[0][w](i, range::all) = g_up_t[w](0, range::all);
                g_w[1][w](i, range::all) = g_dn_t[w](0, range::all);
            }
        }

        return g_w;
    }

    b_g_Dw_t dyson_mu(b_g_Dw_t g_w, double mu) {
        auto iw_mesh = g_w[0].mesh();
        int size = g_w[0].target().shape()[0];

        matrix<double> Mu(size, size);
        Mu() = mu;

        #pragma omp parallel for shared(g_w, Mu)
        for (int i = 0; i < iw_mesh.size(); ++i) {
            g_w[0][i] = inverse(inverse(g_w[0][i]) + Mu);
            g_w[1][i] = inverse(inverse(g_w[1][i]) + Mu);
        }

        return g_w;
    }

    b_g_Dw_t dyson_mu_sigma(b_g_Dw_t g_w, double mu, b_g_Dw_t sigma_w) {
        auto iw_mesh = g_w[0].mesh();
        int size = g_w[0].target().shape()[0];

        matrix<double> Mu(size, size);
        Mu() = mu;

        #pragma omp parallel for shared(g_w, Mu, sigma_w)
        for (int i = 0; i < iw_mesh.size(); ++i) {
            g_w[0][i] = inverse(inverse(g_w[0][i]) + Mu - sigma_w[0][i]);
            g_w[1][i] = inverse(inverse(g_w[1][i]) + Mu - sigma_w[1][i]);
        }
        
        return g_w;
    }

    double total_density(b_g_Dw_t g_w) {

        auto iw_mesh = g_w[0].mesh();
        int iw_size = iw_mesh.size();
        int size = g_w[0].target().shape()[0];

        double tot = 0.0;

        #pragma omp parallel for shared(g_w) reduction(+:tot)
        for (int i = 0; i < size; ++i) {
            auto g_up = gf<dlr_imfreq, scalar_valued>{iw_mesh};
            auto g_dn = gf<dlr_imfreq, scalar_valued>{iw_mesh};

            for (int w = 0; w < iw_size; ++w){
                g_up[w] = g_w[0][w](i, i);
                g_dn[w] = g_w[1][w](i, i);
            }

            tot += density(make_gf_dlr(g_up)).real() + density(make_gf_dlr(g_dn)).real();
        }

        return tot;
    }

    g_Dw_t inv(g_Dw_t g_w) {
        auto iw_mesh = g_w.mesh();

        #pragma omp parallel for shared(g_w)
        for (int i = 0; i < iw_mesh.size(); ++i) {
            g_w[i] = inverse(g_w[i]);
        }

        return g_w;
    }

    b_g_Dw_t polarization(b_g_Dw_cvt g_w, dlr_imfreq iw_mesh_b) {

        auto tau_mesh_b = make_adjoint_mesh(iw_mesh_b);
        int tau_mesh_size = tau_mesh_b.size();

        int orbitals = g_w[0].target().shape()[0];
        auto g_t = iw_to_tau(g_w);

        auto P_t = make_block_gf<dlr_imtime>(g_w.block_names(), {gf(tau_mesh_b, g_w[0].target().shape()), gf(tau_mesh_b, g_w[0].target().shape())});
        
        
        #pragma omp parallel for collapse(4) shared(P_t, g_t)
        for (int a = 0; a < orbitals; ++a) {
            for (int b = 0; b < orbitals; ++b) {
                for (int j = 0; j < 2; ++j) {
                    for (int i = 0; i < tau_mesh_size; ++i) {
                        P_t[j][i](a, b) = -1.0 * g_t[j][i](a, b) * g_t[j][tau_mesh_size - i - 1](b, a);
                    }
                }
            }
        }

        
        return tau_to_iw(P_t);
    }

    b_g_Dw_t screened_potential(b_g_Dw_t P_w, matrix<double> V, bool self_interactions) {
         
        auto iw_mesh = P_w[0].mesh();
        int size = P_w[0].target().shape()[0];
        auto I = nda::eye<double>(size);

        auto V_t = V;

        if (!self_interactions) {
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                V_t(i, i) = 0;
            }
        }

        #pragma omp parallel for shared(P_w, V_t, V)
        for (int i = 0; i < iw_mesh.size(); ++i) {
            
            auto A = I - V_t * P_w[0][i];
            auto B =   -   V * P_w[1][i];
            auto C =   -   V * P_w[0][i];
            auto D = I - V_t * P_w[1][i];

            auto Ainv = inverse(A);

            auto S = inverse(D - C * Ainv * B);

            P_w[0][i] = (Ainv + Ainv * B * S * C * Ainv) * V_t - Ainv * B * S * V;
            P_w[1][i] = -S * C * Ainv * V + S * V_t;
        }
        

        return P_w;
    }


    b_g_Dw_t dyn_self_energy(b_g_Dw_t g_w, b_g_Dw_cvt W_w, matrix<double> V, bool self_interactions) {
         
        auto iw_mesh_f = g_w[0].mesh();
        auto iw_mesh_b = W_w[0].mesh();
        int size = g_w[0].target().shape()[0];

        auto V_t = V;

        if (!self_interactions) {
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                V_t(i, i) = 0;
            }
        }

        auto W_dyn = block_gf(W_w);
        

        
        #pragma omp parallel for collapse(2) shared(W_dyn, W_w, V_t)
        for (int j = 0; j < iw_mesh_b.size(); ++j) {
            for (int i = 0; i < 2; ++i) {
                    W_dyn[i][j] = W_w[i][j] - V_t;
            }    
        }

        auto W_dyn_t = iw_to_tau(W_dyn);
        auto g_t = iw_to_tau(g_w);

        
        #pragma omp parallel for collapse(4) shared(g_t, W_dyn_t)
        for (int a = 0; a < size; ++a) {
            for (int b = 0; b < size; ++b) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < iw_mesh_f.size(); ++j) {
                        g_t[i][j](a, b) = -W_dyn_t[i][j](a, b) * g_t[i][j](a, b);
                    }
                }
            }
        }
    
        return tau_to_iw(g_t);
    }

    std::vector<matrix<std::complex<double>>> density(b_g_Dw_cvt g_w) {

        auto iw_mesh = g_w[0].mesh();
        int iw_size = iw_mesh.size();
        int size = g_w[0].target().shape()[0];

        matrix<std::complex<double>> rho_up(size, size);
        matrix<std::complex<double>> rho_dn(size, size);

        #pragma omp parallel for shared(g_w, rho_up, rho_dn)
        for (int i = 0; i < size; ++i) {
            auto g_up = gf(iw_mesh, {1, size});
            auto g_dn = gf(iw_mesh, {1, size});

            for (int w = 0; w < iw_size; ++w){
                g_up[w](0, range::all) = g_w[0][w](i, range::all);
                g_dn[w](0, range::all) = g_w[1][w](i, range::all);
            }


            rho_up(i, range::all) = density(make_gf_dlr(g_up))(0, range::all);
            rho_dn(i, range::all) = density(make_gf_dlr(g_dn))(0, range::all);
        }

        std::vector<matrix<std::complex<double>>> rho = {rho_up, rho_dn};
        return rho;
    }

    b_g_Dw_t hartree_self_energy(b_g_Dw_t g_w, matrix<double> V, bool self_interactions) {
        auto iw_mesh = g_w[0].mesh();
        int size = g_w[0].target().shape()[0];

        auto V_t = V;

        if (!self_interactions) {
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                V_t(i, i) = 0;
            }
        }

        std::vector<matrix<std::complex<double>>> rho = density(g_w);


        matrix<double> hartree_up(size, size);
        matrix<double> hartree_dn(size, size);
        hartree_up() = 0.0;
        hartree_dn() = 0.0;

        #pragma omp parallel for shared(rho, V_t, V, hartree_up, hartree_dn)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                hartree_up(i, i) += V_t(i, j) * rho[0](j, j).real() + V(i, j) * rho[1](j, j).real();
                hartree_dn(i, i) += V(i, j) * rho[0](j, j).real() + V_t(i, j) * rho[1](j, j).real();
            }
        }

        #pragma omp parallel for shared(g_w)
        for (int i = 0; i < iw_mesh.size(); ++i) {
            g_w[0][i] = hartree_up;
            g_w[1][i] = hartree_dn;
        } 

        return g_w;
    }

    b_g_Dw_t fock_self_energy(b_g_Dw_t g_w, matrix<double> V, bool self_interactions) {

        auto iw_mesh = g_w[0].mesh();
        int size = g_w[0].target().shape()[0];

        auto V_t = V;

        if (!self_interactions) {
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                V_t(i, i) = 0;
            }
        }


        std::vector<matrix<std::complex<double>>> rho = density(g_w);

        matrix<double> fock_up(size, size);
        matrix<double> fock_dn(size, size);
        fock_up() = 0.0;
        fock_dn() = 0.0;

        #pragma omp parallel for collapse(2) shared(rho, V_t, fock_up, fock_dn)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                fock_up(i, j) = -V_t(i, j) * rho[0](i, j).real();
                fock_dn(i, j) = -V_t(i, j) * rho[1](i, j).real();
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < iw_mesh.size(); ++i) {
            g_w[0][i] = fock_up;
            g_w[1][i] = fock_dn;
        } 

        return g_w;
    }

}


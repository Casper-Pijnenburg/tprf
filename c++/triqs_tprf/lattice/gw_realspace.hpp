namespace triqs_tprf {
    b_g_Dw_t polarization(b_g_Dw_cvt g_w, dlr_imfreq iw_mesh_b);
    b_g_Dw_t screened_potential(b_g_Dw_t P, matrix<double> V, bool self_interactions);
    b_g_Dw_t dyn_self_energy(b_g_Dw_t G, b_g_Dw_cvt W, matrix<double> V, bool self_interactions);
    b_g_Dw_t hartree_self_energy(b_g_Dw_t G, matrix<double> V, bool self_interactions);
    b_g_Dw_t fock_self_energy(b_g_Dw_t G, matrix<double> V, bool self_interactions);


    b_g_Dt_t iw_to_tau(b_g_Dw_cvt g_w);
    b_g_Dw_t tau_to_iw(b_g_Dt_cvt g_t);
    b_g_Dw_t dyson_mu(b_g_Dw_t g_w, double mu);
    b_g_Dw_t dyson_mu_sigma(b_g_Dw_t g_w, double mu, b_g_Dw_t sigma_w);
    double total_density(b_g_Dw_t g_w);
    g_Dw_t inv(g_Dw_t g_w);
}


namespace triqs_tprf {
    /** Compute the polarization :math:`P_{ij}^\sigma(i\omega_n)` from a Green's function :math:`G_{ij}^\sigma(i\omega_n)`.

    The polarization :math:`P_{ij}^\sigma(i\omega_n)`
    is given by

    .. math::
        P_{ij}^\sigma(i\omega_n) = 
        \mathcal{F}_{\tau\to\i\omega_n} \{G_{ij}^\sigma(\tau)G_{ji}^\sigma(-\tau)\}

    @param g_w Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @param iw_mesh_b Bosonic symmetrized DLR imaginary frequency mesh
    @return Polarization :math:`P_{ij}(i\omega_n)`
   */
    b_g_Dw_t polarization(b_g_Dw_cvt g_w, dlr_imfreq iw_mesh_b);


    /** Compute the screened potential :math:`W_{ij}^{\sigma^\sigma'}(i\omega_n)` from the polarization :math:`P_{ij}(i\omega_n)` and the interaction :math:`V_{ij}`.

    The screened potential :math:`W^{\sigma^\sigma'}(i\omega_n)`
    is given by

    .. math::
        W^{\sigma^\sigma'}(i\omega_n) = 
        (I-V^{\sigma\sigma'}P^{\sigma\sigma'}(i\omega_n))^{-1}V^{\sigma\sigma'}

    @param P Polarization block Green's function :math:`P_{ij}^\sigma(i\omega_n)`
    @param V Matrix containing the density-density interactions
    @param self_interactions Bool value on whether or not to allow self-interactions
    @return Screened potential :math:`W^{\sigma^\sigma'}(i\omega_n)`
   */    
    b_g_Dw_t screened_potential(b_g_Dw_t P, matrix<double> V, bool self_interactions);

    /** Compute the dynamical self-energy :math:`\Sigma_{ij}^{dyn,\sigma}(i\omega_n)` from the screened potential :math:`W_{ij}^{\sigma\sigma'}(i\omega_n)` and the Green's function :math:`G_{ij}^\sigma(i\omega_n)`.

    The dynamical self-energy :math:`\Sigma_{ij}^{dyn,\sigma}(i\omega_n)`
    is given by

    .. math::
        \Sigma_{ij}^{dyn,\sigma}(i\omega_n) = 
        \mathcal{F}_{\tau\to\i\omega_n} \{G_{ij}^\sigma(\tau)(W_{ij}^{\sigma\sigma}-V_{ij}^\sigma)(\tau)\}

    @param G Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @param W Screened potential block Green's function :math:`W_{ij}^{\sigma\sigma}(i\omega_n)`
    @param V Matrix containing the density-density interactions
    @param self_interactions Bool value on whether or not to allow self-interactions
    @return Dynamical self-energy :math:`\Sigma_{ij}^{dyn,\sigma}(i\omega_n)`
   */ 
    b_g_Dw_t dyn_self_energy(b_g_Dw_t G, b_g_Dw_cvt W, matrix<double> V, bool self_interactions);

    /** Compute the Hartree self-energy :math:`\Sigma_{ij}^{H,\sigma}(i\omega_n)` from the Green's function :math:`G_{ij}^\sigma(i\omega_n)` and the interaction math:`V_{ij}`.

    The Hartree self-energy :math:`\Sigma_{ij}^{H,\sigma}(i\omega_n)`
    is given by

    .. math::
        \Sigma_{ij}^{H,\sigma}(i\omega_n) = 
        \delta_{ij}\sum_{\sigma'}\sum_k V_{jk}^{\sigma\sigma'}\rho_{kk}^{\sigma'}

    where :math:`\rho_{kk}^{\sigma'}` is the density computed from the Green's function

    @param G Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @param V Matrix containing the density-density interactions
    @param self_interactions Bool value on whether or not to allow self-interactions
    @return Hartree self-energy :math:`\Sigma_{ij}^{H,\sigma}(i\omega_n)`
   */    
    b_g_Dw_t hartree_self_energy(b_g_Dw_t G, matrix<double> V, bool self_interactions);

    /** Compute the Fock self-energy :math:`\Sigma_{ij}^{F,\sigma}(i\omega_n)` from the Green's function :math:`G_{ij}^\sigma(i\omega_n)` and the interaction math:`V_{ij}`.

    The Fock self-energy :math:`\Sigma_{ij}^{H,\sigma}(i\omega_n)`
    is given by

    .. math::
        \Sigma_{ij}^{H,\sigma}(i\omega_n) = 
        -V_{ij}^{\sigma}\rho_{ij}^{\sigma}

    where :math:`\rho_{kk}^{\sigma'}` is the density computed from the Green's function

    @param G Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @param V Matrix containing the density-density interactions
    @param self_interactions Bool value on whether or not to allow self-interactions
    @return Fock self-energy :math:`\Sigma_{ij}^{H,\sigma}(i\omega_n)`
   */ 
    b_g_Dw_t fock_self_energy(b_g_Dw_t G, matrix<double> V, bool self_interactions);


    /** Computes the Fourier transfrom from DLR imaginary frequency to DLR imaginary time parallel over the orbital indices.

    @param g_w Block Green's function :math:`G_{ij}^\sigmaTotal density(i\omega_n)`
    @return Block Green's function :math:`G_{ij}^\sigma(\tau)`
   */
    b_g_Dt_t iw_to_tau(b_g_Dw_cvt g_w);

    /** Computes the Fourier transfrom from DLR imaginary time to DLR imaginary frequency parallel over the orbital indices.

    @param g_t Block Green's function :math:`G_{ij}^\sigma(\tau)`
    @return Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
   */
    b_g_Dw_t tau_to_iw(b_g_Dt_cvt g_t);

    /** Performs the Dyson equation on a block Green's function :math:`G_{ij}^\sigma(i\omega_n)` to add a given chemical potential shift :math:`\mu`.


    @param g_w Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @param mu Chemical potential shift
    @return Shifted block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
   */ 
    b_g_Dw_t dyson_mu(b_g_Dw_t g_w, double mu);

    /** Performs the Dyson equation on a block Green's function :math:`G_{ij}^\sigma(i\omega_n)` to add a given chemical potential shift :math:`\mu` and self-energy :math:`\Sigma_{ij}^{\sigma}(i\omega_n)`.

    @param g_w Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @param mu Chemical potential shift
    @param sigma_w Self-energy block Green's function :math:`\Sigma_{ij}^{dyn,\sigma}(i\omega_n)`
    @return Shifted block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
   */ 
    b_g_Dw_t dyson_mu_sigma(b_g_Dw_t g_w, double mu, b_g_Dw_t sigma_w);

    /** Computes the total density of a block Green's function :math:`G_{ij}^\sigma(i\omega_n)`

    @param g_w Block Green's function :math:`G_{ij}^\sigma(i\omega_n)`
    @return Total density
   */    
    double total_density(b_g_Dw_t g_w);

    /** Computes the inverse of a block Green's function parallel over the mesh

    @param g_w Block Green's function
    @return Inverted block Green's function
   */      
    g_Dw_t inv(g_Dw_t g_w);
}


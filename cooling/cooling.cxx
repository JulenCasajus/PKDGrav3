/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2017 Matthieu Schaller (matthieu.schaller@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
/**
 * @file src/cooling/EAGLE/cooling.c
 * @brief EAGLE cooling functions
 */

/* Config parameters. */

/* Some standard headers. */
#include <float.h>
#include <hdf5.h>
#include <math.h>
#include <time.h>

#include "cooling.h"
#include "cooling_rates.h"

#include "pkd.h"
#include "master.h"
#include "pkd_config.h"
#include <stdio.h>
#include <algorithm>

/* Maximum number of iterations for bisection scheme */
static const int bisection_max_iterations = 150;

/* Tolerances for termination criteria. */
static const float explicit_tolerance = 0.05;
static const float bisection_tolerance = 1.0e-6;
static const double bracket_factor = 1.5;
static inline void get_redshift_index(const float z, int *z_index, float *dz,
                                      struct cooling_function_data *restrict cooling);

void MSR::SetCoolingParam() {
    calc.dCoolingMinu = parameters.get_dCoolingMinTemp() * dTuFacPrimNeutral;
}

/**
 * @brief Common operations performed on the cooling function at a
 * given time-step or redshift. Predominantly used to read cooling tables
 * above and below the current redshift, if not already read in.
 *
 * Also calls the additional H reionization energy injection if need be.
 *
 * @param cosmo The current cosmological model.
 * @param cooling The #cooling_function_data used in the run.
 * @param s The space data, including a pointer to array of particles
 *
 *
 */
void MSR::CoolingUpdate(float redshift) {
    printf("Updating cooling.. z=%f\n", redshift);

    /* Current redshift */
    //const float redshift = cosmo->z;

    /* What is the current table index along the redshift axis? */
    int z_index = -1;
    float dz = 0.f;
    get_redshift_index(redshift, &z_index, &dz, cooling);
    cooling->dz = dz;

    // We update dz in each process
    pstCoolingUpdateZ(pst, &dz, sizeof(float), NULL, 0);

    /* Do we already have the correct tables loaded? */
    if (cooling->z_index == z_index) return;

    /* Which table should we load ? */
    if (z_index == eagle_cooling_N_redshifts) {
        /* Before reionization. Just load the corresponding table */
        get_redshift_invariant_table(cooling, /* photodis=*/1);
    }
    else {
        /* After reionization. Just load the corresponding table */
        if (z_index > eagle_cooling_N_redshifts) {
            /* Between reionization and first z-dependent table */
            get_redshift_invariant_table(cooling, /* photodis=*/0);
        }
        else {
            /* Normal case: two tables bracketing the current z */
            get_cooling_table(cooling, z_index, z_index + 1);
        }

        /* Extra energy for reionization */
        if (!cooling->H_reion_done) {
            printf("Hydrogen reionization! Injecting %.1f eV per proton\n",
                   parameters.get_fH_reion_eV_p_H());
            pstCoolingHydReion(pst, NULL, 0, NULL, 0);

            /* Flag that reionization happened */
            cooling->H_reion_done = 1;
        }
    }

    /* Store the currently loaded index */
    cooling->z_index = z_index;
    // We send the newly loaded table
    printf("Sending cooling tables...\n");
    struct inCoolUpdate in;
    in.z_index = cooling->z_index;
    in.previous_z_index = cooling->previous_z_index;
    in.dz = cooling->dz;

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_metal_heating ; i++)
        in.metal_heating[i] = cooling->table.metal_heating[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_HpHe_heating; i++)
        in.H_plus_He_heating[i] = cooling->table.H_plus_He_heating[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_HpHe_electron_abundance; i++)
        in.H_plus_He_electron_abundance[i] = cooling->table.H_plus_He_electron_abundance[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_temperature; i++)
        in.temperature[i] = cooling->table.temperature[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_electron_abundance; i++)
        in.electron_abundance[i] = cooling->table.electron_abundance[i];

    pstCoolingUpdate(pst, &in, sizeof(in), NULL, 0);
}

/**
 * Initialises properties stored in the cooling_function_data struct
 */
void MSR::CoolingInit(float redshift) {
    printf("Initializing cooling \n");

    /* Allocate the needed structs */
    cooling = (struct cooling_function_data *) malloc(sizeof(struct cooling_function_data));

    /* Read model parameters */

    strcpy(cooling->cooling_table_path, parameters.get_achCoolingTables().data());

    /* Despite the names, the values of H_reion_heat_cgs and He_reion_heat_cgs
     * that are read in are actually in units of electron volts per proton mass.
     * We later convert to units just below */

    cooling->H_reion_z = parameters.get_fH_reion_z();
    if (redshift <= cooling->H_reion_z) {
        cooling->H_reion_done = 1;
    }
    else {
        cooling->H_reion_done = 0;
    }
    cooling->H_reion_heat_cgs = parameters.get_fH_reion_eV_p_H();
    cooling->He_reion_z_centre = parameters.get_fHe_reion_z_centre();
    cooling->He_reion_z_sigma = parameters.get_fHe_reion_z_sigma();
    cooling->He_reion_heat_cgs = parameters.get_fHe_reion_eV_p_H();

    /* Optional parameters to correct the abundances */
    cooling->Ca_over_Si_ratio_in_solar = calc.fCa_over_Si_in_Solar;
    cooling->S_over_Si_ratio_in_solar = calc.fS_over_Si_in_Solar;

    /* Convert H_reion_heat_cgs and He_reion_heat_cgs to cgs
     * (units used internally by the cooling routines). This is done by
     * multiplying by 'eV/m_H' in internal units, then converting to cgs units.
     * Note that the dimensions of these quantities are energy/mass = velocity^2
     */
#define EV_CGS 1.602e-12

    cooling->H_reion_heat_cgs *= (EV_CGS) / (MHYDR);

    cooling->He_reion_heat_cgs *= (EV_CGS) / (MHYDR) ;

    /* Read in the list of redshifts */
    get_cooling_redshifts(cooling);

    /* Read in cooling table header */
    char fname[eagle_table_path_name_length + 12];
    snprintf(fname, sizeof(fname), "%s/z_0.000.hdf5", cooling->cooling_table_path);
    read_cooling_header(fname, cooling);

    /* Allocate space for cooling tables */
    allocate_cooling_tables(cooling);

    /* Compute conversion factors */
    // This is not ideal, and does not follow PKDGRAV3 philosophy... someday
    // it should be reworked
    cooling->internal_energy_to_cgs = units.dErgPerGmUnit;
    cooling->internal_energy_from_cgs = 1. / cooling->internal_energy_to_cgs;
    cooling->number_density_to_cgs = pow(units.dKpcUnit * KPCCM, -3.);
    cooling->units = units;
    cooling->dConstGamma = parameters.get_dConstGamma();
    cooling->dCoolingMinu = calc.dCoolingMinu;

    /* Store some constants in CGS units */
    const double proton_mass_cgs = MHYDR;
    cooling->inv_proton_mass_cgs = 1. / proton_mass_cgs;
    cooling->T_CMB_0 = parameters.get_fT_CMB_0();

    const double compton_coefficient_cgs = 1.0178085e-37;

    /* And now the Compton rate */
    cooling->compton_rate_cgs = compton_coefficient_cgs * cooling->T_CMB_0 *
                                cooling->T_CMB_0 * cooling->T_CMB_0 *
                                cooling->T_CMB_0;

    /* Set the redshift indices to invalid values */
    cooling->z_index = -10;

    /* set previous_z_index and to last value of redshift table*/
    cooling->previous_z_index = eagle_cooling_N_redshifts - 2;

    // We prepare the data to be scattered among the processes
    struct inCoolInit in;
    in.in_cooling_data = *(cooling);
    for (int i = 0; i < eagle_cooling_N_redshifts; i++) in.Redshifts[i] = cooling->Redshifts[i];
    for (int i = 0; i < eagle_cooling_N_density; i++) in.nH[i] = cooling->nH[i];
    for (int i = 0; i < eagle_cooling_N_He_frac; i++) in.HeFrac[i] = cooling->HeFrac[i];
    for (int i = 0; i < eagle_cooling_N_temperature; i++) in.Temp[i] = cooling->Temp[i];
    for (int i = 0; i < eagle_cooling_N_temperature; i++) in.Therm[i] = cooling->Therm[i];
    for (int i = 0; i < eagle_cooling_N_abundances; i++) in.SolarAbundances[i] = cooling->SolarAbundances[i];
    for (int i = 0; i < eagle_cooling_N_abundances; i++) in.SolarAbundances_inv[i] = cooling->SolarAbundances_inv[i];

    pstCoolingInit(pst, &in, sizeof(in), NULL, 0);
}

/**
 * @brief Find the index of the current redshift along the redshift dimension
 * of the cooling tables.
 *
 * Since the redshift table is not evenly spaced, compare z with each
 * table value in decreasing order starting with the previous redshift index
 *
 * The returned difference is expressed in units of the table separation. This
 * means dx = (x - table[i]) / (table[i + 1] - table[i]). It is always between
 * 0 and 1.
 *
 * @param z Redshift we are searching for.
 * @param z_index (return) Index of the redshift in the table.
 * @param dz (return) Difference in redshift between z and table[z_index].
 * @param cooling #cooling_function_data structure containing redshift table.
 */
static inline void get_redshift_index(
    const float z, int *z_index, float *dz,
    struct cooling_function_data *restrict cooling) {

    /* Before the earliest redshift or before hydrogen reionization, flag for
     * collisional cooling */
    if (z > cooling->H_reion_z) {
        *z_index = eagle_cooling_N_redshifts;
        *dz = 0.0;
    }

    /* From reionization use the cooling tables */
    else if (z > cooling->Redshifts[eagle_cooling_N_redshifts - 1] &&
             z <= cooling->H_reion_z) {
        *z_index = eagle_cooling_N_redshifts + 1;
        *dz = 0.0;
    }

    /* At the end, just use the last value */
    else if (z <= cooling->Redshifts[0]) {
        *z_index = 0;
        *dz = 0.0;
    }

    /* Normal case: search... */
    else {

        /* start at the previous index and search */
        for (int i = cooling->previous_z_index; i >= 0; i--) {
            if (z > cooling->Redshifts[i]) {

                *z_index = i;
                cooling->previous_z_index = i;

                *dz = (z - cooling->Redshifts[i]) /
                      (cooling->Redshifts[i + 1] - cooling->Redshifts[i]);
                break;
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Bisection integration scheme
 *
 * @param u_ini_cgs Internal energy at beginning of hydro step in CGS.
 * @param n_H_cgs Hydrogen number density in CGS.
 * @param redshift Current redshift.
 * @param n_H_index Particle hydrogen number density index.
 * @param d_n_H Particle hydrogen number density offset.
 * @param He_index Particle helium fraction index.
 * @param d_He Particle helium fraction offset.
 * @param Lambda_He_reion_cgs Cooling rate coming from He reionization.
 * @param ratefact_cgs Multiplication factor to get a cooling rate.
 * @param cooling #cooling_function_data structure.
 * @param abundance_ratio Array of ratios of metal abundance to solar.
 * @param dt_cgs timestep in CGS.
 */
inline static double bisection_iter(
    const double u_ini_cgs, const double n_H_cgs, const double redshift,
    const int n_H_index, const float d_n_H, const int He_index,
    const float d_He, const double Lambda_He_reion_cgs,
    const double ratefact_cgs,
    const struct cooling_function_data *restrict cooling,
    const float abundance_ratio[eagle_cooling_N_abundances],
    const double dt_cgs) {

    /* Bracketing */
    double u_lower_cgs = u_ini_cgs;
    double u_upper_cgs = u_ini_cgs;

    /*************************************/
    /* Let's get a first guess           */
    /*************************************/

    double LambdaNet_cgs =
        Lambda_He_reion_cgs +
        eagle_cooling_rate(log10(u_ini_cgs), redshift, n_H_cgs, abundance_ratio,
                           n_H_index, d_n_H, He_index, d_He, cooling);

    /*************************************/
    /* Let's try to bracket the solution */
    /*************************************/

    if (LambdaNet_cgs < 0) {

        /* we're cooling! */
        u_lower_cgs /= bracket_factor;
        u_upper_cgs *= bracket_factor;

        /* Compute a new rate */
        LambdaNet_cgs = Lambda_He_reion_cgs +
                        eagle_cooling_rate(log10(u_lower_cgs), redshift, n_H_cgs,
                                           abundance_ratio, n_H_index, d_n_H,
                                           He_index, d_He, cooling);

        int i = 0;
        while (u_lower_cgs - u_ini_cgs - LambdaNet_cgs * ratefact_cgs * dt_cgs >
                0 &&
                i < bisection_max_iterations) {

            u_lower_cgs /= bracket_factor;
            u_upper_cgs /= bracket_factor;

            /* Compute a new rate */
            LambdaNet_cgs = Lambda_He_reion_cgs +
                            eagle_cooling_rate(log10(u_lower_cgs), redshift, n_H_cgs,
                                               abundance_ratio, n_H_index, d_n_H,
                                               He_index, d_He, cooling);
            i++;
        }

        if (i >= bisection_max_iterations) {
            printf(
                "particle exceeded max iterations searching for bounds when "
                "cooling, u_ini_cgs %.5e n_H_cgs %.5e \n",
                u_ini_cgs, n_H_cgs);
        }
    }
    else {

        /* we are heating! */
        u_lower_cgs /= bracket_factor;
        u_upper_cgs *= bracket_factor;

        /* Compute a new rate */
        LambdaNet_cgs = Lambda_He_reion_cgs +
                        eagle_cooling_rate(log10(u_upper_cgs), redshift, n_H_cgs,
                                           abundance_ratio, n_H_index, d_n_H,
                                           He_index, d_He, cooling);

        int i = 0;
        while (u_upper_cgs - u_ini_cgs - LambdaNet_cgs * ratefact_cgs * dt_cgs <
                0 &&
                i < bisection_max_iterations) {

            u_lower_cgs *= bracket_factor;
            u_upper_cgs *= bracket_factor;

            /* Compute a new rate */
            LambdaNet_cgs = Lambda_He_reion_cgs +
                            eagle_cooling_rate(log10(u_upper_cgs), redshift, n_H_cgs,
                                               abundance_ratio, n_H_index, d_n_H,
                                               He_index, d_He, cooling);
            i++;
        }

        if (i >= bisection_max_iterations) {
            printf(
                "particle exceeded max iterations searching for bounds when "
                "heating, u_ini_cgs %.5e n_H_cgs %.5e \n",
                u_ini_cgs, n_H_cgs);
        }
    }

    /********************************************/
    /* We now have an upper and lower bound.    */
    /* Let's iterate by reducing the bracketing */
    /********************************************/

    /* bisection iteration */
    int i = 0;
    double u_next_cgs;

    do {

        /* New guess */
        u_next_cgs = 0.5 * (u_lower_cgs + u_upper_cgs);

        /* New rate */
        LambdaNet_cgs = Lambda_He_reion_cgs +
                        eagle_cooling_rate(log10(u_next_cgs), redshift, n_H_cgs,
                                           abundance_ratio, n_H_index, d_n_H,
                                           He_index, d_He, cooling);
#ifdef SWIFT_DEBUG_CHECKS
        if (u_next_cgs <= 0)
            error(
                "Got negative energy! u_next_cgs=%.5e u_upper=%.5e u_lower=%.5e "
                "Lambda=%.5e",
                u_next_cgs, u_upper_cgs, u_lower_cgs, LambdaNet_cgs);
#endif

        /* Where do we go next? */
        if (u_next_cgs - u_ini_cgs - LambdaNet_cgs * ratefact_cgs * dt_cgs > 0.0) {
            u_upper_cgs = u_next_cgs;
        }
        else {
            u_lower_cgs = u_next_cgs;
        }

        i++;
    } while (fabs(u_upper_cgs - u_lower_cgs) / u_next_cgs > bisection_tolerance &&
             i < bisection_max_iterations);

    if (i >= bisection_max_iterations)
        printf("Particle failed to converge \n");

    return u_upper_cgs;
}

/**
 * @brief Apply the cooling function to a particle.
 *
 * We want to compute u_new such that u_new = u_old + dt * du / dt(u_new, X),
 * where X stands for the metallicity, density and redshift. These are
 * kept constant.
 *
 * We first compute du / dt(u_old). If dt * du / dt(u_old) is small enough, we
 * use an explicit integration and use this as our solution.
 *
 * Otherwise, we try to find a solution to the implicit time - integration
 * problem. This leads to the root - finding problem:
 *
 * f(u_new) = u_new - u_old - dt * du / dt(u_new, X) = 0
 *
 * We first try a few Newton - Raphson iteration if it does not converge, we
 * revert to a bisection scheme.
 *
 * This is done by first bracketing the solution and then iterating
 * towards the solution by reducing the window down to a certain tolerance.
 * Note there is always at least one solution since
 * f(+inf) is < 0 and f(-inf) is > 0.
 *
 * @param phys_const The physical constants in internal units.
 * @param us The internal system of units.
 * @param cosmo The current cosmological model.
 * @param hydro_properties the hydro_props struct
 * @param floor_props Properties of the entropy floor.
 * @param cooling The #cooling_function_data used in the run.
 * @param p Pointer to the particle data.
 * @param xp Pointer to the extended particle data.
 * @param dt The cooling time-step of this particle.
 * @param dt_therm The hydro time-step of this particle.
 * @param time The current time (since the Big Bang or start of the run) in
 * internal units.
 */
void cooling_cool_part(PKD pkd,
                       const struct cooling_function_data *cooling,
                       //struct part *restrict p, struct xpart *restrict xp,
                       particleStore::ParticleReference &p, meshless::FIELDS *psph,
                       const float dt, const double time,
                       const float delta_redshift, const double redshift) {

    /* No cooling happens over zero time */
    if (dt == 0.) return;

#ifdef SWIFT_DEBUG_CHECKS
    if (cooling->Redshifts == NULL)
        error(
            "Cooling function has not been initialised. Did you forget the "
            "--cooling runtime flag?");
#endif

    /* Get internal energy at the last kick step */
    const float u_start = psph->lastUint;

    /* Get the change in internal energy due to hydro forces */
//  const float hydro_du_dt = hydro_get_physical_internal_energy_dt(p, cosmo);

    /* Get internal energy at the end of the step (assuming dt does not
     * increase) */
//  double u_0 = (u_start + hydro_du_dt * dt);

    /* Check for minimal energy */
//  u_0 = max(u_0, hydro_properties->minimal_internal_energy);

    /* IA: In our case we are using operator splitting so this is simpler */
    const float fMass = p.mass();
    double u_0 = psph->Uint / fMass;
    if (u_0 < cooling->dCoolingMinu) u_0 = cooling->dCoolingMinu;

    /* Convert to CGS units */
    double u_0_cgs = u_0 * cooling->internal_energy_to_cgs;
    const double dt_cgs =  cooling->units.dSecUnit * dt;

    /* Change in redshift over the course of this time-step
       (See cosmology theory document for the derivation) */
    //const double delta_redshift = -dt * cosmo->H * cosmo->a_inv;

    /* Get this particle's abundance ratios compared to solar
     * Note that we need to add S and Ca that are in the tables but not tracked
     * by the particles themselves.
     * The order is [H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe] */
    float abundance_ratio[eagle_cooling_N_abundances];
    abundance_ratio_to_solar(psph, fMass, cooling, abundance_ratio);

    /* Get the Hydrogen and Helium mass fractions */
    const auto &elem_mass = psph->ElemMass;
    //chemistry_get_metal_mass_fraction_for_cooling(p);
    const float XH = elem_mass[ELEMENT_H] / fMass;
    const float XHe = elem_mass[ELEMENT_He] / fMass;

    /* Get the Helium mass fraction. Note that this is He / (H+He), i.e. a
     * metal - free Helium mass fraction as per the Wiersma + 08 definition */
    const float HeFrac = XHe / (XH + XHe);

    /* convert Hydrogen mass fraction into physical Hydrogen number density */
    const float a_m3 = pow(1.+redshift, 3.);
    const float rho = p.density() * a_m3 ;
    const double n_H = rho * XH / MHYDR * cooling->units.dMsolUnit * MSOLG;
    const double n_H_cgs = n_H * cooling->number_density_to_cgs;

    /* ratefact = n_H * n_H / rho; Might lead to round - off error: replaced by
     * equivalent expression  below */
    const double ratefact_cgs = n_H_cgs * (XH * cooling->inv_proton_mass_cgs);

    /* compute hydrogen number density and helium fraction table indices and
     * offsets (These are fixed for any value of u, so no need to recompute them)
     */
    int He_index, n_H_index;
    float d_He, d_n_H;
    get_index_1d(cooling->HeFrac, eagle_cooling_N_He_frac, HeFrac, &He_index,
                 &d_He);
    get_index_1d(cooling->nH, eagle_cooling_N_density, log10(n_H_cgs), &n_H_index,
                 &d_n_H);

    /* Start by computing the cooling (heating actually) rate from Helium
       reionization as this needs to be added on no matter what */

    /* Get helium and hydrogen reheating term */
    const double Helium_reion_heat_cgs =
        eagle_helium_reionization_extraheat(redshift, delta_redshift, cooling);

    /* Convert this into a rate */
    const double Lambda_He_reion_cgs =
        Helium_reion_heat_cgs / (dt_cgs * ratefact_cgs);

    /* Let's compute the internal energy at the end of the step */
    /* Initialise to the initial energy to appease compiler; this will never not
       be overwritten. */
    double u_final_cgs = u_0_cgs;

    /* First try an explicit integration (note we ignore the derivative) */
    const double LambdaNet_cgs =
        Lambda_He_reion_cgs +
        eagle_cooling_rate(log10(u_0_cgs), redshift, n_H_cgs, abundance_ratio,
                           n_H_index, d_n_H, He_index, d_He, cooling);

    /* if cooling rate is small, take the explicit solution */
    if (fabs(ratefact_cgs * LambdaNet_cgs * dt_cgs) <
            explicit_tolerance * u_0_cgs) {
        u_final_cgs = u_0_cgs + ratefact_cgs * LambdaNet_cgs * dt_cgs;
    }
    else {
        /* Otherwise, go the bisection route. */
        u_final_cgs =
            bisection_iter(u_0_cgs, n_H_cgs, redshift, n_H_index, d_n_H, He_index,
                           d_He, Lambda_He_reion_cgs, ratefact_cgs, cooling,
                           abundance_ratio, dt_cgs);
    }

    /* Convert back to internal units */
    double u_final = u_final_cgs * cooling->internal_energy_from_cgs;
    if (u_final < cooling->dCoolingMinu) u_final = cooling->dCoolingMinu;
    psph->E = psph->E - psph->Uint;
    psph->Uint = u_final * fMass;
    psph->E = psph->E + psph->Uint;

#ifdef ENTROPY_SWITCH
    psph->S = psph->Uint *
              (cooling->dConstGamma -1.) *
              pow(p.density(), -cooling->dConstGamma + 1);
#endif

}

/**
 * @brief Computes the cooling time-step.
 *
 * The time-step is not set by the properties of cooling.
 *
 * @param cooling The #cooling_function_data used in the run.
 * @param phys_const #phys_const data struct.
 * @param us The internal system of units.
 * @param cosmo #cosmology struct.
 * @param hydro_props the properties of the hydro scheme.
 * @param p #part data.
 * @param xp extended particle data.
 */
//__attribute__((always_inline)) INLINE float cooling_timestep(
//    const struct cooling_function_data *restrict cooling,
//    const struct phys_const *restrict phys_const,
//    const struct cosmology *restrict cosmo,
//    const struct unit_system *restrict us,
//    const struct hydro_props *hydro_props, const struct part *restrict p,
//    const struct xpart *restrict xp) {
//
//  return FLT_MAX;
//}

/**
 * @brief Sets the cooling properties of the (x-)particles to a valid start
 * state.
 *
 * @param phys_const #phys_const data structure.
 * @param us The internal system of units.
 * @param hydro_props The properties of the hydro scheme.
 * @param cosmo #cosmology data structure.
 * @param cooling #cooling_function_data struct.
 * @param p #part data.
 * @param xp Pointer to the #xpart data.
 */
//__attribute__((always_inline)) INLINE void cooling_first_init_part(
//    const struct phys_const *restrict phys_const,
//    const struct unit_system *restrict us,
//    const struct hydro_props *hydro_props,
//    const struct cosmology *restrict cosmo,
//    const struct cooling_function_data *restrict cooling,
//    const struct part *restrict p, struct xpart *restrict xp) {
//
//  xp->cooling_data.radiated_energy = 0.f;
//}

/**
 * @brief Compute the temperature of a #part based on the cooling function.
 *
 * We use the Temperature table of the Wiersma + 08 set. This computes the
 * equilibirum temperature of a gas for a given redshift, Hydrogen density,
 * internal energy per unit mass and Helium fraction.
 *
 * The temperature returned is consistent with the cooling rates.
 *
 * @param phys_const #phys_const data structure.
 * @param hydro_props The properties of the hydro scheme.
 * @param us The internal system of units.
 * @param cosmo #cosmology data structure.
 * @param cooling #cooling_function_data struct.
 * @param p #part data.
 * @param xp Pointer to the #xpart data.
 */
float cooling_get_temperature(PKD pkd, const float redshift,
                              const struct cooling_function_data *restrict cooling,
                              particleStore::Particle &p, meshless::FIELDS *psph) {

#ifdef SWIFT_DEBUG_CHECKS
    if (cooling->Redshifts == NULL)
        error(
            "Cooling function has not been initialised. Did you forget the "
            "--temperature runtime flag?");
#endif

    /* Get physical internal energy */
    const float fMass = p.mass();
    const float u = psph->Uint / fMass;
    double u_cgs = u * cooling->internal_energy_to_cgs;
    if (u_cgs < 1e10) return psph->P / p.density() / cooling->units.dGasConst *1.14 ;
    //printf("u_cgs %e \n", u_cgs);

    /* Get the Hydrogen and Helium mass fractions */
    const auto &elem_mass = psph->ElemMass;
    //chemistry_get_metal_mass_fraction_for_cooling(p);
    const float XH = elem_mass[ELEMENT_H] / fMass;
    const float XHe = elem_mass[ELEMENT_He] / fMass;

    /* Get the Helium mass fraction. Note that this is He / (H+He), i.e. a
     * metal - free Helium mass fraction as per the Wiersma + 08 definition */
    const float HeFrac = XHe / (XH + XHe);

    /* Convert Hydrogen mass fraction into Hydrogen number density */
    const float rho = p.density() * pow(1.+redshift, 3.); // TODO: why does not work with dComovingGmPerCcUnit?
    //printf("dComovingGmPerCcUnit %e \n", cooling->units.dComovingGmPerCcUnit);
    const double n_H = rho * XH / MHYDR * cooling->units.dMsolUnit * MSOLG;
    const double n_H_cgs = n_H * cooling->number_density_to_cgs;

    /* compute hydrogen number density and helium fraction table indices and
     * offsets */
    int He_index, n_H_index;
    float d_He, d_n_H;
    get_index_1d(cooling->HeFrac, eagle_cooling_N_He_frac, HeFrac, &He_index,
                 &d_He);
    get_index_1d(cooling->nH, eagle_cooling_N_density, log10(n_H_cgs), &n_H_index,
                 &d_n_H);
    //printf("n_H %e \t n_to_cgs %e \t n_H_cgs %e \n", n_H, cooling->number_density_to_cgs, n_H_cgs);
    //abort();

    /* Compute the log10 of the temperature by interpolating the table */
    const double log_10_T = eagle_convert_u_to_temp(
                                log10(u_cgs), redshift, n_H_index, He_index, d_n_H, d_He, cooling);

    /* Undo the log! */
    return exp(log_10_T * M_LN10);
}
#ifdef __cplusplus
}
#endif

/**
 * @brief Returns the total radiated energy by this particle.
 *
 * @param xp #xpart data struct
 */
//__attribute__((always_inline)) INLINE float cooling_get_radiated_energy(
//    const struct xpart *restrict xp) {
//
//  return xp->cooling_data.radiated_energy;
//}

/**
 * @brief Split the coolong content of a particle into n pieces
 *
 * @param p The #part.
 * @param xp The #xpart.
 * @param n The number of pieces to split into.
 */
//void cooling_split_part(struct part *p, struct xpart *xp, double n) {
//
//  xp->cooling_data.radiated_energy /= n;
//}
//
/**
 * @brief Inject a fixed amount of energy to each particle in the simulation
 * to mimic Hydrogen reionization.
 *
 * @param cooling The properties of the cooling model.
 * @param cosmo The cosmological model.
 * @param s The #space containing the particles.
 */
void cooling_Hydrogen_reionization(PKD pkd) {

    struct cooling_function_data *cooling = pkd->cooling;
    /* Energy to inject in internal units */
    const double extra_heat_per_proton =
        cooling->H_reion_heat_cgs * cooling->internal_energy_from_cgs ;

    cooling->H_reion_done = 1;
    /* Loop through particles and set new heat */
    for (auto &p : pkd->particles) {
        if (p.is_gas()) {
            auto &sph = p.sph();

            const double old_u = sph.Uint ;

            /* IA: Mass in hydrogen */
            const double extra_heat = extra_heat_per_proton * sph.ElemMass[ELEMENT_H];
            const double new_u = old_u + extra_heat;

            //printf("Applying extra energy for H reionization! U=%e dU=%e \n", old_u, extra_heat);
#ifdef ENTROPY_SWITCH
            sph.S += extra_heat * (cooling->dConstGamma - 1.) *
                     pow(p.density(), -cooling->dConstGamma + 1);
#endif

            //hydro_set_physical_internal_energy(p, xp, cosmo, new_u);
            //hydro_set_drifted_physical_internal_energy(p, cosmo, new_u);
            //printf("old_u %e \t new_u %e \t du %e \n", old_u, new_u, extra_heat);
            sph.Uint = new_u;
            sph.E += extra_heat;
            /* IA: TODO: Can this cause problems as now lastUint and Uint can be very different? */
        }
    }
}

/* IA:
 * The master process will send us the cooling_function_data, which we then need to allocate
 *  in each process and copy everything, taking special care of other arrays (deep copy).
 *
 *  Furthermore, we initialize the abundances if needed
 */
void pkd_cooling_init_backend(PKD pkd, struct cooling_function_data in_cooling_data,
                              float Redshifts[eagle_cooling_N_redshifts],
                              float nH[eagle_cooling_N_density],
                              float Temp[eagle_cooling_N_temperature],
                              float HeFrac[eagle_cooling_N_He_frac],
                              float Therm[eagle_cooling_N_temperature],
                              float SolarAbundances[eagle_cooling_N_abundances],
                              float SolarAbundances_inv[eagle_cooling_N_abundances]
                             ) {
    //printf("Initializing in a single process \n");

    /* IA: Allocate the needed structs */
    pkd->cooling = (struct cooling_function_data *) malloc(sizeof(struct cooling_function_data));
    if (pkd->cooling == NULL) printf("Error allocating cooling_function_data\n");
    //struct cooling_function_data *cooling = pkd->cooling;

    memcpy(pkd->cooling, &in_cooling_data, sizeof(struct cooling_function_data));

#define ALLOC_AND_COPY(arr, n) \
  pkd->cooling->arr = (float *) malloc(sizeof(float)*n); \
  if (pkd->cooling->arr == NULL) printf("Error allocating (arr) with %d elements \n", n); \
  for (int i = 0; i < n; i++) pkd->cooling->arr[i] = arr[i];

    ALLOC_AND_COPY(Redshifts, eagle_cooling_N_redshifts)
    ALLOC_AND_COPY(nH, eagle_cooling_N_density)
    ALLOC_AND_COPY(Temp, eagle_cooling_N_temperature)
    ALLOC_AND_COPY(HeFrac, eagle_cooling_N_He_frac)
    ALLOC_AND_COPY(Therm, eagle_cooling_N_temperature)
    ALLOC_AND_COPY(SolarAbundances, eagle_cooling_N_abundances)
    ALLOC_AND_COPY(SolarAbundances_inv, eagle_cooling_N_abundances)

#undef ALLOC_AND_COPY

    /* Allocate space for cooling tables */
    allocate_cooling_tables(pkd->cooling);
}

void pkd_cooling_update(PKD pkd, struct inCoolUpdate *in) {

    pkd->cooling->z_index = in->z_index;
    pkd->cooling->previous_z_index = in->previous_z_index;
    pkd->cooling->dz  = in->dz;

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_metal_heating ; i++)
        pkd->cooling->table.metal_heating[i] = in->metal_heating[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_HpHe_heating; i++)
        pkd->cooling->table.H_plus_He_heating[i] = in->H_plus_He_heating[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_HpHe_electron_abundance; i++)
        pkd->cooling->table.H_plus_He_electron_abundance[i] = in->H_plus_He_electron_abundance[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_temperature; i++)
        pkd->cooling->table.temperature[i] = in->temperature[i];

    for (int i = 0; i < eagle_cooling_N_loaded_redshifts * num_elements_electron_abundance; i++)
        pkd->cooling->table.electron_abundance[i] = in->electron_abundance[i];

}

/**
 * @brief Restore cooling tables (if applicable) after
 * restart
 *
 * @param cooling the #cooling_function_data structure
 * @param cosmo #cosmology structure
 */
//void cooling_restore_tables(struct cooling_function_data *cooling,
//                            const struct cosmology *cosmo) {
//
//  /* Read redshifts */
//  get_cooling_redshifts(cooling);
//
//  /* Read cooling header */
//  char fname[eagle_table_path_name_length + 12];
//  sprintf(fname, "%sz_0.000.hdf5", cooling->cooling_table_path);
//  read_cooling_header(fname, cooling);
//
//  /* Allocate memory for the tables */
//  allocate_cooling_tables(cooling);
//
//  /* Force a re - read of the cooling tables */
//  cooling->z_index = -10;
//  cooling->previous_z_index = eagle_cooling_N_redshifts - 2;
//  cooling_update(cosmo, cooling, /*space=*/NULL);
//}

/**
 * @brief Prints the properties of the cooling model to stdout.
 *
 * @param cooling #cooling_function_data struct.
 */
//void cooling_print_backend(const struct cooling_function_data *cooling) {
//
//  message("Cooling function is 'EAGLE'.");
//}

/**
 * @brief Clean - up the memory allocated for the cooling routines
 *
 * We simply free all the arrays.
 *
 * @param cooling the cooling data structure.
 */
void cooling_clean(struct cooling_function_data *cooling) {

    /* Free the side arrays */
    free(cooling->Redshifts);
    free(cooling->nH);
    free(cooling->Temp);
    free(cooling->HeFrac);
    free(cooling->Therm);
    free(cooling->SolarAbundances);
    free(cooling->SolarAbundances_inv);

    /* Free the tables */
    free(cooling->table.metal_heating);
    free(cooling->table.electron_abundance);
    free(cooling->table.temperature);
    free(cooling->table.H_plus_He_heating);
    free(cooling->table.H_plus_He_electron_abundance);
}

/**
 * @brief Write a cooling struct to the given FILE as a stream of bytes.
 *
 * @param cooling the struct
 * @param stream the file stream
 */
//void cooling_struct_dump(const struct cooling_function_data *cooling,
//                         FILE *stream) {
//
//  /* To make sure everything is restored correctly, we zero all the pointers to
//     tables. If they are not restored correctly, we would crash after restart on
//     the first call to the cooling routines. Helps debugging. */
//  struct cooling_function_data cooling_copy = *cooling;
//  cooling_copy.Redshifts = NULL;
//  cooling_copy.nH = NULL;
//  cooling_copy.Temp = NULL;
//  cooling_copy.Therm = NULL;
//  cooling_copy.SolarAbundances = NULL;
//  cooling_copy.SolarAbundances_inv = NULL;
//  cooling_copy.table.metal_heating = NULL;
//  cooling_copy.table.H_plus_He_heating = NULL;
//  cooling_copy.table.H_plus_He_electron_abundance = NULL;
//  cooling_copy.table.temperature = NULL;
//  cooling_copy.table.electron_abundance = NULL;
//
//  restart_write_blocks((void *)&cooling_copy,
//                       sizeof(struct cooling_function_data), 1, stream,
//                       "cooling", "cooling function");
//}

/**
 * @brief Restore a hydro_props struct from the given FILE as a stream of
 * bytes.
 *
 * Read the structure from the stream and restore the cooling tables by
 * re - reading them.
 *
 * @param cooling the struct
 * @param stream the file stream
 * @param cosmo #cosmology structure
 */
//void cooling_struct_restore(struct cooling_function_data *cooling, FILE *stream,
//                            const struct cosmology *cosmo) {
//  restart_read_blocks((void *)cooling, sizeof(struct cooling_function_data), 1,
//                      stream, NULL, "cooling function");
//
//  cooling_restore_tables(cooling, cosmo);
//}

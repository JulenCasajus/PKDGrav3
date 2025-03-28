#include "blackhole/init.h"
#include "master.h"

void MSR::SetBlackholeParam() {
    calc.dBHAccretionEddFac = parameters.get_dBHAccretionEddFac() * (1e3 / MSOLG / units.dMsolUnit) /
                              pow(100. / KPCCM / units.dKpcUnit, 3) /
                              units.dSecUnit / parameters.get_dBHRadiativeEff();
    // We precompute the factor such that we only need to multiply
    // AccretionRate by this amount to get E_feed
    calc.dBHFBEff = parameters.get_dBHFBEff() * parameters.get_dBHRadiativeEff() *
                    pow(LIGHTSPEED * 1e-5 / units.dKmPerSecUnit, 2);

    // This, in principle, will not be a parameter
    double n_heat = 1.0;

    // We convert from Delta T to energy per mass.
    // This needs to be multiplied by the mass of the gas particle
    calc.dBHFBEcrit = parameters.get_dBHFBDT() * dTuFacPrimIonised * n_heat;
}

int MSR::ValidateBlackholeParam() {
    if (parameters.get_bBHAccretion()) {
        if (parameters.get_dBHAccretionAlpha() <= 0) {
            fprintf(stderr, "ERROR: dBHAccretionAlpha should be positive."
                    "If you want to avoid boosting the Bondi accretion rate, "
                    "just set dBHAccretionAlpha = 1.0\n");
            return 0;
        }
    }
    if (parameters.get_bBHPlaceSeed() && !parameters.get_bFindGroups()) {
        parameters.set_bFindGroups(true);
        fprintf(stderr, "WARNING: Blackhole seeding requires Friends-of-friends group finding. "
                "Setting bFindGroups to true\n");
    }
    if (!parameters.get_bMemPotential()) {
        parameters.set_bMemPotential(true);
        fprintf(stderr, "WARNING: The blackhole module requires bMemPotential. "
                "Setting bMemPotential to true\n");
    }
    if (!parameters.get_bMemMass()) {
        parameters.set_bMemMass(true);
        fprintf(stderr, "WARNING: The blackhole module requires bMemMass. "
                "Setting bMemMass to true\n");
    }
    if ((parameters.get_bBHMerger() || parameters.get_bBHAccretion()) && !parameters.get_bAddDelete()) {
        parameters.set_bAddDelete(true);
        fprintf(stderr, "WARNING: Blackhole mergers and gas accretion require bAddDelete. "
                "Setting bAddDelete to true\n");
    }
    return 1;
}

void MSR::BlackholeInit(uint8_t uRungMax) {
    // We reuse this struct for simplicity
    struct inPlaceBHSeed in;

    in.uRungMax = uRungMax;

    pstBHInit(pst, &in, sizeof(in), NULL, 0);
}

void pkdBHInit(PKD pkd, uint8_t uRungMax) {
    for (auto &p : pkd->particles) {
        if (p.is_bh()) {
            p.set_rung(uRungMax);
            p.set_new_rung(uRungMax);
        }
    }
}
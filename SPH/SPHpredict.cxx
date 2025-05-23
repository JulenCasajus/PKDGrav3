/*  This file is part of PKDGRAV3 (http://www.pkdgrav.org/).
 *  Copyright (c) 2001-2018 Joachim Stadel & Douglas Potter
 *
 *  PKDGRAV3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  PKDGRAV3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with PKDGRAV3.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#else
    #include "pkd_config.h"
#endif

#include "SPHpredict.h"

float getDtPredDrift(struct pkdKickParameters *kick, int bMarked, int uRungLo, int uRung) {
    if (uRung < uRungLo) {
        return kick->dtPredDrift[uRung];
    }
    else {
        if (bMarked) {
            return - kick->dtOpen[uRung];
        }
        else {
            return kick->dtClose[uRung];
        }
    }
}

void SPHpredictOnTheFly(PKD pkd, particleStore::ParticleReference &p, struct pkdKickParameters *kick, int uRungLo, float *vpred, float *P, float *cs, float *T, SPHOptions *SPHoptions) {
    auto &NewSph = p.newsph();
    float dtPredDrift = getDtPredDrift(kick, p.marked(), uRungLo, p.rung());
    const auto &ap = p.acceleration();
    const auto &v = p.velocity();
    if (SPHoptions->doConsistentPrediction) {
        vpred[0] = NewSph.vpred[0];
        vpred[1] = NewSph.vpred[1];
        vpred[2] = NewSph.vpred[2];
    }
    else {
        vpred[0] = v[0] + dtPredDrift * ap[0];
        vpred[1] = v[1] + dtPredDrift * ap[1];
        vpred[2] = v[2] + dtPredDrift * ap[2];
        if ((SPHoptions->VelocityDamper > 0.0) && p.marked()) {
            vpred[0] *= exp(kick->dtOpen[p.rung()] * SPHoptions->VelocityDamper);
            vpred[1] *= exp(kick->dtOpen[p.rung()] * SPHoptions->VelocityDamper);
            vpred[2] *= exp(kick->dtOpen[p.rung()] * SPHoptions->VelocityDamper);
        }
    }
    if (SPHoptions->doSPHForces || SPHoptions->doDensityCorrection) {
        if (SPHoptions->doOnTheFlyPrediction) {
            float uPred = 0.0f;
            if (SPHoptions->useIsentropic && !(p.imaterial() == 0 && SPHoptions->useBuiltinIdeal)) {
                if (p.marked()) {
                    // undo kick
                    uPred = NewSph.u + dtPredDrift * NewSph.uDot;
                }
                else {
                    // undo kick
                    uPred = NewSph.u + kick->dtPredISPHUndoOpen[p.rung()] * NewSph.uDot;
                    // new opening kick
                    uPred += kick->dtPredISPHOpen[p.rung()] * NewSph.uDot;
                    // isentropic evolution
                    uPred = SPHEOSIsentropic(pkd, NewSph.oldRho, uPred, p.density(), p.imaterial(), SPHoptions);
                    // new closing kick
                    uPred += kick->dtPredISPHClose[p.rung()] * NewSph.uDot;
                }
            }
            else {
                uPred = NewSph.u + dtPredDrift * NewSph.uDot;
            }
            *P = SPHEOSPCTofRhoU(pkd, p.density(), uPred, cs, T, p.imaterial(), SPHoptions);
        }
        else {
            *P = NewSph.P;
            *cs = NewSph.cs;
            if (T) *T = NewSph.T;
        }
    }
}

void SPHpredictInDensity(PKD pkd, particleStore::ParticleReference &p, struct pkdKickParameters *kick, int uRungLo, float *P, float *cs, float *T, SPHOptions *SPHoptions) {
    // CAREFUL!! When this is called, p.marked() does not mean "has been kicked", but it is a fastgas marker
    auto &NewSph = p.newsph();
    if (SPHoptions->doUConversion && SPHoptions->doInterfaceCorrection) {
        *T = NewSph.u;
        *P = SPHEOSPofRhoT(pkd, p.density(), NewSph.u, p.imaterial(), SPHoptions);
    }
    else {
        float dtPredDrift = getDtPredDrift(kick, 0, uRungLo, p.rung());
        float uPred = 0.0f;
        if (SPHoptions->useIsentropic && !(p.imaterial() == 0 && SPHoptions->useBuiltinIdeal)) {
            // undo kick
            uPred = NewSph.u + kick->dtPredISPHUndoOpen[p.rung()] * NewSph.uDot;
            // new opening kick
            uPred += kick->dtPredISPHOpen[p.rung()] * NewSph.uDot;
            // isentropic evolution
            uPred = SPHEOSIsentropic(pkd, NewSph.oldRho, uPred, p.density(), p.imaterial(), SPHoptions);
            // new closing kick
            uPred += kick->dtPredISPHClose[p.rung()] * NewSph.uDot;
        }
        else {
            uPred = NewSph.u + dtPredDrift * NewSph.uDot;
        }
        *P = SPHEOSPCTofRhoU(pkd, p.density(), uPred, cs, T, p.imaterial(), SPHoptions);
        if (SPHoptions->doConsistentPrediction) {
            const auto &v = p.velocity();
            const auto &ap = p.acceleration();
            NewSph.vpred[0] = v[0] + dtPredDrift * ap[0];
            NewSph.vpred[1] = v[1] + dtPredDrift * ap[1];
            NewSph.vpred[2] = v[2] + dtPredDrift * ap[2];
        }
    }
}

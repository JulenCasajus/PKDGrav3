#ifndef HYDRO_H
#define HYDRO_H
#include "pkd.h"
#include "smooth/smoothfcn.h"
#include <vector>

#define XX 0
#define YY 3
#define ZZ 5
#define XY 1
#define XZ 2
#define YZ 4

template<typename T>
int sign(T v) {return (v > 0) - (v < 0);}

/* -----------------
 * MAIN FUNCTIONS AND CLASSES
 * -----------------
 */

// Cubic spline kernel as defined in Dehnen & Aly (2012)
//
// (1 − q)^3 − 4(1 / 2 − q)^3  for   0 < q <= 1 / 2
//                (1 − q)^3  for 1 / 2 < q <= 1
//                        0  for       q > 1
//
// where q = r / H.
template<typename dtype = double, typename mtype = bool>
inline dtype cubicSplineKernel(dtype r, dtype H) {
    const dtype dNormFac = 16 * M_1_PI;
    const dtype dHInv = 1.0 / H;
    const dtype q = r * dHInv;
    const dtype dNorm = dNormFac * dHInv * dHInv * dHInv;

    if (q < 0.5) {
        return dNorm * (0.5 - 3 * q * q * (1.0 - q));
    }
    else if (q < 1.0) {
        return dNorm * (1.0 - q) * (1.0 - q) * (1.0 - q);
    }
    else {
        return 0.0;
    }
}

#ifdef SIMD_H
template<>
inline dvec cubicSplineKernel(dvec r, dvec H) {
    const dvec dNormFac = 16 * M_1_PI;
    const dvec dHInv = 1.0 / H;
    const dvec q = r * dHInv;
    const dvec dNorm = dNormFac * dHInv * dHInv * dHInv;
    const dvec oneMq = 1.0 - q;
    dvec out = 0.0;
    dvec t;
    t = dNorm * oneMq * oneMq * oneMq;
    out = mask_mov(out, q < 1.0, t);
    t = dNorm * (0.5 - 3 * q * q * oneMq);
    out = mask_mov(out, q < 0.5, t);
    return out;
}
#endif

// First derivative (i.e. dW / dq) of the cubic spline kernel
template<typename dtype = double, typename mtype = bool>
inline dtype cubicSplineKernelDeriv(dtype r, dtype H) {
    const dtype dHInv = 1.0 / H;
    const dtype q = r * dHInv;
    const dtype dNorm = 48. * M_1_PI * dHInv * dHInv * dHInv;

    if (q < 0.5) {
        return dNorm * q * (3.*q - 2.);
    }
    else if (q < 1.0) {
        return - dNorm * (1.-q) * (1.-q);
    }
    else {
        return 0.0;
    }
}

#ifdef SIMD_H
template<>
inline dvec cubicSplineKernelDeriv(dvec r, dvec H) {
    const dvec dHInv = 1.0 / H;
    const dvec q = r * dHInv;
    const dvec dNorm = 48. * M_1_PI * dHInv * dHInv * dHInv;
    dvec out{0.}, t;

    t = - dNorm * (1.-q) * (1.-q);
    out = mask_mov(out, q < 1.0, t);
    t = dNorm * q * (3.*q - 2.);
    out = mask_mov(out, q < 0.5, t);
    return out;
}
#endif

/* Density loop */
struct hydroDensityPack {
    blitz::TinyVector<double, 3> position;
    float fBall;
    uint8_t iClass;
    bool bMarked;
};

void hydroDensity(PARTICLE *p, float fBall, int nSmooth, NN *nnList, SMF *smf);
void hydroDensity_node(PKD pkd, SMF *smf, Bound bnd_node, const std::vector<PARTICLE *> &sinks,
                       NN *nnList, int nCnt);
void hydroDensityFinal(PARTICLE *p, float fBall, int nSmooth, NN *nnList, SMF *smf);
void packHydroDensity(void *vpkd, void *dst, const void *src);
void unpackHydroDensity(void *vpkd, void *dst, const void *src);

/* Gradient loop */
struct hydroGradientsPack {
    blitz::TinyVector<double, 3> position;
    blitz::TinyVector<double, 3> velocity;
    double P;
    float fBall;
    float fDensity;
    uint8_t iClass;
    bool bMarked;
};

void hydroGradients(PARTICLE *p, float fBall, int nSmooth, NN *nnList, SMF *smf);
void packHydroGradients(void *vpkd, void *dst, const void *src);
void unpackHydroGradients(void *vpkd, void *dst, const void *src);

/* Flux loop */
struct hydroFluxesPack {
    blitz::TinyVector<double, 3> position;
    blitz::TinyVector<double, 3> velocity;
    blitz::TinyVector<double, 6> B;
    blitz::TinyVector<meshless::myreal, 3> gradRho, gradVx, gradVy, gradVz, gradP;
    meshless::myreal lastUpdateTime;
    blitz::TinyVector<meshless::myreal, 3> lastAcc;
    double omega;
    double spheom;
    double P;
    float fBall;
    float fDensity;
    uint8_t uRung;
    uint8_t iClass;
    bool bMarked;
};

struct hydroFluxesFlush {
    meshless::myreal Frho;
    blitz::TinyVector<meshless::myreal, 3> Fmom;
    meshless::myreal Fene;
#ifndef USE_MFM
    blitz::TinyVector<double, 3> drDotFrho;
#endif
    blitz::TinyVector<double, 3> mom;
    double E;
    double Uint;
#ifndef USE_MFM
    float fMass;
#endif
};

void packHydroFluxes(void *vpkd, void *dst, const void *src);
void unpackHydroFluxes(void *vpkd, void *dst, const void *src);
void initHydroFluxes(void *vpkd, void *vp);
void initHydroFluxesCached(void *vpkd, void *vp);
void hydroRiemann(PARTICLE *p, float fBall, int nSmooth, int nBuff,
                  meshless::myreal *restrict input_buffer,
                  meshless::myreal *restrict output_buffer,
                  SMF *smf);
void hydroRiemann_wrapper(PARTICLE *p, float fBall, int nSmooth, int nBuff,
                          meshless::myreal *restrict input_buffer,
                          meshless::myreal *restrict output_buffer,
                          SMF *smf);
void flushHydroFluxes(void *vpkd, void *dst, const void *src);
void combHydroFluxes(void *vpkd, void *p1, const void *p2);
void hydroFluxFillBuffer(meshless::myreal *input_buffer, PARTICLE *q, int i, int nBuff,
                         double dr2, blitz::TinyVector<double, 3> dr, SMF *);
void hydroFluxUpdateFromBuffer(meshless::myreal *output_buffer, meshless::myreal *input_buffer,
                               PARTICLE *p, PARTICLE *q, int i, int nBuff, SMF *);
void hydroFluxGetBufferInfo(int *in, int *out);

/* Time step loop */
struct hydroStepPack {
    blitz::TinyVector<double, 3> position;
    blitz::TinyVector<double, 3> velocity;
    float c;
    float fBall;
    uint8_t uRung;
    uint8_t uNewRung;
    uint8_t iClass;
    bool bMarked;
};

struct hydroStepFlush {
    uint8_t uNewRung;
    uint8_t uWake;
};

void hydroStep(PARTICLE *p, float fBall, int nSmooth, NN *nnList, SMF *smf);
void packHydroStep(void *vpkd, void *dst, const void *src);
void unpackHydroStep(void *vpkd, void *dst, const void *src);
void initHydroStep(void *vpkd, void *dst);
void flushHydroStep(void *vpkd, void *dst, const void *src);
void combHydroStep(void *vpkd, void *dst, const void *src);
void pkdWakeParticles(PKD pkd, int iRoot, double dTime, double dDelta);

/* Source terms */
void hydroSourceGravity(PKD pkd, particleStore::ParticleReference &p, meshless::FIELDS *psph,
                        double pDelta, blitz::TinyVector<double, 3> &pa, double dScaleFactor,
                        int bComove);
void hydroSourceExpansion(PKD pkd, particleStore::ParticleReference &p, meshless::FIELDS *psph,
                          double pDelta, double dScaleFactor, double dHubble,
                          int bComove, double dConstGamma);
void hydroSyncEnergies(PKD pkd, particleStore::ParticleReference &p, meshless::FIELDS *psph,
                       const blitz::TinyVector<double, 3> &pa, double dConstGamma);
void hydroSetPrimitives(PKD pkd, particleStore::ParticleReference &p, meshless::FIELDS *psph,
                        double dTuFac, double dConstGamma);
void hydroSetLastVars(PKD pkd, particleStore::ParticleReference &p, meshless::FIELDS *psph,
                      const blitz::TinyVector<double, 3> &pa, double dScaleFactor,
                      double dTime, double dDelta, double dConstGamma);
void hydroResetFluxes(meshless::FIELDS *psph);

/* -----------------
 * HELPERS
 * -----------------
 */
#define SIGN(x) (((x) > 0) ? 1 : (((x) < 0) ? -1 : 0) )
#define MIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)  ((X) > (Y) ? (X) : (Y))
void inverseMatrix(double *E, double *B);
double conditionNumber(double *E, double *B);
void compute_Ustar(double rho_K, double S_K, double v_K,
                   double p_K, double h_K, double S_s,
                   double *rho_sK, double *rhov_sK, double *e_sK);

#endif

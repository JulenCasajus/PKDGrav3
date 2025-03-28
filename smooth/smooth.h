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

#ifndef SMOOTH_HINCLUDED
#define SMOOTH_HINCLUDED

extern "C" {
#include "listcomp.h"
}
#include "pkd.h"
#include "smooth/smoothfcn.h"
#include "hydro/hydro.h"
#include "group/group.h"
#include "blitz/array.h"

#define NNLIST_INCREMENT    200     /* number of extra neighbor elements added to nnList */

struct hashElement {
    void *p;
    struct hashElement *coll;
};

struct smExtraArray {
    uint32_t iIndex;
    char bDone;
};

typedef struct smContext {
    PKD pkd;
    PARTICLE *pSentinel;
    void (*fcnSmooth)(PARTICLE *, float, int, NN *, SMF *);
    void (*fcnSmoothNode)(PARTICLE *, float, int, int, double *, double *, SMF *);
    void (*fcnSmoothGetBufferInfo)(int *, int *);
    void (*fcnSmoothFillBuffer)(double *, PARTICLE *, int, int,
                                double, blitz::TinyVector<double, 3>, SMF *);
    void (*fcnSmoothUpdate)(double *, double *, PARTICLE *, PARTICLE *, int, int, SMF *);
    int nSmooth;
    int nQueue;
    int bPeriodic;
    int bOwnCache;
    int bSymmetric;
    int iSmoothType;
    int bSearchGasOnly;
    blitz::TinyVector<double, 3> rLast; /* For the snake */
    PQ *pq;
    /*
    ** Flags to mark local particles which are inactive either because they
    ** are source inactive or because they are already present in the prioq.
    ** In this extra array is also space for a queue of particles, needed
    ** for the fast gas routines or for friends-of-friends.
    ** This will point to the pLite array, so it will be destroyed after a tree
    ** build or domain decomposition.
    */
    struct smExtraArray *ea;
    /*
    ** Flags to mark local particles which are finished in the processing
    ** of the smFastGas routine. They have updated their densities.
    */
    char *bDone;
    /*
    ** Hash table to indicate whether a remote particle is already present in the
    ** priority queue.
    */
    int nHash;  /* should be a prime number > nSmooth */
    struct hashElement *pHash;
    struct hashElement *pFreeHash;
    int nnListSize;
    int nnListMax;
    NN *nnList;
    int *S;
    double *Smin;
    /*
     ** Also need the stacks for the tree search
     */
    struct stStack {
        int id;
        int iCell;
        double min;
    } *ST;
    /*
    ** Context for nearest neighbor lists.
    */
    LCODE lcmp;
    /*
    ** Some variables needed for smNewFof().
    */
    uint32_t iHead;
    uint32_t iTail;
    int  *Fifo;
} *SMX;

#ifdef __cplusplus
extern "C" {
#endif
int smInitialize(SMX *psmx, PKD pkd, SMF *smf, int nSmooth,
                 int bPeriodic, int bSymmetric, int iSmoothType);
int smInitializeRO(SMX *psmx, PKD pkd, SMF *smf, int nSmooth,
                   int bPeriodic, int iSmoothType);
void smFinish(SMX, SMF *);
void smSmoothInitialize(SMX smx);
void smSmoothFinish(SMX smx);
float smSmoothSingle(SMX smx, SMF *smf, particleStore::ParticleReference &p, int iRoot1, int iRoot2);
void smSmooth(SMX, SMF *);
void smReSmoothSingle(SMX smx, SMF *smf, particleStore::ParticleReference &p, double fBall);
int  smReSmooth(SMX, SMF *, int);
#ifdef OPTIM_SMOOTH_NODE
int  smReSmoothNode(SMX, SMF *, int);
void buildInteractionList(SMX smx, SMF *smf, KDN *node, Bound bnd_node, int *nCnt, blitz::TinyVector<double, 3> r, int ix, int iy, int iz);
#endif

void smGather(SMX smx, double fBall2, blitz::TinyVector<double, 3> r);

inline void smSwapNN(NN *nnList, int i, int j) {
    NN temp;
    memcpy(&temp,      &nnList[i], sizeof(NN));
    memcpy(&nnList[i], &nnList[j], sizeof(NN));
    memcpy(&nnList[j], &temp,      sizeof(NN));
}

#ifdef __cplusplus
}
#endif

#endif

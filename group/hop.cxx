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
#include "hop.h"
#include <math.h>
#include <algorithm>

static void initJoinLoops(void *vctx, void *v) {}
static void combJoinLoops(void *vctx, void *v1, const void *v2) {
    SMF *smf = (SMF *)vctx;
    GHtmpGroupTable *g1 = (GHtmpGroupTable *)v1;
    const GHtmpGroupTable *g2 = (const GHtmpGroupTable *)v2;
    if ( g1->iPid > g2->iPid || (g1->iPid == g2->iPid && g1->iIndex > g2->iIndex) ) {
        g1->iPid = g2->iPid;
        g1->iIndex = g2->iIndex;
        smf->bDone = 0;
    }
}

static void initSetArc(void *vpkd, void *v) {}
static void combSetArc(void *vpkd, void *v1, const void *v2) {
    PKD pkd = (PKD) vpkd;
    PARTICLE *p1 = (PARTICLE *)v1;
    const PARTICLE *p2 = (const PARTICLE *)v2;
    if (p2->bMarked) p1->bMarked = 1;
    assert( pkdGetGroup(pkd, p1) == pkdGetGroup(pkd, p2) );
}

/*
** Follow the link for the given particle
*/
static int traverseLink(PKD pkd, struct smGroupArray *ga, int pi,
                        int *iPid2, int *iIndex2, int *iMinPartPid, int *iMinPartIndex) {
    MDL mdl = pkd->mdl;
    int gid2;
    GHtmpGroupTable *g2;
    int bDone = 0;
    int bRemote = *iPid2 != pkd->Self();

    auto p = (!bRemote) ? pkd->particles[*iIndex2]
             : pkd->particles[static_cast<PARTICLE *>(mdlFetch(pkd->mdl, CID_PARTICLE, *iIndex2, *iPid2))];
    gid2 = p.group();
    g2 = static_cast<GHtmpGroupTable *>(mdlAcquire(pkd->mdl, CID_GROUP, gid2, *iPid2));

    /* Update the minimum group id in case we need it below */
    if (iMinPartPid && (*iPid2 < *iMinPartPid || (*iPid2 == *iMinPartPid && *iIndex2 < *iMinPartIndex))) {
        *iMinPartPid = *iPid2;
        *iMinPartIndex = *iIndex2;
    }

    /*
    ** This group points to the same node, so it is a terminal group.
    ** The particle ids may not match, but the group will point to
    ** a particle that points back to this group.
    */
    if (g2->iPid == *iPid2) {
        ga[pi].id.iPid   = g2->iPid;
        ga[pi].id.iIndex = g2->iIndex;
        bDone = 1;
    }
    /*
    ** If we have linked back to our group on our node, then we are done.
    ** We choice the lowest gid we have seen so far. We also have to mark
    ** the remote particle that started this, because it starts an arc!
    */
    else if (*iPid2 == mdlSelf(mdl) && gid2 == pi) { /*LOOP*/
        auto p2 = pkd->particles[static_cast<PARTICLE *>(mdlAcquire(pkd->mdl, CID_PARTICLE, ga[pi].id.iIndex, ga[pi].id.iPid))];
        p2.set_marked(true);
        mdlRelease(pkd->mdl, CID_PARTICLE, &p2);
        assert(ga[pi].id.iPid != pkd->Self());
        ga[pi].id.iPid = *iMinPartPid;
        ga[pi].id.iIndex = *iMinPartIndex;
        bDone = 1;
    }

    /* Follow the link - ignored if bDone is set. */
    *iPid2 = g2->iPid;
    *iIndex2 = g2->iIndex;

    mdlRelease(pkd->mdl, CID_GROUP, g2);

    return bDone;
}

/*
** Link particles based on density gradients
** After we finish, all "chains" will be complete globally across all domains.
** Chains still need to be joined if their terminating loops are close together.
** Prerequisites: Density and fBall from smooth
*/
int smHopLink(SMX smx, SMF *smf) {
    PKD pkd = smx->pkd;
    MDL mdl = pkd->mdl;
    int pi, j, gid, nLoop, nSpur;
    int nGroups, nLocal, nRemote;
    int iIndex1, iIndex2, iMinPartIndex, iPid1, iPid2, iMinPartPid, iParticle;
    struct smGroupArray *ga;
    /* Next particle in the chain or -1 if at the end in which case */
    /* the next particle will be remote and can be found in the group array */
    uint32_t *pl;

    pkd->ga = ga = (struct smGroupArray *)(pkd->pLite);
    pl = (uint32_t *)(((char *)pkd->pLite + 1ul * pkd->Local()*pkd->EphemeralBytes()) - (pkd->Local()+1)*sizeof(uint32_t));
    assert((uint32_t *)ga < pl);

    ga[0].iGid = 0;
    ga[0].id.iPid = mdlSelf(mdl);
    ga[0].id.iIndex = -1; /* Sentinel */
    for (pi = 0; pi < pkd->Local(); ++pi) {
        auto p = pkd->particles[pi];
        p.set_group(0);     // Ungrouped
        p.set_marked(true); // Used by smooth to determine active particles
    }
    smSmoothInitialize(smx);

    pkd->nGroups = 0;
    pkd->nLocalGroups = 0;
    pkd->tmpHopGroups = NULL;

    /*
    ** Run through all of the particles, and find a particle to link to
    ** by calculating the gradient and looking in that direction. We make
    ** a link to this particle, then follow the link. Once we have reached
    ** the end of the chain, we consider the next particle (which may have
    ** been already done in which case we skip it).
    */
    nGroups = 1;
    nSpur = nLoop = nRemote = 0;
    for (pi = 0; pi < pkd->Local(); ++pi) {
        auto p = pkd->particles[iParticle = pi];
        if ( p.group() > 0 ) continue; /* Already done (below) */
        assert((uint32_t *)&ga[nGroups + 1] < pl);
        ga[nGroups].iGid = nGroups;
        for (;;) {
            p.set_group(nGroups);
            /* Need: fDensity and fBall but pl[iParticle] is not yet used */
            /* fDensity MUST persist but fBall could be stashed into pl[] */
            smReSmoothSingle(smx, smf, p, p.ball());
            /* Done: fBall */
            ga[nGroups].id = smf->hopParticleLink;
            /*
            ** Call mdlCacheCheck to make sure we are making progress!
            */
            mdlCacheCheck(mdl);
            /*
            ** If we link to a remote node, then this group chain is done for now.
            ** Later we will merge remote groups.
            **/
            if (smf->hopParticleLink.iPid != pkd->Self() ) { ++nGroups; ++nRemote; break; }
            else {
                iParticle = pl[iParticle]=smf->hopParticleLink.iIndex;
                p = pkd->particles[iParticle];
                gid = p.group();
                /* We link to ourselves: this forms a new "loop" */
                if (gid == nGroups) { ++nGroups; ++nLoop; break; }
                /* Ok, some other group: merge this "spur" on to it */
                else if (gid > 0) {
                    for (j = pi; j != iParticle; j = pl[j]) {
                        p = pkd->particles[j];
                        p.set_group(gid);
                    }
                    ++nSpur;
                    break;
                }
            }
            /* None of the above? Just keep following the links. */
        }
    }
    smSmoothFinish(smx);
    /*
    ** All chains now terminate at a specific *particle*, either local or remote.
    ** Spurs were automatically joined, and point to their corresponding "loop".
    ** This "loop" particle is guaranteed to be part of the terminating loop.
    ** Remote particles are still pending, so we combine duplicates so there are
    ** fewer remote searches to perform. This results in substantially fewer chains.
    */
    mdlFinishCache(mdl, CID_PARTICLE);
    nGroups = pkdGroupCombineDuplicateIds(pkd, nGroups, ga, 0);
    mdlROcache(mdl, CID_PARTICLE, NULL, pkd->particles, pkd->particles.ParticleSize(), pkd->Local());
    /*
    ** Now lets go looking for remote loops and see if we are part of them.
    ** The group table still contains PARTICLE links, so we use it to
    ** accelerate the search. We will find three cases:
    **  1. We find a terminating loop. Update the link to this loop.
    **  2. We are part of a terminating loop. Update the link to the lowest
    **     loop id and we are done. All members will do this.
    **  3. We encounter a loop that we are not a part of. Save any member
    **     of the loop. Later, we will fetch the correct id from it.
    */

    /* We will update bMarked (an arc) remotely, so we need a combiner cache */
    mdlFinishCache(mdl, CID_PARTICLE);
    for (auto &p : pkd->particles) p.set_marked(false); // Not an arc (yet)
    mdlCOcache(pkd->mdl, CID_PARTICLE, NULL,
               pkd->particles, pkd->particles.ParticleSize(),
               pkd->Local(), pkd, initSetArc, combSetArc);
    pkd->tmpHopGroups = static_cast<GHtmpGroupTable *>(mdlMalloc(mdl, nGroups * sizeof(GHtmpGroupTable)));
    assert(pkd->tmpHopGroups != NULL);
    for (pi = 1; pi < nGroups; ++pi) {
        pkd->tmpHopGroups[pi].iPid   = ga[pi].id.iPid;
        pkd->tmpHopGroups[pi].iIndex = ga[pi].id.iIndex;
    }
    mdlROcache(mdl, CID_GROUP, NULL, pkd->tmpHopGroups, sizeof(GHtmpGroupTable), nGroups);
    for (pi = 1; pi < nGroups; ++pi) {
        /* Completely local loop: any particle can start the "arc" */
        if (ga[pi].id.iPid == pkd->Self()) {
            pkd->particles[ga[pi].id.iIndex].set_marked(true);
        }
        /* Remote: may form a loop, or we may simply be a spur */
        else {
            iMinPartPid = iPid1 = iPid2 = ga[pi].id.iPid;
            iMinPartIndex = iIndex1 = iIndex2 = ga[pi].id.iIndex;
            for (;;) {
                /* We use Floyd's cycle - finding algorithm here */
                if (traverseLink(pkd, ga, pi, &iPid2, &iIndex2, &iMinPartPid, &iMinPartIndex)) break;
                if (traverseLink(pkd, ga, pi, &iPid2, &iIndex2, &iMinPartPid, &iMinPartIndex)) break;
                traverseLink(pkd, ga, pi, &iPid1, &iIndex1, NULL, NULL);
                /*
                ** If we see a loop here, then we are not a part of it (we are a spur).
                ** We point ourselves at one of the loop members so we can later
                ** retrieve the proper terminating particle. We can be pointing
                ** to our node of course, but to a different group.
                */
                if (iPid1 == iPid2 && iIndex1 == iIndex2) {
                    ga[pi].id.iPid   = iPid1;
                    ga[pi].id.iIndex = iIndex1;
                    break;
                }
            }
        }
    }
    mdlFinishCache(mdl, CID_GROUP);
    /*
    ** By now, we have marked the particle that starts each arc on the node,
    ** and exactly one particle if the group is a local loop.
    ** We now want to mark the rest of the particles in the arcs and loops.
    */
    mdlFinishCache(mdl, CID_PARTICLE);
    for (pi = 0; pi < pkd->Local(); ++pi) {
        auto p = pkd->particles[pi];
        if (!p.marked()) continue;
        iIndex1 = pl[pi];
        while (iIndex1!=-1 && !(p = pkd->particles[iIndex1]).marked()) {
            p.set_marked(true);
            iIndex1 = pl[iIndex1];
        }
    }
    /* Done: pl[] is no longer needed */
    pl = NULL;

    nGroups = pkdGroupCombineDuplicateIds(pkd, nGroups, ga, 0);
    /*
    ** Now handle deferred groups (ones pointing to, but not part of, loops).
    */
    for (pi = 1; pi < nGroups; ++pi) {
        pkd->tmpHopGroups[pi].iPid   = ga[pi].id.iPid;
        pkd->tmpHopGroups[pi].iIndex = ga[pi].id.iIndex;
    }
    mdlROcache(mdl, CID_PARTICLE, NULL, pkd->particles, pkd->particles.ParticleSize(), pkd->Local());
    mdlROcache(mdl, CID_GROUP, NULL, pkd->tmpHopGroups, sizeof(GHtmpGroupTable), nGroups);
    nSpur = 0;
    for (pi = 1; pi < nGroups; ++pi) {
        auto p1 = static_cast<PARTICLE *>(mdlAcquire(mdl, CID_PARTICLE, ga[pi].id.iIndex, ga[pi].id.iPid));
        int gid1 = pkdGetGroup(pkd, p1);
        auto g1 = static_cast<GHtmpGroupTable *>(mdlAcquire(mdl, CID_GROUP, gid1, ga[pi].id.iPid));
        auto p2 = static_cast<PARTICLE *>(mdlAcquire(mdl, CID_PARTICLE, g1->iIndex, g1->iPid));
        int gid2 = pkdGetGroup(pkd, p2);
        auto g2 = static_cast<GHtmpGroupTable *>(mdlAcquire(mdl, CID_GROUP, gid2, g1->iPid));
        assert (g2->iPid == g1->iPid);
        ga[pi].id.iPid   = g2->iPid;
        ga[pi].id.iIndex = g2->iIndex;
        mdlRelease(mdl, CID_GROUP, g2);
        mdlRelease(mdl, CID_PARTICLE, p2);
        mdlRelease(mdl, CID_GROUP, g1);
        mdlRelease(mdl, CID_PARTICLE, p1);
    }
    mdlFinishCache(mdl, CID_GROUP);

    /* If we end up point on - node, then we might have to combine the groups */
    for (pi = 1; pi < nGroups; ++pi) {
        if (ga[pi].id.iPid == pkd->Self()) {
            gid = pkd->particles[ga[pi].id.iIndex].group();
            if (gid != pi) {
                assert(ga[gid].id.iPid == pkd->Self());
                ga[pi].iGid = pi;
                ga[pi].id.iPid = ga[gid].id.iPid;
                ga[pi].id.iIndex = ga[gid].id.iIndex;
            }
        }
    }
    mdlFinishCache(mdl, CID_PARTICLE);
    nGroups = pkdGroupCombineDuplicateIds(pkd, nGroups, ga, 0);
    mdlROcache(mdl, CID_PARTICLE, NULL, pkd->particles, pkd->particles.ParticleSize(), pkd->Local());

    /*
    ** Whew. The situation for groups is now one of the following.
    **  1. Groups that point locally form a local terminal loop or arc,
    **     so it must point to itself.
    **  2. Groups that point to a remote group must point to a remote
    **     group that points to itself.
    ** All particles that form a loop (either local or remote) have
    ** been marked (bMarked = 1).
    */

    /* Turn particle IDs into group IDs */
    for (pi = 1; pi < nGroups; ++pi) {
        pkd->tmpHopGroups[pi].iPid   = ga[pi].id.iPid;
        pkd->tmpHopGroups[pi].iIndex = ga[pi].id.iIndex;
    }
    mdlROcache(mdl, CID_GROUP, NULL, pkd->tmpHopGroups, sizeof(GHtmpGroupTable), nGroups);
    for (pi = 1; pi < nGroups; ++pi) {
        if (ga[pi].id.iPid == pkd->Self()) {
            gid = pkd->particles[ga[pi].id.iIndex].group();
            assert(gid == pi); /* I point to myself! */
        }
        else {
            auto p = pkd->particles[static_cast<PARTICLE *>(mdlAcquire(mdl, CID_PARTICLE, ga[pi].id.iIndex, ga[pi].id.iPid))];
            gid = p.group();
            mdlRelease(mdl, CID_PARTICLE, &p);
            auto g = static_cast<GHtmpGroupTable *>(mdlAcquire(mdl, CID_GROUP, gid, ga[pi].id.iPid));
            /* The remote particle must point to itself */
            assert(g->iPid == ga[pi].id.iPid && g->iIndex == ga[pi].id.iIndex);
            mdlRelease(mdl, CID_GROUP, g);
        }
        ga[pi].id.iIndex = gid;
        assert(ga[pi].id.iPid != pkd->Self() || ga[pi].id.iIndex ==pi);
    }
    mdlFinishCache(mdl, CID_GROUP);

    pkd->nGroups = nGroups;
    nLocal = nRemote = 0;
    for (pi = 1; pi < pkd->nGroups; ++pi) {
        if (ga[pi].id.iPid == mdlSelf(mdl)) nLocal++;
        else nRemote++;
        pkd->tmpHopGroups[pi].iPid   = ga[pi].id.iPid;
        pkd->tmpHopGroups[pi].iIndex = ga[pi].id.iIndex;
    }
    pkd->nLocalGroups = nLocal;

    /* tmpHopGroups is still allocated: freed in pkdHopFinishUp */
    return nLocal;
}

/*
** Join together chains if their terminating loops are withing a linking length.
**
** OPTIMIZATION: we should mark individual groups as done / not done so we can skip
**               subsequent iterations for those groups.
*/
int smHopJoin(SMX smx, SMF *smf, double dHopTau, int *nLocal) {
    PKD pkd = smx->pkd;
    MDL mdl = pkd->mdl;
    auto pRoot = pkd->tree[ROOT];
    struct smGroupArray *ga = pkd->ga;
    int pi;

    smf->bDone = 1;
    mdlCOcache(mdl, CID_GROUP, NULL, pkd->tmpHopGroups, sizeof(GHtmpGroupTable), pkd->nGroups,
               smf, initJoinLoops, combJoinLoops );
    /*
    ** Check all particles that are part of a loop or an arc (they are marked).
    ** We have contructed a tree with only marked particles, so just check those.
    */
    for (auto &p : *pRoot) {
        assert(p.marked());
        float fBall = (dHopTau < 0.0) ? -dHopTau * p.soft() : dHopTau;
        fBall = std::max(fBall, p.ball()*0.5f);
        smReSmoothSingle(smx, smf, p, fBall);
    }
    mdlFinishCache(mdl, CID_GROUP);

    /* Follow the chains to the end */
    mdlROcache(mdl, CID_GROUP, NULL, pkd->tmpHopGroups, sizeof(GHtmpGroupTable), pkd->nGroups);
    for (pi = 1; pi < pkd->nGroups; ++pi) {
        int iPid = pkd->Self();
        int iIndex = pi;
        int iNextPid = pkd->tmpHopGroups[iIndex].iPid;
        int iNextIndex = pkd->tmpHopGroups[iIndex].iIndex;
        while (iPid != iNextPid || iIndex != iNextIndex) {
            auto g = static_cast<GHtmpGroupTable *>(mdlAcquire(mdl, CID_GROUP, iNextIndex, iNextPid));
            iPid = iNextPid;
            iIndex = iNextIndex;
            iNextPid = g->iPid;
            iNextIndex = g->iIndex;
            mdlRelease(mdl, CID_GROUP, g);
        }
        pkd->tmpHopGroups[pi].iPid = iPid;
        pkd->tmpHopGroups[pi].iIndex = iIndex;
    }
    mdlFinishCache(mdl, CID_GROUP);

    /* Merge duplicates for the next round */
    for (pi = 1; pi < pkd->nGroups; ++pi) {
        ga[pi].iGid   = pi;
        ga[pi].id.iPid   = pkd->tmpHopGroups[pi].iPid;
        ga[pi].id.iIndex = pkd->tmpHopGroups[pi].iIndex;
    }
    mdlFinishCache(mdl, CID_PARTICLE);
    pkd->nGroups = pkdGroupCombineDuplicateIds(pkd, pkd->nGroups, ga, 1);
    mdlROcache(mdl, CID_PARTICLE, NULL, pkd->particles, pkd->particles.ParticleSize(), pkd->Local());
    int nRemote = 0;
    *nLocal = 0;
    for (pi = 1; pi < pkd->nGroups; ++pi) {
        if (ga[pi].id.iPid == mdlSelf(mdl)) (*nLocal)++;
        else nRemote++;
        pkd->tmpHopGroups[pi].iPid   = ga[pi].id.iPid;
        pkd->tmpHopGroups[pi].iIndex = ga[pi].id.iIndex;
    }
    pkd->nLocalGroups = *nLocal;
    return smf->bDone;
}


int pkdHopFinishUp(PKD pkd, int nMinGroupSize, int bPeriodic, blitz::TinyVector<double, 3> dPeriod) {
    MDL mdl = pkd->mdl;
    struct smGroupArray *ga = (struct smGroupArray *)(pkd->pLite);
    int i;

    for (i = 1; i < pkd->nGroups; ++i) {
        ga[i].iGid = i;
        ga[i].id.iPid = pkd->tmpHopGroups[i].iPid;
        ga[i].id.iIndex = pkd->tmpHopGroups[i].iIndex;
    }
    mdlFree(mdl, pkd->tmpHopGroups); pkd->tmpHopGroups = NULL;

    pkd->nGroups = pkdPurgeSmallGroups(pkd, pkd->nGroups, ga, nMinGroupSize);
    pkd->hopSavedRoots = 0;
    return pkd->nLocalGroups;
}


static void initHopGetRoots(void *vpkd, void *v) {
    HopGroupTable *g = (HopGroupTable *)v;
    g->rmt.iPid = -1;
    g->rmt.iIndex = 0;
    g->iGlobalId = 0;
}

static void combHopGetRoots(void *vctx, void *v1, const void *v2) {
    PKD pkd = (PKD)vctx;
    HopGroupTable *g1 = (HopGroupTable *)v1;
    const HopGroupTable *g2 = (const HopGroupTable *)v2;

    if (g2->iGlobalId) {
        int gid = g1 - pkd->hopGroups;
        int iRoot = pkd->hopRootIndex[gid]++;
        assert(gid < pkd->nLocalGroups); /* We should only get data for local groups */
        assert(pkd->hopRoots[iRoot].iPid==-1 && pkd->hopRoots[iRoot].iIndex==-1);
        pkd->hopRoots[iRoot].iPid = g2->rmt.iPid;
        pkd->hopRoots[iRoot].iIndex = g2->rmt.iIndex;
    }
}

void pkdHopTreeBuild(PKD pkd, int nBucket, int nGroup) {
    MDL mdl = pkd->mdl;
    int i, gid, iRoot;
    int nRootsTotal;

    /*
    ** The remote groups should be sorted by processor and remote group id,
    ** but we will rely on this behaviour later so we verify it now.
    */
    for (i = 2 + pkd->nLocalGroups; i < pkd->nGroups; ++i) {
        assert( pkd->hopGroups[i - 1].id.iPid < pkd->hopGroups[i].id.iPid
                || ( pkd->hopGroups[i - 1].id.iPid == pkd->hopGroups[i].id.iPid
                     && pkd->hopGroups[i - 1].id.iIndex < pkd->hopGroups[i].id.iIndex) );
    }

    /* Setup the buffer for tree roots */
    pkd->hopRootIndex = new int[pkd->nGroups + 1];

    /* Calculate the index into the tree roots table */
    nRootsTotal = 0;
    for (i = 1; i < pkd->nGroups; ++i) {
        pkd->hopRootIndex[i] = nRootsTotal;
        pkd->hopGroups[i].iAllRoots = nRootsTotal;
        nRootsTotal += 1 + pkd->hopGroups[i].nRemote;
    }

    pkd->hopRoots = new remoteID[nRootsTotal + 1];

    /* Invalidate all entries (for debug checks) */
    for (iRoot = 0; iRoot <= nRootsTotal; ++iRoot)
        pkd->hopRoots[iRoot].iPid = pkd->hopRoots[iRoot].iIndex = -1;

    /* We build a tree of each group of particles - hopGroups[].iTreeRoot is set */
    pkdTreeBuildByGroup(pkd, nBucket, nGroup);

    /* Add ourselves for local groups */
    for (gid = 1; gid < pkd->nLocalGroups; ++gid) {
        iRoot = pkd->hopRootIndex[gid]++;
        assert(iRoot < nRootsTotal);
        assert(pkd->hopRoots[iRoot].iPid==-1 && pkd->hopRoots[iRoot].iIndex==-1);
        pkd->hopRoots[iRoot].iPid = mdlSelf(mdl);
        pkd->hopRoots[iRoot].iIndex = i;
    }

    /*
    ** Let the group master know about the remote tree roots. We can use the
    ** combiner cache because there is at most 1 root per group per node.
    */
    mdlCOcache(mdl, CID_GROUP, NULL, pkd->hopGroups, sizeof(HopGroupTable), pkd->nGroups,
               pkd, initHopGetRoots, combHopGetRoots );
    for (i = 1 + pkd->nLocalGroups; i < pkd->nGroups; ++i) {
        assert(pkd->hopGroups[i].id.iPid != pkd->Self());
//  if (pkd->hopNumRoots[i] == 0) continue;
        auto g = static_cast<HopGroupTable *>(mdlAcquire(mdl, CID_GROUP, pkd->hopGroups[i].id.iIndex, pkd->hopGroups[i].id.iPid));
        g->rmt.iPid = pkd->Self();
        g->rmt.iIndex = pkd->hopGroups[i].iTreeRoot;
        g->iGlobalId = i;
        pkd->hopGroups[i].iAllRoots = g->iAllRoots; /* We will fix this later! */
        mdlRelease(mdl, CID_GROUP, g);
    }
    mdlFinishCache(mdl, CID_GROUP);

    /*
    ** Now we just need to fetch all of the roots for remote groups
    */
    mdlROcache(mdl, CID_GROUP, NULL, pkd->hopRoots, sizeof(*pkd->hopRoots), nRootsTotal);
    for (gid = 1 + pkd->nLocalGroups; gid < pkd->nGroups; ++gid) {
        int iIndex = pkd->hopGroups[gid].iAllRoots;
        for (i = 0; i < pkd->hopGroups[gid].nRemote + 1; ++i) {
            auto pRoot = static_cast<remoteID *>(mdlFetch(mdl, CID_GROUP, iIndex + i, pkd->hopGroups[gid].id.iPid));
            int iRoot = pkd->hopRootIndex[gid]++;
            assert(pkd->hopRoots[iRoot].iPid==-1 && pkd->hopRoots[iRoot].iIndex==-1);
            pkd->hopRoots[iRoot].iPid = pRoot->iPid;
            pkd->hopRoots[iRoot].iIndex = pRoot->iIndex;
        }
    }
    mdlFinishCache(mdl, CID_GROUP);

    /* Fix the indexes */
    nRootsTotal = 0;
    for (gid = 1; gid < pkd->nGroups; ++gid) {
        pkd->hopGroups[gid].iAllRoots = nRootsTotal;
        nRootsTotal += 1 + pkd->hopGroups[gid].nRemote;
    }

    for (iRoot = 1; iRoot < nRootsTotal; ++iRoot)
        assert(pkd->hopRoots[iRoot].iPid >= 0 && pkd->hopRoots[iRoot].iIndex > 0);

    for (gid = 1; gid < pkd->nGroups; ++gid) {
        int iRootUs = pkd->hopRootIndex[gid];
        for (iRoot = pkd->hopRootIndex[gid]; iRoot < pkd->hopRootIndex[gid + 1]; ++iRoot) {
            assert(iRoot < nRootsTotal);
            assert(pkd->hopRoots[iRoot].iPid >= 0 && pkd->hopRoots[iRoot].iIndex > 0);
            /* Move our local root to the start */
            if (pkd->hopRoots[iRoot].iPid == pkd->Self() && iRoot > iRootUs) {
                remoteID t;
                assert(pkd->hopRoots[iRootUs].iPid != pkd->Self());
                t = pkd->hopRoots[iRootUs];
                pkd->hopRoots[iRootUs] = pkd->hopRoots[iRoot];
                pkd->hopRoots[iRoot] = t;
            }
        }
    }
    delete [] pkd->hopRootIndex;
    pkd->hopRootIndex = NULL;
    /* free: We have allocated pkd->hopRoots -- needed for gravity */
}

typedef struct EnergyElement {
    double dPot;
    double dSqrtKin;
    double dTot;
    int i;
} EE;


static int cmpEE(const void *p1, const void *p2) {
    EE *a = (EE *)p1;
    EE *b = (EE *)p2;
    if (a->dTot > b->dTot) return 1;
    if (a->dTot < b->dTot) return -1;
    return 0;
}

static void initMaxHopEnergy(void *vpkd, void *v) {}
static void combMaxHopEnergy(void *vctx, void *v1, const void *v2) {
    HopGroupTable *g1 = (HopGroupTable *)v1;
    const HopGroupTable *g2 = (const HopGroupTable *)v2;
    if (g2->dEnergy > g1->dEnergy) g1->dEnergy = g2->dEnergy;
}

int pkdHopUnbind(PKD pkd, double dTime, int nMinGroupSize, int bPeriodic, blitz::TinyVector<double, 3> dPeriod) {
    MDL mdl = pkd->mdl;
    EE *ee = (EE *)(pkd->pLite);
    int gid, iRoot, i, j;
    int nEvaporated;
    double a = csmTime2Exp(pkd->csm, dTime);
    double ia = 1.0 / a;
    double a2 = a * a;
    double dv, dv2;
    double dEnergy;

    mdlCOcache(mdl, CID_GROUP, NULL, pkd->hopGroups, sizeof(HopGroupTable), pkd->nGroups,
               pkd, initMaxHopEnergy, combMaxHopEnergy);
    for (gid = 1; gid < pkd->nGroups; ++gid) {
        if (pkd->hopGroups[gid].bComplete) continue;
        iRoot = pkd->hopGroups[gid].iTreeRoot;
        auto pNode = pkd->tree[iRoot];
        //n = pNode->pUpper - pNode->pLower + 1;
        /* Calculate kinetic energy & total energy */
        for (auto &P : *pNode) {
            assert(P.group()==gid);
            auto v = P.velocity();
            dv2 = 0.0;
            for (j = 0; j < 3; ++j) {
                dv = v[j] - pkd->hopGroups[gid].vcom[j];
                dv2 += dv * dv;
            }
            ee[i].i = i;
            ee[i].dPot = P.potential() * ia;
            if (P.is_gas() && pkd->particles.present(PKD_FIELD::oSph)) {  /* TODO: is this correct? */
                const auto &sph = P.sph();
                ee[i].dPot += sph.Uint / P.mass();
            }
            ee[i].dSqrtKin = a * sqrt(0.5 * dv2);
            ee[i].dTot = 0.5 * a2 * dv2 + ee[i].dPot;
        }
        qsort(ee + pNode->lower(), pNode->count(), sizeof(EE), cmpEE);
        i = pNode->count()>100 ? pNode->upper()-100 : pNode->lower();
        while ( i <= pNode->upper() && ee[i].dTot <= 0.0 ) ++i;
        if (i > pNode->upper()) dEnergy = 0.0;
        else  dEnergy = ee[i].dTot;
        pkd->hopGroups[gid].dEnergy = dEnergy;

        if (pkd->hopGroups[gid].id.iPid != pkd->Self()) {
            auto g = static_cast<HopGroupTable *>(mdlAcquire(mdl, CID_GROUP, pkd->hopGroups[gid].id.iIndex, pkd->hopGroups[gid].id.iPid));
            g->dEnergy = dEnergy;
            mdlRelease(mdl, CID_GROUP, g);
        }
    }
    mdlFinishCache(mdl, CID_GROUP);

    mdlROcache(mdl, CID_GROUP, NULL, pkd->hopGroups, sizeof(HopGroupTable), pkd->nGroups);
    for (gid = 1 + pkd->nLocalGroups; gid < pkd->nGroups; ++gid) {
        if (pkd->hopGroups[gid].bComplete) continue;
        assert(pkd->hopGroups[gid].id.iPid != pkd->Self());
        auto g = static_cast<HopGroupTable *>(mdlAcquire(mdl, CID_GROUP, pkd->hopGroups[gid].id.iIndex, pkd->hopGroups[gid].id.iPid));
        assert(pkd->hopGroups[gid].id.iIndex == g->id.iIndex && pkd->hopGroups[gid].id.iPid == g->id.iPid);
        pkd->hopGroups[gid].dEnergy = g->dEnergy;
        mdlRelease(mdl, CID_GROUP, g);
    }
    mdlFinishCache(mdl, CID_GROUP);
    nEvaporated = 0;
    for (gid = 1; gid < pkd->nGroups; ++gid) {
        if (pkd->hopGroups[gid].bComplete) continue;
        iRoot = pkd->hopGroups[gid].iTreeRoot;
        dEnergy = pkd->hopGroups[gid].dEnergy;
        auto pNode = pkd->tree[iRoot];
        // n = pNode->count();
        for ( i = pNode->upper(); i >= pNode->lower() && ee[i].dTot >= dEnergy; --i) {
            auto P = pkd->particles[ee[i].i];
            P.set_group(0);
            ++nEvaporated;
        }
        if (i == pNode->upper()) pkd->hopGroups[gid].bComplete = 1;
        /* Move evaporated particles to the end */
        else for (i = pNode->lower(); i <= pNode->upper(); ) {
                auto P = pkd->particles[i];
                if (P.group()) ++i;
                else {
                    auto Q = pkd->particles[pNode->upper()--];
                    swap(P, Q);
                }
            }
    }

    assert(0);  /* the purge small groups below must work on ga... */
    pkd->nGroups = pkdPurgeSmallGroups(pkd, pkd->nGroups, pkd->ga, nMinGroupSize);

    return nEvaporated;
}


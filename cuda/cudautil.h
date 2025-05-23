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

#ifndef CUDAUTIL_H
#define CUDAUTIL_H
#ifdef USE_CUDA
#include "basetype.h"

#define CUDA_STREAMS 8
#define CUDA_WP_MAX_BUFFERED 128

#ifdef __cplusplus
#include "mdlcuda.h"
#include <vector>
#include <cstdlib>
#include "check.h"
#include "gpu/pppcdata.h"
#include "gpu/dendata.h"
#include "gpu/dencorrdata.h"
#include "gpu/sphforcedata.h"
#include "cudapppc.h"
#include "SPH/SPHOptions.h"
#include <queue>

class MessageEwald : public mdl::cudaMessage, public gpu::hostData {
protected:
    class CudaClient &cuda;
    std::vector<workParticle *> ppWP; // [CUDA_WP_MAX_BUFFERED]
    virtual void launch(mdl::Stream &stream, void *pCudaBufIn, void *pCudaBufOut) override;
    virtual void finish() override;
    int nParticles, nMaxParticles;
public:
    explicit MessageEwald(class CudaClient &cuda);
    bool queue(workParticle *work); // Add work to this message; return false if this is not possible
};

class MessageEwaldSetup : public mdl::cudaMessage {
protected:
    virtual void launch(mdl::Stream &stream, void *pCudaBufIn, void *pCudaBufOut) override;
public:
    explicit MessageEwaldSetup(struct EwaldVariables *const ew, EwaldTable *const ewt, int iDevice=-1);
protected:
    struct EwaldVariables *const ewIn;
    EwaldTable *const ewt;
    std::vector<momFloat> dLx, dLy, dLz;
    std::vector<int> ibHole;
};

class MessageSPHOptionsSetup : public mdl::cudaMessage {
protected:
    virtual void launch(mdl::Stream &stream, void *pCudaBufIn, void *pCudaBufOut) override;
public:
    explicit MessageSPHOptionsSetup(SPHOptionsGPU *const SPHoptions, int iDevice=-1);
protected:
    SPHOptionsGPU *const SPHoptionsIn;
};

class CudaClient {
    friend class MessageEwald;
protected:
    mdl::messageQueue<MessageEwald> freeEwald;
    MessageEwald *ewald;
    mdl::messageQueue<MessagePP> freePP;
    MessagePP *pp;
    mdl::messageQueue<MessagePC> freePC;
    MessagePC *pc;
    mdl::messageQueue<MessageDen<32>> freeDen;
    MessageDen<32> *den;
    mdl::messageQueue<MessageDenCorr<32>> freeDenCorr;
    MessageDenCorr<32> *denCorr;
    mdl::messageQueue<MessageSPHForce<32>> freeSPHForce;
    MessageSPHForce<32> *sphForce;

    int nEwhLoop;
    std::list<MessageEwald> free_Ewald, busy_Ewald;
    mdl::CUDA &cuda;
    mdl::gpu::Client &gpu;
protected:
    template<class MESSAGE, class QUEUE, class TILE>
    int queue(MESSAGE *&M, QUEUE &Q, workParticle *work, TILE &tile, bool bGravStep) {
        if (M) { // If we are in the middle of building data for a kernel
            if (M->queue(work, tile, bGravStep)) return work->nP; // Successfully queued
            flush(M); // The buffer is full, so send it
        }
        gpu.flushCompleted();
        if (Q.empty()) return 0; // No buffers so the CPU has to do this part
        M = & Q.dequeue();
        if (M->queue(work, tile, bGravStep)) return work->nP; // Successfully queued
        return 0; // Not sure how this would happen, but okay.
    }
    template<class MESSAGE>
    void flush(MESSAGE *&M) {
        if (M) {
            M->prepare();
            cuda.enqueue(*M, gpu);
            M = nullptr;
        }
    }
public:
    std::queue<workParticle *> wps;
    explicit CudaClient( mdl::CUDA &cuda, mdl::gpu::Client &gpu);
    void flushCUDA();
    int queuePP(workParticle *work, ilpTile &tile, bool bGravStep) {
        return queue(pp, freePP, work, tile, bGravStep);
    }

    int queuePC(workParticle *work, ilcTile &tile, bool bGravStep) {
        return queue(pc, freePC, work, tile, bGravStep);
    }

    int queueDen(workParticle *work, ilpTile &tile) {
        return queue(den, freeDen, work, tile, false);
    }

    int queueDenCorr(workParticle *work, ilpTile &tile) {
        return queue(denCorr, freeDenCorr, work, tile, false);
    }

    int queueSPHForce(workParticle *work, ilpTile &tile) {
        return queue(sphForce, freeSPHForce, work, tile, false);
    }
    int  queueEwald(workParticle *wp);
    void setupEwald(struct EwaldVariables *const ew, EwaldTable *const ewt);
    void setupSPHOptions(SPHOptionsGPU *const SPHoptions);
};
#endif
#endif
#endif

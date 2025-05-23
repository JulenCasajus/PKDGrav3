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
#include "output.h"
#include "pst.h"
#include <complex>
#define COMPLEX std::complex<float>

/* Generic context: all things/stuff from iIndex (starts at zero) */
struct packCtx {
    PKD pkd;
    size_t iIndex;
    int iGrid;
};

static int unpackWrite(void *vctx, int *id, size_t nSize, void *vBuff) {
    auto info = reinterpret_cast<asyncFileInfo *>(vctx);
    io_write(info, vBuff, nSize);
    return 1;
}

/*
** Tiny Group statistics
*/
static int packGroupStats(void *vctx, int *id, size_t nSize, void *vBuff) {
    struct packCtx *ctx = (struct packCtx *)vctx;
    int nLeft = ctx->pkd->nLocalGroups - ctx->iIndex;
    int n = nSize / sizeof(TinyGroupTable);
    if (n > nLeft) n = nLeft;
    memcpy(vBuff, ctx->pkd->tinyGroupTable + 1 + ctx->iIndex, n * sizeof(TinyGroupTable) );
    ctx->iIndex += n;
    return n * sizeof(TinyGroupTable);
}

static int packGridK(void *vctx, int *id, size_t nSize, void *vBuff) {
    struct packCtx *ctx = (struct packCtx *)vctx;
    PKD pkd = ctx->pkd;
    if (mdlCore(pkd->mdl) == 0) {
        auto fftData = reinterpret_cast<COMPLEX *>(pkd->pLite) + ctx->iGrid * pkd->fft->kgrid->nLocal;
        size_t nLocal = 1ul * pkd->fft->kgrid->a1 * pkd->fft->kgrid->n2 * pkd->fft->kgrid->nSlab;
        size_t nLeft = nLocal - ctx->iIndex;
        size_t n = nSize / sizeof(*fftData);
        if (n > nLeft) n = nLeft;
        memcpy(vBuff, fftData + ctx->iIndex, n * sizeof(*fftData) );
        ctx->iIndex += n;
        return n * sizeof(*fftData);
    }
    else return 0;
}

static int packGridR(void *vctx, int *id, size_t nSize, void *vBuff) {
    struct packCtx *ctx = (struct packCtx *)vctx;
    PKD pkd = ctx->pkd;
    if (mdlCore(pkd->mdl) == 0) {
        auto fftData = reinterpret_cast<float *>(pkd->pLite) + ctx->iGrid * pkd->fft->rgrid->nLocal;
        auto pOutput = reinterpret_cast<float *>(vBuff);
        size_t nLocal = 1ul * pkd->fft->rgrid->a1 * pkd->fft->rgrid->n2 * pkd->fft->rgrid->nSlab;
        size_t n = nSize / sizeof(*fftData);
        size_t iOutput = 0;
        while (ctx->iIndex < nLocal && iOutput + pkd->fft->rgrid->n1 <= n ) {
            memcpy(pOutput + iOutput, fftData + ctx->iIndex, pkd->fft->rgrid->n1 * sizeof(*fftData) );
            iOutput += pkd->fft->rgrid->n1;
            ctx->iIndex += pkd->fft->rgrid->a1;
        }
        return iOutput * sizeof(*fftData);
    }
    else return 0;
}

/*
** We do not do the write, rather we send to another thread.
*/

void pkdOutputSend(PKD pkd, outType eOutputType, int iPartner, int iGrid) {
    struct packCtx ctx;
    mdlPack pack;
    ctx.pkd = pkd;
    ctx.iIndex = 0;
    ctx.iGrid = iGrid;
    switch (eOutputType) {
    case OUT_TINY_GROUP:
        pack = packGroupStats;
        break;
    case OUT_KGRID:
        pack = packGridK;
        break;
    case OUT_RGRID:
        pack = packGridR;
        break;
    default:
        fprintf(stderr, "ERROR: invalid output type %d\n", eOutputType);
        abort();
    }
    mdlSend(pkd->mdl, iPartner, pack, &ctx);
}

int pstOutputSend(PST pst, void *vin, int nIn, void *vout, int nOut) {
    auto in = reinterpret_cast<struct inOutputSend *>(vin);
    pkdOutputSend(pst->plcl->pkd, in->eOutputType, in->iPartner, in->iGrid);
    return 0;
}

/*
** We are the writer. We may need to receive as well.
*/

static void localWrite(PKD pkd, mdlPack unpack, void *info, mdlPack pack, int iGrid) {
    const size_t SEND_BUFFER_SIZE = 1 * 1024 * 1024;
    packCtx ctx;
    ctx.pkd = pkd;
    ctx.iIndex = 0;
    ctx.iGrid = iGrid;
    int id = 0;

    auto vOut = new char[SEND_BUFFER_SIZE];
    while ( auto n = (*pack)(&ctx, &id, SEND_BUFFER_SIZE, vOut) ) {
        (*unpack)(info, &id, n, vOut);
    }
    delete[] vOut;
}

void pkdOutput(PKD pkd, outType eOutputType, int iProcessor, int nProcessor,
               int iPartner, int nPartner, const char *fname, int iGrid ) {
    mdlPack unpack;
    asyncFileInfo info;
    char achOutFile[256];
    strcpy(achOutFile, fname);
    int n = strlen(achOutFile);
    snprintf(achOutFile + n, sizeof(achOutFile)-n, ".%d", iProcessor);
    io_init(&info, 4, 1024 * 1024, IO_AIO | IO_LIBAIO);
    if (io_create(&info, achOutFile) < 0) { perror(fname); abort(); }

    switch (eOutputType) {
    case OUT_TINY_GROUP:
        io_write(&info, pkd->tinyGroupTable + 1, sizeof(TinyGroupTable)*pkd->nLocalGroups);
        unpack = unpackWrite;
        break;
    case OUT_KGRID:
        localWrite(pkd, unpackWrite, &info, packGridK, iGrid);
        unpack = unpackWrite;
        break;
    case OUT_RGRID:
        localWrite(pkd, unpackWrite, &info, packGridR, iGrid);
        unpack = unpackWrite;
        break;
    default:
        unpack = NULL;
        fprintf(stderr, "ERROR: invalid output type %d\n", eOutputType);
        abort();
    }
    while (--nPartner) {
        struct inOutputSend send;
        send.iPartner = pkd->Self();
        send.eOutputType = eOutputType;
        send.iGrid = iGrid;
        ++iPartner;
        int rID = mdlReqService(pkd->mdl, iPartner, PST_OUTPUT_SEND, &send, sizeof(send));
        mdlRecv(pkd->mdl, iPartner, unpack, &info);
        mdlGetReply(pkd->mdl, rID, NULL, NULL);
    }
    io_close(&info);
    io_free(&info);
}

int pstOutput(PST pst, void *vin, int nIn, void *vout, int nOut) {
    auto in = reinterpret_cast<struct inOutput *>(vin);

    mdlassert(pst->mdl, nIn >= sizeof(struct inOutput));
    if (pstNotCore(pst)) {
        int nProcessor = in->nProcessor;
        int iProcessor = in->iProcessor;

        /* Still allowed to write more in parallel */
        if (nProcessor > 1) {
            int nLower, nUpper;
            nLower = nProcessor * pst->nLower / pst->nLeaves;
            if (nLower == 0) nLower = 1;
            nUpper = nProcessor - nLower;
            in->nProcessor = nUpper;
            in->iProcessor = iProcessor + nLower;
            int rID = mdlReqService(pst->mdl, pst->idUpper, PST_OUTPUT, in, nIn);
            in->nProcessor = nLower;
            in->iProcessor = iProcessor;
            pstOutput(pst->pstLower, in, nIn, NULL, 0);
            mdlGetReply(pst->mdl, rID, NULL, NULL);
        }
        /* We are the node that will be the writer for all of the pst children */
        else if (nProcessor == 1) {
            in->iPartner = pst->idSelf;
            in->nPartner = pst->nLeaves;
            in->nProcessor = 0;
            pstOutput(pst->pstLower, in, nIn, NULL, 0); /* Keep decending to write */
        }
        else {
            pstOutput(pst->pstLower, in, nIn, NULL, 0); /* Keep decending to write */
        }
    }
    else {
        /* If it is fully parallel then there is just us writing. */
        if (in->nProcessor > 0) {
            in->iPartner = pst->idSelf;
            in->nPartner = 1;
        }
        PKD pkd = pst->plcl->pkd;
        pkdOutput(pkd, in->eOutputType, in->iProcessor, in->nProcessor,
                  in->iPartner, in->nPartner, in->achOutFile, in->iGrid);
    }
    return 0;
}

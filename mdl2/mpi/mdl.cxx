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

#include "mdl.h"
using namespace mdl;

#include <algorithm>
#include <numeric>
#include <cstring>
#include <set>
#include <queue>

static inline int size_t_to_int(size_t v) {
    return (int)v;
}

#ifdef USE_HWLOC
    #include "hwloc.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#ifdef HAVE_MALLOC_H
    #include <malloc.h>
#endif
#ifdef HAVE_SIGNAL_H
    #include <signal.h>
#endif
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <stdarg.h>
#ifdef HAVE_SYS_TIME_H
    #include <sys/time.h>
#endif
#ifdef HAVE_UNISTD_H
    #include <unistd.h>
#endif
#ifdef __linux__
    #include <sys/resource.h>
    #include <sys/mman.h>
#endif
#ifdef HAVE_NUMA
    #include <numa.h>
#endif
#include "mpi.h"

template<typename T>
MPI_Datatype mpi_type(T v) {
    static_assert(sizeof(T)==0, "Create a specialization for this MPI datatype");
    return MPI_DATATYPE_NULL;
}
template<> MPI_Datatype mpi_type(int32_t v) { return MPI_INT32_T; }
template<> MPI_Datatype mpi_type(uint32_t v) { return MPI_UINT32_T; }
template<> MPI_Datatype mpi_type(int64_t v) { return MPI_INT64_T; }
template<> MPI_Datatype mpi_type(uint64_t v) { return MPI_UINT64_T; }


#ifdef USE_ITT
    #include "ittnotify.h"
#endif

#ifdef __cplusplus
    #define CAST(T, V) static_cast<T>(V)
#else
    #define CAST(T, V) ((T)(V))
#endif

#define MDL_DEFAULT_CACHEIDS    32

#define MDL_TRANS_SIZE      5000000

/*
** The MPI specification allows tags to at least 32767 (but most implementations go higher).
** We multiply this offset by the destination core, so 32768 / 8 = 4096 cores max per process,
** or more if the MPI implementation allows it.
*/
#define MDL_TAG_THREAD_OFFSET   MDL_TAG_MAX

/*
 ** This structure should be "maximally" aligned, with 4 ints it
 ** should align up to at least QUAD word, which should be enough.
 */
typedef struct srvHeader {
    int32_t idFrom;      /* Global thread ID */
    int16_t replyTag;    /* Which MPI tag to send the reply */
    int16_t sid;
    int32_t nInBytes;
    int32_t nOutBytes;
} SRVHEAD;



/*****************************************************************************\
* State retrieval
\*****************************************************************************/
pthread_key_t mdl_key; // One MDL per thread
pthread_key_t worker_key; // One worker context per thread

MDL   mdlMDL(void)    { return pthread_getspecific(mdl_key); }
void *mdlWORKER(void) { return pthread_getspecific(worker_key); }

/*****************************************************************************\
* Class / Thread overview
*   mpiClass - A single instance that handles MPI communication.
*              Derived from mdlClass, and may also be a worker in non-deticated
*   mdlClass - A worker thread instance
\*****************************************************************************/

class legacyCACHEhelper : public CACHEhelper {
protected:
    void *ctx;
    void (*init_function)(void *, void *);
    void (*combine_function)(void *, void *, const void *);
    virtual void    init(void *dst)                                  override {(*init_function)(ctx, dst);}
    virtual void combine(void *dst, const void *src, const void *key) override {(*combine_function)(ctx, dst, src); }
public:
    explicit legacyCACHEhelper( uint32_t nData, void *ctx,
                                void (*init_function)(void *, void *),
                                void (*combine_function)(void *, void *, const void *))
        : CACHEhelper(nData, true), ctx(ctx), init_function(init_function), combine_function(combine_function) {}
};

/*****************************************************************************\
* CACHE open / close
\*****************************************************************************/

// A list of cache tables is constructed when mdlClass is created. They are all set to NOCACHE.
CACHE::CACHE(mdlClass *mdl, uint16_t iCID)
    : mdl(mdl), CacheRequest(iCID, mdl->Self()), iCID(iCID) {
}

// This opens a read - only cache. Called from a worker outside of MDL
extern "C"
void mdlROcache(MDL mdl, int cid,
                void *(*getElt)(void *pData, int i, int iDataSize),
                void *pData, int iDataSize, int nData) {
    static_cast<mdlClass *>(mdl)->CacheInitialize(cid, getElt, pData, nData,
            std::make_shared < CACHEhelper>(iDataSize));
}

// This opens a combiner (read / write) cache. Called from a worker outside of MDL
extern "C"
void mdlCOcache(MDL mdl, int cid,
                void *(*getElt)(void *pData, int i, int iDataSize),
                void *pData, int iDataSize, int nData,
                void *ctx, void (*init)(void *, void *), void (*combine)(void *, void *, const void *)) {
    assert(init);
    assert(combine);
    static_cast<mdlClass *>(mdl)->CacheInitialize(cid, getElt, pData, nData,
            std::make_shared < legacyCACHEhelper>(iDataSize, ctx, init, combine));
}

CACHE *mdlClass::CacheInitialize(int cid,
                                 void *(*getElt)(void *pData, int i, int iDataSize),
                                 void *pData, int nData, int iDataSize) {
    return CacheInitialize(cid, getElt, pData, nData, std::make_shared < CACHEhelper>(iDataSize));
}

// Cache creation is a collective operation (all worker threads participate): call the initialize() member to set it up.
CACHE *mdlClass::CacheInitialize(
    int cid,
    void *(*getElt)(void *pData, int i, int iDataSize),
    void *pData, int nData,
    std::shared_ptr < CACHEhelper> helper) {

    // We cannot reallocate this structure because there may be other threads accessing it.
    // This might be safe to do with an appropriate barrier, but it would shuffle CACHE objects.
    // The problem isn't the cache we are about to open, rather other caches that are already open.
    assert(cid >= 0 && cid <cache.size());
    if (cid < 0 || cid >= cache.size()) abort();

    auto c = cache[cid].get();
    c->initialize(cacheSize, getElt, pData, nData, helper);

    /* Nobody should start using this cache until all threads have started it! */
    ThreadBarrier(true);

    /* We might need to resize the cache buffer */
    enqueueAndWait(mdlMessageCacheOpen());

    return (c);
}

class packedCACHEhelperRO : public CACHEhelper {
protected:
    uint32_t nPack;
    void *ctx;
    void (*pack_function)   (void *, void *, const void *);
    void (*unpack_function) (void *, void *, const void *);

    virtual void   pack(void *dst, const void *src)                  override { (*pack_function)(ctx, dst, src); }
    virtual void unpack(void *dst, const void *src, const void *key) override { (*unpack_function)(ctx, dst, src); }
    virtual uint32_t pack_size() override {return nPack;}

public:
    explicit packedCACHEhelperRO(uint32_t nData, void *ctx,
                                 uint32_t pack_size,
                                 void (*pack_function)   (void *, void *, const void *),
                                 void (*unpack_function) (void *, void *, const void *))
        : CACHEhelper(nData, false), nPack(pack_size), ctx(ctx),
          pack_function(pack_function), unpack_function(unpack_function) {}
};

void mdlPackedCacheRO(MDL mdl, int cid,
                      void *(*getElt)(void *pData, int i, int iDataSize),
                      void *pData, int nData, uint32_t iDataSize,
                      void *ctx, uint32_t iPackSize,
                      void (*pack)   (void *, void *, const void *),
                      void (*unpack) (void *, void *, const void *)) {
    static_cast<mdlClass *>(mdl)->CacheInitialize(cid, getElt, pData, nData,
            std::make_shared < packedCACHEhelperRO>(iDataSize, ctx, iPackSize,
                    pack, unpack));
}

class packedCACHEhelperCO : public CACHEhelper {
protected:
    uint32_t nPack, nFlush;
    void *ctx;
    void (*pack_function)   (void *, void *, const void *);
    void (*unpack_function) (void *, void *, const void *);
    void (*init_function)   (void *, void *);
    void (*flush_function)  (void *, void *, const void *);
    void (*combine_function)(void *, void *, const void *);

    virtual void    pack(void *dst, const void *src)                  override { (*pack_function)(ctx, dst, src); }
    virtual void  unpack(void *dst, const void *src, const void *key) override { (*unpack_function)(ctx, dst, src); }
    virtual void    init(void *dst)                                   override { (*init_function)(ctx, dst); }
    virtual void   flush(void *dst, const void *src)                  override { (*flush_function)(ctx, dst, src); }
    virtual void combine(void *dst, const void *src, const void *key) override { (*combine_function)(ctx, dst, src); }
    virtual uint32_t pack_size()  override {return nPack;}
    virtual uint32_t flush_size() override {return nFlush;}

public:
    explicit packedCACHEhelperCO(uint32_t nData, void *ctx,
                                 uint32_t pack_size,
                                 void (*pack_function)   (void *, void *, const void *),
                                 void (*unpack_function) (void *, void *, const void *),
                                 void (*init_function)   (void *, void *),
                                 uint32_t flush_size,
                                 void (*flush_function)  (void *, void *, const void *),
                                 void (*combine_function)(void *, void *, const void *))
        : CACHEhelper(nData, true), nPack(pack_size), nFlush(flush_size), ctx(ctx),
          pack_function(pack_function), unpack_function(unpack_function),
          init_function(init_function), flush_function(flush_function),
          combine_function(combine_function) {}
};

void mdlPackedCacheCO(MDL mdl, int cid,
                      void *(*getElt)(void *pData, int i, int iDataSize),
                      void *pData, int nData, uint32_t iDataSize,
                      void *ctx, uint32_t iPackSize,
                      void (*pack)   (void *, void *, const void *),
                      void (*unpack) (void *, void *, const void *),
                      uint32_t iFlushSize,
                      void (*init)   (void *, void *),
                      void (*flush)  (void *, void *, const void *),
                      void (*combine)(void *, void *, const void *)) {
    static_cast<mdlClass *>(mdl)->CacheInitialize(cid, getElt, pData, nData,
            std::make_shared < packedCACHEhelperCO>(iDataSize, ctx, iPackSize,
                    pack, unpack, init, iFlushSize, flush, combine));
}

class advancedCACHEhelper : public CACHEhelper {
protected:
    uint32_t nPack, nFlush;
    void *ctx;
    void (*pack_function)   (void *, void *, const void *);
    void (*unpack_function) (void *, void *, const void *);
    void (*init_function)   (void *, void *);
    void (*flush_function)  (void *, void *, const void *);
    void (*combine_function)(void *, void *, const void *);
    uint32_t (*get_thread)  (void *, uint32_t, uint32_t, const void *);
    void *(*create_function) (void *, uint32_t, const void *);

    virtual void    pack(void *dst, const void *src)                  override { (*pack_function)(ctx, dst, src); }
    virtual void  unpack(void *dst, const void *src, const void *key) override { (*unpack_function)(ctx, dst, src); }
    virtual void    init(void *dst)                                   override { if (init_function)   (*init_function)(ctx, dst); }
    virtual void   flush(void *dst, const void *src)                  override { if (flush_function)  (*flush_function)(ctx, dst, src); }
    virtual void combine(void *dst, const void *src, const void *key) override { if (combine_function)(*combine_function)(ctx, dst, src); }
    virtual void *create(uint32_t size, const void *pKey) override
    { return (*create_function)(ctx, size, pKey); }
    virtual uint32_t getThread(uint32_t uLine, uint32_t uId, uint32_t size, const void *pKey) override
    { return (*get_thread)(ctx, uLine, size, pKey); }
    virtual uint32_t pack_size()  override {return nPack;}
    virtual uint32_t flush_size() override {return nFlush;}
public:
    explicit advancedCACHEhelper(uint32_t nData, bool bModify, void *ctx,
                                 uint32_t (*get_thread)  (void *, uint32_t, uint32_t, const void *),
                                 uint32_t pack_size,
                                 void (*pack_function)   (void *, void *, const void *),
                                 void (*unpack_function) (void *, void *, const void *),
                                 void (*init_function)   (void *, void *),
                                 uint32_t flush_size,
                                 void (*flush_function)  (void *, void *, const void *),
                                 void (*combine_function)(void *, void *, const void *),
                                 void *(*create_function) (void *, uint32_t, const void *))
        : CACHEhelper(nData, bModify), nPack(pack_size), nFlush(flush_size), ctx(ctx),
          pack_function(pack_function), unpack_function(unpack_function),
          init_function(init_function), flush_function(flush_function),
          combine_function(combine_function),
          get_thread(get_thread), create_function(create_function) {}
};

extern "C"
void mdlAdvancedCache(MDL mdl, int cid, void *pHash, int iDataSize, bool bModify, void *ctx,
                      uint32_t (*get_thread)  (void *, uint32_t, uint32_t, const void *),
                      uint32_t pack_size,
                      void (*pack_function)   (void *, void *, const void *),
                      void (*unpack_function) (void *, void *, const void *),
                      void (*init_function)   (void *, void *),
                      uint32_t flush_size,
                      void (*flush_function)  (void *, void *, const void *),
                      void (*combine_function)(void *, void *, const void *),
                      void *(*create_function) (void *, uint32_t, const void *)) {
    auto hash = static_cast<hash::GHASH *>(pHash);
    static_cast<mdlClass *>(mdl)->AdvancedCacheInitialize(cid, hash, iDataSize,
            std::make_shared < advancedCACHEhelper>(iDataSize, bModify, ctx, get_thread,
                    pack_size, pack_function, unpack_function, init_function,
                    flush_size, flush_function, combine_function, create_function));
}

extern "C"
void mdlAdvancedCacheRO(MDL mdl, int cid, void *pHash, int iDataSize) {
    auto hash = static_cast<hash::GHASH *>(pHash);
    static_cast<mdlClass *>(mdl)->AdvancedCacheInitialize(cid, hash, iDataSize,
            std::make_shared < CACHEhelper>(iDataSize));
}

extern "C"
void mdlAdvancedCacheCO(MDL mdl, int cid, void *pHash, int iDataSize,
                        void *ctx, void (*init)(void *, void *), void (*combine)(void *, void *, const void *)) {
    auto hash = static_cast<hash::GHASH *>(pHash);
    static_cast<mdlClass *>(mdl)->AdvancedCacheInitialize(cid, hash, iDataSize,
            std::make_shared < legacyCACHEhelper>(iDataSize, ctx, init, combine));
}

CACHE *mdlClass::AdvancedCacheInitialize(int cid, hash::GHASH *hash, int iDataSize, std::shared_ptr < CACHEhelper> helper) {
    assert(cid >= 0 && cid <cache.size());
    if (cid < 0 || cid >= cache.size()) abort();
    auto c = cache[cid].get();
    c->initialize_advanced(cacheSize, hash, iDataSize, helper);

    /* Nobody should start using this cache until all threads have started it! */
    ThreadBarrier(true);

    /* We might need to resize the cache buffer */
    enqueueAndWait(mdlMessageCacheOpen());

    return c;
}

// This function save the message on the SendReceiveMessages list,
// and creates a MPI_Request on the SendReceiveRequests list.
// This is returned and used by an MPI route (e.g., MPI_Isend).
// When complete message->finish() will be called (see finishRequests).
MPI_Request *mpiClass::newRequest(mdlMessageMPI *message, MPI_Request request) {
#ifndef NDEBUG
    ++nRequestsCreated;
#endif
    if (SendReceiveAvailable.size()) {
        auto i = SendReceiveAvailable.back();
        SendReceiveAvailable.pop_back();
        assert(SendReceiveMessages[i]==nullptr);
        assert(SendReceiveRequests[i]==MPI_REQUEST_NULL);
        assert(i < SendReceiveMessages.size());
        assert(i < SendReceiveRequests.size());
        SendReceiveMessages[i] = message;
        SendReceiveRequests[i] = request;
        return &SendReceiveRequests[i];
    }
    else {
        SendReceiveMessages.push_back(message);
        SendReceiveRequests.push_back(request);
        return &SendReceiveRequests.back();
    }
}

// Open the cache by posting the receive if required
void mpiClass::MessageCacheOpen(mdlMessageCacheOpen *message) {
    if (nOpenCaches == 0) {
        msgCacheReceive->action(this);
        countCacheInflight.resize(Procs(), 0);
        PendingRequests.set_capacity(Threads()); // This is very conservative, but uses some memory
#ifdef DEBUG_COUNT_CACHE
        countCacheSend.resize(Procs());
        countCacheRecv.resize(Procs());
        std::fill(countCacheSend.begin(), countCacheSend.end(), 0);
        std::fill(countCacheRecv.begin(), countCacheRecv.end(), 0);
#endif
    }
    ++nOpenCaches;
    message->sendBack();
}

// Post the recieve. Receive will continue to be reposted as message are recieved.
void mpiClass::MessageCacheReceive(mdlMessageCacheReceive *message) {
    auto request = newRequest(message);
    MPI_Recv_init(message->getBuffer(), iCacheBufSize, MPI_BYTE, MPI_ANY_SOURCE,
                  MDL_TAG_CACHECOM, commMDL, request);
    MPI_Start(request);
}

// Called when the receive finishes. Decode the type of message and process it,
// then restart the receive.
void mpiClass::FinishCacheReceive(mdlMessageCacheReceive *message, MPI_Request request, MPI_Status status) {
    int bytes, source, cancelled;
    assert(message);
    MPI_Test_cancelled(&status, &cancelled);
    if (cancelled) {
        MPI_Request_free(&request);
        assert(request == MPI_REQUEST_NULL);
        return;
    }
    MPI_Get_count(&status, MPI_BYTE, &bytes);
    source = status.MPI_SOURCE;
    assert(source != Proc());

    CacheHeader *ph = reinterpret_cast<CacheHeader *>(message->getBuffer());
#ifdef DEBUG_COUNT_CACHE
    if (ph->sequence != countCacheRecv[source]) {
        printf("MISMATCH: %d -> %d: sequence %llu expected %llu\n", source, Proc(), ph->sequence, countCacheRecv[source]);
    }
    assert(ph->sequence == countCacheRecv[source]);
    ++countCacheRecv[source];
#endif

    /* Well, could be any threads cache */
    int iCore = ph->idTo - Self();
    assert(iCore >= 0 && iCore < Cores());

    CacheReceive(bytes, ph, source);
    MPI_Start(newRequest(message, request));
}

void mpiClass::CacheReceive(int bytes, CacheHeader *ph, int iProcFrom) {
    // This message can consist of a list of any of the three, so process them individually
    while (bytes) {
        int consumed;
        switch (ph->mid) {
        case CacheMessageType::REQUEST: consumed = CacheReceiveRequest(bytes, ph, iProcFrom); break;
        case CacheMessageType::FLUSH:   consumed = CacheReceiveFlush(bytes, ph, iProcFrom);  break;
        case CacheMessageType::REPLY:   consumed = CacheReceiveReply(bytes, ph, iProcFrom);  break;
        default:
            assert(0);
        }
        assert(consumed <= bytes);
        bytes -= consumed;
        ph = reinterpret_cast<CacheHeader *>(reinterpret_cast<char *>(ph)+consumed);
    }
    expedite_flush(iProcFrom);
}

/*
** This is the default element fetch routine.  It impliments the old behaviour
** of a single large array.  New data structures need to be more clever.
*/
void *CACHE::getArrayElement(void *vData, int i, int iDataSize) {
    char *pData = CAST(char *, vData);
    return pData + (size_t)i*(size_t)iDataSize;
}

// This records the callback information, and updates the ARC cache to match (if necessary)
void CACHE::initialize(uint32_t cacheSize,
                       void *(*getElt)(void *pData, int i, int iDataSize),
                       void *pData, int nData,
                       std::shared_ptr < CACHEhelper> helper) {

    assert(!cache_helper);

    this->getElt = getElt == NULL ? getArrayElement : getElt;
    this->pData = pData;
    this->nData = nData;
    this->iDataSize = helper->data_size();
    this->cache_helper = helper;

    if (iDataSize > MDL_CACHE_DATA_SIZE) nLineBits = 0;
    else nLineBits = log2(MDL_CACHE_DATA_SIZE / iDataSize);
    /*
    ** Let's not have too much of a good thing. We want to fetch in cache lines both for
    ** performance, and to save memory (ARC cache has a high overhead), but too much has
    ** a negative effect on performance. Empirically, 16 elements maximal is optimal.
    ** This was tested with the particle, cell and P(k) caches.
    */
    if (nLineBits > 4) nLineBits = 4;
    iLineSize = getLineElementCount()*iDataSize;
    OneLine.resize(iLineSize);

    nAccess = nMiss = 0; // Clear statistics. Are these even used any more?

    auto arc = arc_cache.get();
    if (!arc || typeid(*arc)!=typeid(ARC<>)) arc_cache.reset(new ARC<>());
    arc_cache->initialize(this, cacheSize, iLineSize, nLineBits);
}

void CACHE::initialize_advanced(uint32_t cacheSize, hash::GHASH *hash, int iDataSize, std::shared_ptr < CACHEhelper> helper) {
    hash_table = hash;
    this->iDataSize = this->iLineSize = iDataSize;
    assert(helper);
    this->cache_helper = helper;
    this->pData = nullptr;
    this->nData = 0;

    nLineBits = 0;
    OneLine.resize(iDataSize);
    nAccess = nMiss = 0; // Clear statistics. Are these even used any more?

    // Clone the table if it is not the same type as what we have
    auto arc = hash_table->clone(arc_cache.get());
    if (arc) arc_cache.reset(arc);
    arc_cache->initialize(this, cacheSize, iLineSize, nLineBits);
}

// When we are finished using the cache, it is marked as complete. All elements should have been flushed by now.
void CACHE::close() {
    // Keep the arc cache for performance reasonses: arc_cache.reset();
    cache_helper.reset(); // Shared: we are finished with this
    hash_table = nullptr;
}

/*****************************************************************************\
* Functions to allow writing to a local element when the cache is active
\*****************************************************************************/

extern "C"
void *mdlAcquireWrite(MDL cmdl, int cid, int iIndex) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    return mdl->AcquireWrite(cid, iIndex);
}

void mdlReleaseWrite(MDL cmdl, int cid, void *p) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    mdl->ReleaseWrite(cid, p);
}

/*****************************************************************************\
* Here we follow a Cache Request
\*****************************************************************************/

/* Does not lock the element */
extern "C"
void *mdlFetch(MDL mdl, int cid, int iIndex, int id) {
    const bool lock   = false; // we never lock in fetch
    const bool modify = false; // fetch can never modify
    const bool virt   = false; // really fetch the element
    return static_cast<mdlClass *>(mdl)->Access(cid, iIndex, id, lock, modify, virt);
}

extern "C"
const void *mdlKeyFetch(MDL mdl, int cid, uint32_t uHash, void *pKey, int lock, int modify, int virt) {
    return static_cast<mdlClass *>(mdl)->Access(cid, uHash, pKey, lock, modify, virt);
}

/* Locks, so mdlRelease must be called eventually */
extern "C"
void *mdlAcquire(MDL cmdl, int cid, int iIndex, int id) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    const bool lock   = true;  // we always lock in acquire
    const bool modify = mdl->cache[cid]->modify();
    const bool virt   = false; // really fetch the element
    return mdl->Access(cid, iIndex, id, lock, modify, virt);
}

/* Locks, so mdlRelease must be called eventually */
extern "C"
void *mdlKeyAcquire(MDL cmdl, int cid, uint32_t uHash, void *pKey) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    const bool lock   = true;  // we always lock in acquire
    const bool modify = mdl->cache[cid]->modify();
    const bool virt   = false; // really fetch the element
    return static_cast<mdlClass *>(mdl)->Access(cid, uHash, pKey, lock, modify, virt);
}

/* Locks the element, but does not fetch or initialize */
extern "C"
void *mdlVirtualFetch(MDL mdl, int cid, int iIndex, int id) {
    const int lock   = false; // fetch never locks
    const int modify = true;  // virtual always modifies
    const bool virt  = true;  // do not fetch the element
    return static_cast<mdlClass *>(mdl)->Access(cid, iIndex, id, lock, modify, virt);
}

// main routine to perform an immediate cache request. Does not return until the cache element is present
void *mdlClass::Access(int cid, uint32_t uIndex, int uId, bool bLock, bool bModify, bool bVirtual) {
    auto c = cache[cid].get();

    if (!(++c->nAccess & mdl_check_mask)) CacheCheck(); // For non-dedicated MPI thread we periodically check for messages

    /* Short circuit the cache if this belongs to another thread (or ourselves), and is read - only */
    uint32_t uCore = uId - mpi->Self();
    if (uCore < Cores() && !c->modify() ) {
        mdlClass *omdl = pmdl[uCore];
        auto c = omdl->cache[cid].get();
        return c->getElement(uIndex); // Careful; we don't allow updates here (see ReadLock)
    }
    // Retreive the element from the ARC cache. If it is not present then the ARC class will
    // call invokeRequest to initiate the fetch, then finishRequest to copy the result into the cache
    return c->fetch(uIndex, uId, bLock, bModify, bVirtual); // Otherwise we look it up in the cache, or fetch it remotely
}

void *CACHE::fetch(uint32_t uHash, void *pKey, int bLock, int bModify, bool bVirtual) {
    void *data;
    if (!hash_table || !(data = hash_table->lookup(uHash, pKey)))     // Local data we just return
        data = arc_cache->fetch(uHash, pKey, bLock, bModify, bVirtual); // Otherwise find it remotely
    return data;
}

// main routine to perform an immediate cache request. Does not return until the cache element is present
void *mdlClass::Access(int cid, uint32_t uHash, void *pKey, bool bLock, bool bModify, bool bVirtual) {
    auto c = cache[cid].get();

    if (!(++c->nAccess & mdl_check_mask)) CacheCheck(); // For non-dedicated MPI thread we periodically check for messages

    // Retreive the element from the ARC cache. If it is not present then the ARC class will
    // call invokeRequest to initiate the fetch, then finishRequest to copy the result into the cache
    return c->fetch(uHash, pKey, bLock, bModify, bVirtual); // Otherwise we look it up in the cache, or fetch it remotely
}

uint32_t CACHE::getThread(uint32_t uLine, uint32_t uId, uint32_t size, const void *pKey) {
    return cache_helper->getThread(uLine, uId, size, pKey);
}

void *CACHE::getLocalData(uint32_t uHash, uint32_t uId, uint32_t size, const void *pKey) {
    auto mpi = mdl->mpi;
    void *data = nullptr;
    uint32_t uCore = uId - mpi->Self();
    if (uCore < mdl->Cores()) {
        mdlClass *omdl = mdl->pmdl[uCore];
        auto c = omdl->cache[iCID].get();
        if (c->hash_table) data = c->hash_table->lookup(uHash, pKey);
    }
    return data;
}

// When we are missing a cache element then we ask the MPI thread to send a request to the remote node
void *CACHE::invokeRequest(uint32_t uLine, uint32_t uId, uint32_t key_size, const void *pKey, bool bVirtual) {
    uint32_t uCore = uId - mdl->mpi->Self();
    void *data = nullptr;
    ++nMiss;
    mdl->TimeAddComputing();

    if (bVirtual) data = nullptr;
    else if (uCore < mdl->Cores()) data = getLocalData(uLine, uId, key_size, pKey);
    else { // Only send a request if non-virtual and remote
        mdl->enqueue(CacheRequest.makeCacheRequest(getLineElementCount(), uId, uLine, key_size, pKey, OneLine.data()), mdl->queueCacheReply);
        data = nullptr;
    }
    return data; // non-null means we located it locally
}

// The MPI thread sends this to the remote node. This will not be returned until the reply has been received.
void mpiClass::MessageCacheRequest(mdlMessageCacheRequest *message) {
    int iCoreFrom = message->header.idFrom - Self();
    assert(CacheRequestMessages[iCoreFrom]==nullptr);
    CacheRequestMessages[iCoreFrom] = message;
    assert(message->pLine);
    int iProc = ThreadToProc(message->header.idTo);
    message->setRankTo(iProc);
    assert(iProc != Proc());
    if (iCacheMaxInflight) {
        // Cache requests are added to the cache flush buffer, but are expedited
        flush_element(&message->header, message->key_size);
        expedite_flush(iProc);
    }
    else {
        CacheRequestRendezvous[iCoreFrom] = true; // The REQUEST and RESPONSE need to Rendezvous
        ++countCacheInflight[iProc];
#ifdef DEBUG_COUNT_CACHE
        message->header.sequence = countCacheSend[iProc]++;
#endif
        MPI_Isend(&message->header, sizeof(message->header)+message->key_size, MPI_BYTE, iProc, MDL_TAG_CACHECOM, commMDL, newRequest(message));
    }
}

// Normally, when an MPI request finishes, we send it back to the requesting thread. The process is
// different for cache requests. We do nothing because the result is actually sent back, not the request.
// You would think that the "request" MPI send would complete before the response message is received,
// but this is NOT ALWAYS THE CASE. What we do is to rendezvous with the REPLY message. If this SEND completes
// before the the REPLY message is received (actually before it is processed), then it will clear the
// rendezvous flag. If the flag is clear here then we send back the request (with response data).
void mpiClass::FinishCacheRequest(mdlMessageCacheRequest *message, MPI_Request request, MPI_Status status) {
    int iProc = message->getRankTo();
    assert(iProc != Proc());
    --countCacheInflight[iProc];
    int iCore = message->header.idFrom - Self();
    assert(iCore >= 0 && iCore < Cores());
    if (CacheRequestRendezvous[iCore]) CacheRequestRendezvous[iCore] = false;
    else {
        CacheRequestMessages[iCore]->sendBack();
        CacheRequestMessages[iCore] = NULL;
    }
    expedite_flush(iProc);
}

void mpiClass::BufferCacheResponse(FlushBuffer *flush, BufferedCacheRequest &request) {
    int iCore = request.header.idTo - Self();
    assert(iCore >= 0 && iCore < Cores());
    auto c = pmdl[iCore]->cache[request.header.cid].get();
    assert(c->isActive());
    auto key_size = c->key_size();

    auto pack_size = c->cache_helper->pack_size();
    assert(pack_size <= MDL_CACHE_DATA_SIZE);

    if (key_size) { // ADVANCED KEY
        flush->addBuffer(CacheMessageType::REPLY, request.header.cid, Self(), request.header.idFrom, request.header.iLine, request.data?1:0);
        if (request.data) c->cache_helper->pack(flush->getBuffer(pack_size), request.data);
    }
    else { // SIMPLE KEY
        int s = request.header.iLine << c->nLineBits;
        int n = s + c->getLineElementCount();
        flush->addBuffer(CacheMessageType::REPLY, request.header.cid, Self(), request.header.idFrom, request.header.iLine);
        for (auto i = s; i < n; i++ ) {
            auto buffer = flush->getBuffer(pack_size);
            if (i < c->nData) {
                auto p = c->ReadLock(i);
                c->cache_helper->pack(buffer, p);
                c->ReadUnlock(p);
            }
            else memset(buffer, 0, pack_size);
        }
    }

}

// On the remote node, the message is received, and a response is constructed with the data
// pulled directly from thread's cache. This is read - only, so is safe. It is forbidden in MDL
// to remotely "read" part of an element that is also being modified. Results are "unpredictable".
// Instead, one is expected to initialize such data via the "init" function.
int mpiClass::CacheReceiveRequest(int count, CacheHeader *ph, int iProcFrom) {
    assert( count >= sizeof(CacheHeader) );
    assert(ph->mid == CacheMessageType::REQUEST);
    int iCore = ph->idTo - Self();
    assert(iCore >= 0 && iCore < Cores());
    auto c = pmdl[iCore]->cache[ph->cid].get();
    assert(c->isActive());
    auto key_size = c->key_size();
    auto pack_size = c->cache_helper->pack_size();
    assert(pack_size <= MDL_CACHE_DATA_SIZE);

    // Find the data in the advanced cache if appropriate
    const void *data;
    if (key_size) { // ADVANCED KEY
        assert(c->hash_table);
        assert(c->getLineElementCount()==1);
        data = c->hash_table->lookup(ph->iLine, ph + 1);
    }
    else data = nullptr;
    BufferedCacheRequest request(ph, data);
    // Figure out which flush buffer to use
    mdlMessageCacheReply *reply;
    FlushBuffer *flush;
    int iSize = sizeof(CacheHeader) + pack_size * c->getLineElementCount(); // Maximum needed for response
    if (iCacheMaxInflight == 0) {
        if (freeCacheReplies.empty()) reply = new mdlMessageCacheReply(iReplyBufSize);
        else { reply = freeCacheReplies.front(); freeCacheReplies.pop_front(); }
        reply->emptyBuffer();
        reply->setRankTo(iProcFrom);
        flush = reply;
    }
    else {
        reply = nullptr;
        flush = get_flush_buffer(iProcFrom, iSize, false);
        if (flush == nullptr) {
            PendingRequests.push_back(request);
            return sizeof(CacheHeader) + key_size;
        }
    }
    BufferCacheResponse(flush, request);
    if (reply) reply->action(this); // MessageCacheReply()
    return sizeof(CacheHeader) + key_size;
}

// The reply message is sent back to the origin node, and added to the MPI request tracker.
// This is the "action" routine.
void mpiClass::MessageCacheReply(mdlMessageCacheReply *pFlush) {
    auto iProc = pFlush->getRankTo();
    ++countCacheInflight[iProc];
#ifdef DEBUG_COUNT_CACHE
    auto header = reinterpret_cast<CacheHeader *>(pFlush->getBuffer());
    header->sequence = countCacheSend[iProc]++;
#endif
    assert(iProc != Proc());
    MPI_Issend(pFlush->getBuffer(), pFlush->getCount(), MPI_BYTE, iProc,
               MDL_TAG_CACHECOM, commMDL, newRequest(pFlush));
}

// When the send finishes, the buffer is added back to the list of free reply buffers
void mpiClass::FinishCacheReply(mdlMessageCacheReply *pFlush) {
    auto iProc = pFlush->getRankTo();
    freeCacheReplies.push_front(pFlush);
    --countCacheInflight[iProc];
    expedite_flush(iProc);
}

// On the requesting node, the data is copied back to the request, and the message
// is queued back to the requesting thread. It will continue with the result.
// If the REQUEST send for this data is not complete, then we do not send back the
// result, rather we wait to rendezvous with the MPI_Isend completion.
int mpiClass::CacheReceiveReply(int count, CacheHeader *ph, int iProcFrom) {
    assert( count >= sizeof(CacheHeader) );
    assert(ph->mid == CacheMessageType::REPLY);
    int iCore = ph->idTo - Self();
    assert(iCore >= 0 && iCore < Cores());
    auto c = pmdl[iCore]->cache[ph->cid].get();
    assert(c->isActive());
    auto pack_size = c->cache_helper->pack_size();
    assert(pack_size <= MDL_CACHE_DATA_SIZE);
    int iLineSize = c->getLineElementCount() * pack_size;
    //assert( count == sizeof(CacheHeader) + iLineSize || count == sizeof(CacheHeader));
    if (CacheRequestMessages[iCore]) { // This better be true (unless we implement prefetch)
        mdlMessageCacheRequest *pRequest = CacheRequestMessages[iCore];
        assert(pRequest->pLine);
        pRequest->header.nItems = ph->nItems;
        memcpy(pRequest->pLine, ph + 1, ph->nItems * iLineSize);
        if (CacheRequestRendezvous[iCore]) CacheRequestRendezvous[iCore] = false;
        else {
            CacheRequestMessages[iCore]->sendBack();
            CacheRequestMessages[iCore] = NULL;
        }
    }
    return sizeof(CacheHeader) + ph->nItems * iLineSize;
}

// Later when we have found an empty cache element, this is called to wait for the result
// and to copy it into the buffer area.
void *CACHE::finishRequest(uint32_t uLine, uint32_t uId, uint32_t size, const void *pKey, bool bVirtual, void *dst, const void *src) {
    auto success = mdl->finishCacheRequest(iCID, uLine, uId, size, pKey, bVirtual, dst, src);
    mdl->TimeAddWaiting();
    return success;
}

void *mdlClass::finishCacheRequest(int cid, uint32_t uLine, uint32_t uId, uint32_t size, const void *pKey, bool bVirtual, void *data, const void *src) {
    int iCore = uId - mpi->Self();
    auto c = cache[cid].get();
    int s, n;
    uint32_t iIndex = uLine << c->nLineBits;
    auto pack_size = c->cache_helper->pack_size();

    s = iIndex;
    n = s + c->getLineElementCount();

    // A virtual fetch results in an empty cache line (unless init is called below)
    if (bVirtual) {
        assert(!src);
        memset(data, 0, c->iLineSize);
    }
    // Local requests must be from a combiner cache if we get here,
    // otherwise we would have simply returned a reference
    else if (iCore >= 0 && iCore < Cores() ) {
        mdlClass *omdl = pmdl[iCore];
        auto oc = omdl->cache[cid].get();
        if (size) {
            if (!src) return nullptr;
            memcpy(data, src, oc->iDataSize);
        }
        else {
            if (n > oc->nData) n = oc->nData;
            auto pData = static_cast<char *>(data);
            for (auto i = s; i < n; i++ ) {
                auto p = oc->ReadLock(i);
                memcpy(pData, p, oc->iDataSize);
                oc->ReadUnlock(p);
                pData += oc->iDataSize;
            }
        }
    }
    // Here we wait for the reply, and copy the data into the ARC cache
    else {
        assert(!src);
        mdlMessageCacheRequest &M = dynamic_cast<mdlMessageCacheRequest &>(waitQueue(queueCacheReply));
        if (M.header.nItems == 0) return nullptr;
        auto nLine = c->getLineElementCount();
        auto pData = static_cast<char *>(data);
        for (auto i = 0; i < nLine; ++i) {
            c->cache_helper->unpack(&pData[i * c->iDataSize], &c->OneLine[i * pack_size], pKey);
        }
    }

    // A combiner cache can intialize some / all of the elements
    auto pData = static_cast<char *>(data);
    for (auto i = s; i < n; i++ ) {
        c->cache_helper->init(pData);
        pData += c->iDataSize;
    }
    return data;
}

/*****************************************************************************\
* Here we follow a Cache Flush
\*****************************************************************************/

// When we need to evict an element (especially at the end) this routine is called by the ARC cache.
void CACHE::flushElement( uint32_t uLine, uint32_t uId, uint32_t size, const void *pKey, const void *data) {
    if (!mdl->coreFlushBuffer->canBuffer(getLineElementCount()*cache_helper->flush_size()+size)) mdl->flush_core_buffer();
    mdl->coreFlushBuffer->addBuffer(iCID, mdlSelf(mdl), uId, uLine);
    mdl->coreFlushBuffer->addBuffer(size, pKey);
    auto pData = static_cast<const char *>(data);
    for (auto i = 0; i < getLineElementCount(); ++i) {
        cache_helper->flush(mdl->coreFlushBuffer->getBuffer(cache_helper->flush_size()), pData);
        pData += iDataSize;
    }
}

// This sends our local flush buffer to the MPI thread to be flushed (if it's full for example)
void mdlClass::flush_core_buffer() {
    if (!coreFlushBuffer->isEmpty()) {
        enqueue(*coreFlushBuffer, coreFlushBuffers);
        mdlMessageFlushFromCore &M = dynamic_cast<mdlMessageFlushFromCore &>(waitQueue(coreFlushBuffers));
        M.emptyBuffer(); // NOTE: this is a different buffer -- we shouldn't have to really wait
        coreFlushBuffer = &M;
    }
}

// Individual cores flush elements from their cache (see flushElement()). When a
// buffer is filled (or at the very end), this routine is called to flush it.
// Each element can be destined for a diffent node so we process them separately.
void mpiClass::MessageFlushFromCore(mdlMessageFlushFromCore *pFlush) {
    char *pData;
    int count;
    count = pFlush->getCount();
    auto ca = reinterpret_cast<CacheHeader *>(pFlush->getBuffer());
    auto c = pmdl[0]->cache[ca->cid].get();
    while (count > 0) {
        int iLineSize = c->getLineElementCount() * c->cache_helper->flush_size();
        int iSize = iLineSize + c->key_size();
        pData = (char *)(ca + 1);
        flush_element(ca, iSize);
        pData += iSize;
        ca = (CacheHeader *)pData;
        count -= sizeof(CacheHeader) + iSize;
    }
    assert(count == 0);
    pFlush->sendBack();
}

// If there is a pending flush, and it has been expedited, then send it
void mpiClass::expedite_flush(int iProc) {
    std::set<int> ready;
    // Request messages are expedited if the number of cache messages inflight to this rank
    // falls below a certain threshold. Otherwise multiple REQUEST messages are batched.
    if (iCacheMaxInflight) {
        auto iFlush = flushBuffersByRank[iProc];
        // Always expedite REPLY messages to avoid deadlock situations
        if (iFlush != flushHeadBusy.end()) {
            auto pFlush = *iFlush;
            if (pFlush->contains(CacheMessageType::REPLY)) {
                ready.emplace(iProc);
            }
        }
        // Expedite REQUEST message if they meet the criteria
        if (countCacheInflight[iProc]<iCacheMaxInflight) {
            if (iFlush != flushHeadBusy.end()) {
                auto pFlush = *iFlush;
                if (pFlush->contains(CacheMessageType::REQUEST)) {
                    ready.emplace(iProc);
                }
            }
        }
    }
    // Try to process as many deferred requests as we can
    while (!PendingRequests.empty()) {
        auto &hdr = PendingRequests.front().header;
        iProc = ThreadToProc(hdr.idFrom);

        int iCore = hdr.idTo - Self();
        assert(iCore >= 0 && iCore < Cores());
        auto c = pmdl[iCore]->cache[hdr.cid].get();
        assert(c->isActive());
        auto pack_size = c->cache_helper->pack_size();
        assert(pack_size <= MDL_CACHE_DATA_SIZE);
        int iSize = sizeof(CacheHeader) + pack_size * c->getLineElementCount(); // Maximum needed for response

        auto flush = get_flush_buffer(iProc, iSize, false);
        if (!flush) break;
        BufferCacheResponse(flush, PendingRequests.front());
        PendingRequests.pop_front();
        ready.emplace(iProc);
    }
    // Now send the flush buffers that we marked above
    for (auto i : ready) {
        auto iFlush = flushBuffersByRank[i];
        // We do need to check this because we could have flushed a buffer and had no free buffers
        if (iFlush != flushHeadBusy.end()) {
            auto pFlush = *iFlush;
            flushHeadBusy.erase(iFlush);
            flushBuffersByRank[iProc] = flushHeadBusy.end();
            pFlush->action(this);
        }
    }

}

mdlMessageFlushToRank *mpiClass::get_flush_buffer(int iProc, int iSize, bool bWait) {
    auto iFlush = flushBuffersByRank[iProc];
    mdlMessageFlushToRank *pFlush;
    // If the current buffer is full, then send it and remove it from "busy".
    // It will reappear on the "free" list when the send has completed.
    if (iFlush != flushHeadBusy.end()) {
        pFlush = *iFlush;
        flushHeadBusy.erase(iFlush); // We want to add this to the front of the queue later
        if ( !pFlush->canBuffer(iSize)) {
            pFlush->action(this);
            flushBuffersByRank[iProc] = flushHeadBusy.end();
            pFlush = NULL; // We need a buffer
        }
    }
    else pFlush = NULL; // We need a buffer
    if (pFlush == NULL) {
        if (flushBuffCount - flushHeadBusy.size() < flushBuffCount / 5) { // Keep 20% free or in flight
            pFlush = flushHeadBusy.back(); // Remove LRU element
            flushHeadBusy.pop_back();
            flushBuffersByRank[pFlush->getRankTo()] = flushHeadBusy.end();
            pFlush->action(this);
        }
        if (flushHeadFree.empty()) {
            if (!bWait) return nullptr;
            do {
                finishRequests();
                yield();
            } while (flushHeadFree.empty());
        }
        pFlush = flushHeadFree.front();
        flushHeadFree.pop_front();
        pFlush->setRankTo(iProc);
        pFlush->emptyBuffer();
    }
    assert(pFlush->canBuffer(iSize));
    flushHeadBusy.push_front(pFlush); // Front = most recently used
    flushBuffersByRank[iProc] = flushHeadBusy.begin();
    return pFlush;
}

// This takes a single flush element and adds it to the correct buffer for
// the destination rank.
void mpiClass::flush_element(CacheHeader *pHdr, int iSize) {
    int iProc = ThreadToProc(pHdr->idTo);
    auto pFlush = get_flush_buffer(iProc, iSize);
    pFlush->addBuffer(iSize, pHdr);
}

// Called to flush a buffer full of cache data to a particular rank
void mpiClass::MessageFlushToRank(mdlMessageFlushToRank *pFlush) {
    auto iProc = pFlush->getRankTo();
    ++countCacheInflight[iProc];
    if (pFlush->getCount()>0) {
        if (iProc == Proc()) {
            CacheHeader *ph = reinterpret_cast<CacheHeader *>(pFlush->getBuffer());
            CacheReceive(pFlush->getCount(), ph, iProc);
            FinishFlushToRank(pFlush);
        }
        else {
#ifdef DEBUG_COUNT_CACHE
            auto header = reinterpret_cast<CacheHeader *>(pFlush->getBuffer());
            header->sequence = countCacheSend[iProc]++;
#endif
            assert(iProc != Proc());
            MPI_Issend(pFlush->getBuffer(), pFlush->getCount(), MPI_BYTE, iProc,
                       MDL_TAG_CACHECOM, commMDL, newRequest(pFlush));
        }
    }
    else FinishFlushToRank(pFlush);
}

// When the send is done, add this back to the free list.
void mpiClass::FinishFlushToRank(mdlMessageFlushToRank *pFlush) {
    auto iProc = pFlush->getRankTo();
    flushHeadFree.push_front(pFlush);
    --countCacheInflight[iProc];
    expedite_flush(iProc);
}

// Here we receive the flush buffer from a remote node. We have to split it
// apart by target thread by calling queue_local_flush() for each element.
int mpiClass::CacheReceiveFlush(int count, CacheHeader *ph, int iProcFrom) {
    assert( count >= sizeof(CacheHeader) );
    assert(ph->mid == CacheMessageType::FLUSH);
    bookkeeping();
    int iCore = ph->idTo - Self();
    auto c = pmdl[iCore]->cache[ph->cid].get();
    assert(c->isActive());
    assert(c->modify());
    auto key_size = c->key_size();
    auto iLineSize = c->getLineElementCount() * c->cache_helper->flush_size();
    if (key_size == 0) {
        // For some operations (e.g., grid assignment), we use core 0 and an index into the process.
        // Here we convert this to the correct index and core, updating iLine and iCore
        int iIndex = ph->iLine << c->nLineBits;
        while (iIndex >= c->nData) {
            iIndex -= c->nData;
            assert(iCore + 1 < Cores());
            c = pmdl[++iCore]->cache[ph->cid].get();
        }
        ph->iLine = iIndex >> c->nLineBits;
    }
    ph->idTo = iCore;
    queue_local_flush(ph);
    return sizeof(CacheHeader) + iLineSize + key_size;
}

// Send a buffer to a local thread
void mpiClass::queue_local_flush(CacheHeader *ph) {
    int iCore = ph->idTo;
    mdlClass *mdl1 = pmdl[iCore];
    auto c = mdl1->cache[ph->cid].get();
    auto key_size = c->key_size();
    mdlMessageFlushToCore *flush = flushBuffersByCore[iCore];
    if (flush == NULL) flush = & dynamic_cast<mdlMessageFlushToCore &>(localFlushBuffers.wait());
    auto iLineSize = c->getLineElementCount() * c->cache_helper->flush_size();
    if (!flush->canBuffer(iLineSize + key_size)) {
        mdl1->wqCacheFlush.enqueue(* flush, localFlushBuffers);
        flush = & dynamic_cast<mdlMessageFlushToCore &>(localFlushBuffers.wait());
        assert(flush->getCount()==0);
    }
    flush->addBuffer(iLineSize + key_size, ph);
    flushBuffersByCore[iCore] = flush;
}

// Received by the worker thread: call the combiner function for each element
void mdlClass::MessageFlushToCore(mdlMessageFlushToCore *pFlush) {
    CacheHeader *ca = reinterpret_cast<CacheHeader *>(pFlush->getBuffer());
    int count = pFlush->getCount();
    while (count > 0) {
        char *pData = reinterpret_cast<char *>(ca + 1);
        auto c = cache[ca->cid].get();
        auto key_size = c->key_size();
        auto iLineSize = c->getLineElementCount() * c->cache_helper->flush_size();
        if (key_size) { // We are updating an advanced key
            assert(c->hash_table);
            assert(c->getLineElementCount()==1);
            void *data = c->hash_table->lookup(ca->iLine, pData);
            // If we don't have an element, try to insert it and then add it to the hash table
            // Create can return nullptr in which case we just ignore it.
            if (!data && (data = c->cache_helper->create(key_size, pData)) )
                c->hash_table->insert(ca->iLine, pData, data);
            if (data) {
                c->cache_helper->combine(data, pData + key_size, pData);
            }
            pData += key_size + c->cache_helper->flush_size();
        }
        else {
            uint32_t uIndex = ca->iLine << c->nLineBits;
            int s = uIndex;
            int n = s + c->getLineElementCount();
            for (int i = s; i < n; i++ ) {
                if (i < c->nData) {
                    // Here we update a local element that has been flushed back to us.
                    // The danger here is another thread may want to read the value and
                    // end up with an inconsistent view so we lock for write.
                    auto p = c->WriteLock(i);
                    c->cache_helper->combine(p, pData, nullptr);
                    c->WriteUnlock(p);
                }
                pData += c->cache_helper->flush_size();
            }
        }
        count -= sizeof(CacheHeader) + iLineSize + key_size;
        ca = (CacheHeader *)pData;
    }
    assert(count == 0);
    pFlush->emptyBuffer();
    pFlush->sendBack();
}

void CACHE::combineElement(uint32_t uLine, uint32_t uId, uint32_t size, const void *pKey, const void *data) {

}


/*****************************************************************************\
* The following are used to end or synchronize a cache
\*****************************************************************************/

// Continues to process incoming cache requests until all threads on all nodes reach this point
extern "C" void mdlCacheBarrier(MDL mdl, int cid) { static_cast<mdlClass *>(mdl)->CacheBarrier(cid); }
void mdlClass::CacheBarrier(int cid) {
    ThreadBarrier(true);
}

// This does the same thing as a CacheBarrier(), but in addition all data from all caches will have been
// fully flushed and synchonized between all threads on all nodes before it returns.
extern "C" void mdlFlushCache(MDL mdl, int cid) { static_cast<mdlClass *>(mdl)->FlushCache(cid); }
void mdlClass::FlushCache(int cid) {
    auto c = cache[cid].get();

    TimeAddComputing();
    c->clear();
    flush_core_buffer();
    ThreadBarrier();
    if (Core()==0) { // This flushes all buffered data, not just our thread
        enqueueAndWait(mdlMessageCacheFlushOut());
    }
    ThreadBarrier(true); // We must wait for all threads on all nodes to finish with this cache
    //mdl_MPI_Barrier();
    if (Core()==0) { // This flushes all buffered data, not just our thread
        enqueueAndWait(mdlMessageCacheFlushLocal());
    }
    ThreadBarrier();
    TimeAddSynchronizing();
}

// This sends all incomplete buffers to the correct rank
void mpiClass::MessageCacheFlushOut(mdlMessageCacheFlushOut *message) {
    std::fill(flushBuffersByRank.begin(), flushBuffersByRank.end(), flushHeadBusy.end());
    for (auto pFlush : flushHeadBusy) { pFlush->action(this); }
    flushHeadBusy.clear();
    while (flushHeadFree.size() < flushBuffCount) {
        bookkeeping();
        checkMPI();
        yield();
    }
    message->sendBack();
}

// Here we process any remainging incoming buffers
void mpiClass::MessageCacheFlushLocal(mdlMessageCacheFlushLocal *message) {
    mdlMessageFlushToCore *flush;
    for (int iCore = 0; iCore < Cores(); ++iCore) {
        if ((flush = flushBuffersByCore[iCore])) {
            pmdl[iCore]->wqCacheFlush.enqueue(* flush, localFlushBuffers);
            flushBuffersByCore[iCore] = NULL;
        }
    }
    message->sendBack();
}

extern "C" void mdlFinishCache(MDL mdl, int cid) { static_cast<mdlClass *>(mdl)->FinishCache(cid); }
void mdlClass::FinishCache(int cid) {
    auto c = cache[cid].get();

    TimeAddComputing();
    FlushCache(cid);
    enqueueAndWait(mdlMessageCacheClose());
    ThreadBarrier();
    c->close();
    TimeAddSynchronizing();
}

// Close the cache by cancelling the recieve (if there are no more open caches)
void mpiClass::MessageCacheClose(mdlMessageCacheClose *message) {
    assert(nOpenCaches > 0);
    --nOpenCaches;
    if (nOpenCaches == 0) {
        // Make sure that there are no cache message inflight at this point; that would potentially be bad.
        assert(std::all_of(countCacheInflight.begin(), countCacheInflight.end(), [](int i) { return i == 0; }));
#ifdef DEBUG_COUNT_CACHE
        std::vector<uint64_t> countRecvExpected(Procs());
        MPI_Alltoall(countCacheSend.data(), 1, MPI_UINT64_T, countRecvExpected.data(), 1, MPI_UINT64_T, commMDL);
        bool bBAD = false;
        for (auto i = 0; i < Procs(); ++i) {
            if (countCacheRecv[i] != countRecvExpected[i]) {
                printf("Rank %d received %" PRIu64 " from rank %d but expected %" PRIu64 "\n",
                       Proc(), countCacheRecv[i], i, countRecvExpected[i]);
                bBAD = true;
            }
        }
        if (bBAD) {fflush(stdout); sleep(1); MPI_Abort(commMDL, 1); }
#endif
        for (auto i = SendReceiveMessages.begin(); i != SendReceiveMessages.end(); ++i) {
            if (dynamic_cast<mdlMessageCacheReceive *>(*i) != nullptr)
                MPI_Cancel(&SendReceiveRequests[i - SendReceiveMessages.begin()]);
        }
    }
    message->sendBack();
}

void mdlClass::SetCacheMaxInflight(int iMax) {
    if (Core()==0) mpi->SetCacheMaxInflight(iMax);
}

/*****************************************************************************\
*
* The following is the MPI thread only functions
*
\*****************************************************************************/

mpiClass::mpiClass(int (*fcnMaster)(MDL, void *), void *(*fcnWorkerInit)(MDL), void (*fcnWorkerDone)(MDL, void *), int argc, char **argv)
    : mdlClass(this, fcnMaster, fcnWorkerInit, fcnWorkerDone, argc, argv) {
}

mpiClass::~mpiClass() {
}

// While FFTW does its thing we need to wait without spinning. This function does that.
void mpiClass::pthreadBarrierWait() {
    pthread_barrier_wait(&barrier);
}

/// @brief Swap elements between MPI ranks
/// @param buffer start of memory buffer to exchange
/// @param count total number of elements that could fit in this memory
/// @param datasize size of each data element
/// @param counts how many elements to send to each rank
/// @return
int mdlClass::swapglobal(void *buffer, dd_offset_type count, dd_offset_type datasize, /*const*/ dd_offset_type *counts) {
    auto me = Proc();

    ddBuffer = static_cast<char *>(buffer); // Our local buffer as a "char *"
    auto pi = [this, datasize](dd_offset_type i) {      // Returns a pointer to the i'th element of our buffer
        return ddBuffer + i * datasize;
    };

    // These vectors are part of the mdlClass; this is how we exchange information with the MPI thread.
    ddOffset.reserve(Procs()); ddOffset.clear();    // Index of first element to send to each rank
    ddCounts.reserve(Procs()); ddCounts.clear();    // Number of elements to send to each rank
    dd_offset_type offset = 0;
    for (auto i = 0; i < Procs(); ++i) {
        ddOffset.push_back(offset);     // Index of the element
        ddCounts.push_back(counts[i]);  // Count of elements
        offset += counts[i];
    }
    auto my_count   = counts[me];
    ddCounts[me]    = 0;            // We don't send anything
    ddFreeOffset[0] = ddOffset[me]; // A hole could open up here
    ddFreeCounts[0] = 0;            // but there is currently no space
    ddFreeOffset[1] = offset;       // There could also be room at the end
    ddFreeCounts[1] = count - offset; // and we currently have this much

    // This will combine the elements we receive in the lower segment with our data
    // The data between the end of ddFree[0] and us are new elements
    auto coalesce_lower = [this, me, &my_count]() {
        auto free_end = ddFreeOffset[0] + ddFreeCounts[0];
        my_count += ddOffset[me] - free_end;
        ddOffset[me] = free_end;
    };

    int iVote = 0;
    bool bContinue = false; // Most threads vote not to continue

    do {
        ThreadBarrier();
        if (Core()==0) {
            mdlMessageSwapGlobal swap(datasize);
            enqueueAndWait(swap);
            bContinue = !swap.is_finished();
        }
        iVote = ThreadBarrier(false, bContinue);

        coalesce_lower();

        // Compact for next iteration. This can open up two holes; one before me and one at the end
        // Rank zero has no hole before.
        offset = 0;
        for (auto i = 0; i < Procs(); ++i) {
            if (i == me) offset = ddOffset[me] + my_count;
            else {
                auto n = ddCounts[i];
                if (n && ddOffset[i]!=offset) std::memmove(pi(offset), pi(ddOffset[i]), n * datasize );
                ddOffset[i] = offset;
                offset += n;
            }
        }
        auto n = ddFreeOffset[1] - offset;  // Free space in the upper segment is increased by this
        ddFreeCounts[1] += n;
        ddFreeOffset[1] -= n;
        if (me) {                           // If there could be a free segment before me, adjust the size
            ddFreeCounts[0] = ddOffset[me] - (ddOffset[me - 1] + ddCounts[me - 1]);
            ddFreeOffset[0] = ddOffset[me] - ddFreeCounts[0];
        }
    } while (iVote);

    assert(ddFreeOffset[0]==0);
    assert(ddFreeCounts[0]==ddOffset[me]);  // Offset is zero, so this must be true
    assert(ddFreeOffset[1]==ddOffset[me] + my_count);
    if (ddFreeCounts[0]) {
        auto src = ddFreeOffset[1] + ddFreeCounts[1];
        auto nTop = count - src;
        auto nMove = std::min(ddFreeCounts[0], nTop);
        ddFreeCounts[1] += nMove;
        ddFreeCounts[0] -= nMove;   // Both the end and count (offset is zero)
        my_count += nMove;
        ddOffset[me] -= nMove;
        std::memmove(pi(ddOffset[me]), pi(src), nMove * datasize );
    }
    // Now we either have a hole in the bottom, or we have extra above, but not both.
    if (ddFreeCounts[0]) {          // We have a hole at the bottom
        auto beg = ddOffset[me];
        auto end = ddFreeOffset[1]; // Free space is immediatly after us
        auto dst = ddFreeOffset[0];
        auto nMove = beg - dst;
        if (end - beg <= nMove) {     // The hole is larger; just move down
            std::memmove(pi(dst), pi(beg), (end - beg)*datasize );
            // count = dst + (end - beg);
        }
        else {                      // Fill the hole from the elements at the end
            std::memcpy(pi(dst), pi(end - nMove), nMove * datasize );
            // count = end - nMove;
        }
    }
    else {                          // Only excess at the top
        auto dst = ddFreeOffset[1]; // Free space is immediatly after us
        auto src = ddFreeOffset[1] + ddFreeCounts[1];
        auto nMove = count - src;
        std::memmove(pi(dst), pi(src), nMove * datasize );
        ddFreeOffset[1] -= nMove;
        ddFreeCounts[1] += nMove;
        my_count += nMove;
        // count = dst + nMove;
    }
    return my_count;
}

void mpiClass::MessageSwapGlobal(mdlMessageSwapGlobal *message) {
    auto datasize = message->datasize();
    auto me = Proc();

    struct free_segment {
        int iThread;                // Thread or Core ID
        int iList;                  // Which list
        dd_offset_type nCount;    // Count of free elements
        free_segment() = default;
        free_segment(int iThread, dd_offset_type nCount, int iList = 0)
            : iThread(iThread), iList(iList), nCount(nCount) {}
        bool operator<(const free_segment &rhs) const { return nCount < rhs.nCount; }
    };

    // We will do a transfer of a single segment to each MPI rank, so let's choose the largest
    std::vector<int>      largest(Procs(), 0);   // Core with the largest amount to send to each rank
    std::vector<dd_offset_type> scounts(Procs(), 0);   // How many we can send to each rank
    for (auto i = 0; i < Cores(); ++i) {
        auto other = pmdl[i];
        // Calculate how many we can send in one go (the largest segment)
        for (auto j = 0; j < Procs(); ++j) {
            if (other->ddCounts[j] > scounts[j]) {
                largest[j] = i;
                scounts[j] = other->ddCounts[j];
            }
        }
    }
    assert(scounts[me]==0);

    auto nSend = std::accumulate(scounts.begin(), scounts.end(), dd_offset_type(0));
    dd_offset_type nMaxSend;
    /* We are done when there is nothing globally left to send */
    MPI_Allreduce(&nSend, &nMaxSend, 1, mpi_type(nSend), MPI_MAX, commMDL);
    if (nMaxSend == 0) {
        message->finished(true);
        message->sendBack();
        return;
    }

    // First, tell all of our peers how many data elements we are willing to send
    std::vector<dd_offset_type> rcounts(Procs());
    assert(mpi_type(scounts[0]) == mpi_type(rcounts[0]));
    MPI_Alltoall(scounts.data(), 1, mpi_type(scounts[0]), rcounts.data(), 1, mpi_type(rcounts[0]), commMDL);
    // rcounts[] now contains a list of how many we could receive

    // Turn this into a "requests" list so we can match
    std::priority_queue<free_segment> RecvList;
    for (auto i = 0; i < Procs(); ++i) {
        if (rcounts[i]) RecvList.emplace(i, rcounts[i]);
    }

    // build a list of free segments
    std::priority_queue<free_segment> FreeList;
    for (auto i = 0; i < Cores(); ++i) {
        auto other = pmdl[i];
        // Add the free segment(s) to the free list
        for (auto j = 0; j < 2; ++j) {
            if (other->ddFreeCounts[j]) FreeList.emplace(i, other->ddFreeCounts[j], j);
        }
    }

    MPI_Datatype mpitype;
    MPI_Type_contiguous(datasize, MPI_BYTE, &mpitype);
    MPI_Type_commit(&mpitype);
    std::vector<MPI_Request> requests;
    std::vector<MPI_Status> statuses;

    // Try to match up as many requests as we can
    std::fill(rcounts.begin(), rcounts.end(), 0);
    while (!RecvList.empty() && !FreeList.empty()) {
        // Grab this segment (we add it back below if it isn't empty)
        auto iList = FreeList.top().iList;      // Segment list (0 or 1)
        auto iCore = FreeList.top().iThread;    // Segment is on this core
        auto nCount= FreeList.top().nCount;     // It can handle this many elements
        FreeList.pop();

        auto other = pmdl[iCore];
        auto po = [other, datasize](dd_offset_type i) {return other->ddBuffer + i * datasize;};

        auto iRank = RecvList.top().iThread;
        dd_offset_type nSend = std::min(RecvList.top().nCount, nCount);
        RecvList.pop();

        rcounts[iRank] = nSend;

        // Consume from this segment
        assert(other->ddFreeCounts[iList]>=nSend);
        other->ddFreeCounts[iList] -= nSend;
        auto iDest = other->ddFreeOffset[iList] + other->ddFreeCounts[iList];
        if (other->ddFreeCounts[iList]) FreeList.emplace(iCore, other->ddFreeCounts[iList], iList);

        // Post the MPI receive
        requests.emplace_back();
        statuses.emplace_back();
        MPI_Irecv(po(iDest), nSend, mpitype, iRank, MDL_TAG_SWAP, commMDL, &requests.back() );
    }

    // Now communicate how many we actually have to send
    MPI_Alltoall(rcounts.data(), 1, mpi_type(rcounts[0]), scounts.data(), 1, mpi_type(scounts[0]), commMDL);

    /* Now we post the matching sends. We can use "rsend" here because the receives are posted. */
    for (auto i = 0; i < Procs(); ++i) {
        if (scounts[i]) {
            auto other = pmdl[largest[i]];
            auto po = [other, datasize](dd_offset_type i) {return other->ddBuffer + i * datasize;};
            assert(scounts[i]<=other->ddCounts[i]);
            other->ddCounts[i] -= scounts[i];
            requests.emplace_back();
            statuses.emplace_back();
            MPI_Irsend(po(other->ddOffset[i]+other->ddCounts[i]), scounts[i], mpitype,
                       i, MDL_TAG_SWAP, commMDL, &requests.back() );
        }
    }

    /* Wait for all communication (send and recv) to finish */
    MPI_Waitall(requests.size(), &requests.front(), &statuses.front());
    MPI_Type_free(&mpitype);

    message->sendBack();
}


//! Swap elements of a memory buffer between local threads.
//! @param buffer A pointer to the start of the memory block
//! @param count The total number of elements that count fit in the memory block
//! @param datasize The size (in bytes) of each element
//! @param counts The number of elements destined for each thread
int mdlClass::swaplocal(void *buffer, dd_offset_type count, dd_offset_type datasize, /*const*/ dd_offset_type *counts) {
    auto pBegin = static_cast<char *>(buffer);
    auto pi = [pBegin, datasize](dd_offset_type i) {return pBegin + i * datasize;};
    auto me = Core();

    // Example layout (we are rank 3)
    // +-------------------------+----------+---------------+------------+
    // | Thread 0 to 2 elements  | thread 3 | thread 4-     | unused     |
    // +-------------------------+----------+---------------+------------+

    // Setup counts and offsets. Other threads will update as they take elements
    std::vector<dd_offset_type> ddFreeBeg, ddFreeEnd;
    ddFreeBeg.reserve(Cores()); ddFreeEnd.reserve(Cores());
    ddOffset.reserve(std::max(Procs(), Cores())); ddOffset.clear();
    ddCounts.reserve(std::max(Procs(), Cores())); ddCounts.clear();
    ddBuffer = pBegin;
    dd_offset_type offset = 0;
    for (auto i = 0; i < Cores(); ++i) {
        ddCounts.push_back(counts[i]);
        ddOffset.push_back(offset);
        ddFreeBeg.push_back(offset);
        offset += counts[i];
    }
    assert(count >= offset);

    // Check that we have enough space to receive everything
    ThreadBarrier();
    dd_offset_type nTotal = 0;
    for (auto i = 0; i < Cores(); ++i) {
        auto other = pmdl[i];
        nTotal += other->ddCounts[me];
    }
    ThreadBarrier();
    assert(nTotal < count);

    // We have a special case for "our" element. Nobody will be taking them, so we
    // setup the free space at the end as available.
    ddFreeBeg[me] = offset;
    ddOffset[me] = count;

    dd_offset_type iTake = 0, iFree = 0;
    do {
        ddFreeEnd.clear();
        std::copy(ddOffset.begin(), ddOffset.end(), std::back_inserter(ddFreeEnd));
        ThreadBarrier();
        iTake = iFree = 0;
        while (iTake < Cores() && iFree < Cores()) {
            dd_offset_type nTake, nFree;
            // Find some data to transfer to us
            while (iTake == me || (iTake < Cores() && (nTake = pmdl[iTake]->ddCounts[me])==0)) ++iTake;
            if (iTake == Cores()) break; // We have moved everything we can (we are done)
            // Now find somewhere to move it
            while (iFree < Cores() && (nFree = ddFreeEnd[iFree] - ddFreeBeg[iFree]) == 0) ++iFree;
            if (iFree == Cores()) break; // No more free space -- we have to iterate
            if (nTake > nFree) nTake = nFree;
            auto other = pmdl[iTake];
            auto po = [other, datasize](dd_offset_type i) {return other->ddBuffer + i * datasize;};
            auto src = other->ddOffset[me];
            auto dst = ddFreeBeg[iFree];
            memcpy(pi(dst), po(src), nTake * datasize );
            other->ddCounts[me] -= nTake;
            other->ddOffset[me] += nTake;
            ddFreeBeg[iFree] += nTake;
        }
    } while (ThreadBarrier(false, iTake < Cores()));

    // Now compact the storage
    dd_offset_type dst = 0, src = 0;
    for (auto i = 0; i < Cores(); ++i) {
        auto n = i == me ? counts[me] : ddFreeBeg[i] - src;
        std::memmove(pi(dst), pi(src), n * datasize );
        dst += n;
        src += counts[i];
    }
    // Compact the elements from the intial free space block
    auto n = ddFreeBeg[me] - src;
    std::memmove(pi(dst), pi(src), n * datasize );
    dst += n;

    ThreadBarrier();
    return dst;
}

// An indication that a thread has finished
void mpiClass::MessageSTOP(mdlMessageSTOP *message) {
    --nActiveCores;
    message->sendBack();
}

void mpiClass::MessageBarrierMPI(mdlMessageBarrierMPI *message) {
    MPI_Ibarrier(commMDL, newRequest(message));
}
void mpiClass::MessageGridShare(mdlMessageGridShare *share) {
    MPI_Allgather(&share->grid->sSlab, sizeof(*share->grid->rs), MPI_BYTE,
                  share->grid->rs, sizeof(*share->grid->rs), MPI_BYTE,
                  commMDL);
    MPI_Allgather(&share->grid->nSlab, sizeof(*share->grid->rn), MPI_BYTE,
                  share->grid->rn, sizeof(*share->grid->rn), MPI_BYTE,
                  commMDL);
    share->sendBack();
}

#ifdef MDL_FFTW
// MPI thread: initiate a real to complex transform
void mpiClass::MessageDFT_R2C(mdlMessageDFT_R2C *message) {
    FFTW3(execute_dft_r2c)(message->fft->fplan, message->data, message->kdata);
    pthreadBarrierWait();
}

// MPI thread: initiate a complex to real transform
void mpiClass::MessageDFT_C2R(mdlMessageDFT_C2R *message) {
    FFTW3(execute_dft_c2r)(message->fft->iplan, message->kdata, message->data);
    pthreadBarrierWait();
}

void mpiClass::MessageFFT_Sizes(mdlMessageFFT_Sizes *sizes) {
    sizes->nLocal = 2 * FFTW3(mpi_local_size_3d_transposed)
                    (sizes->n3, sizes->n2, sizes->n1 / 2 + 1, commMDL,
                     &sizes->nz, &sizes->sz, &sizes->ny, &sizes->sy);
    sizes->sendBack();
}

void mpiClass::MessageFFT_Plans(mdlMessageFFT_Plans *plans) {
    auto result = fft_plans.emplace(fft_plan_key(plans->n1, plans->n2, plans->n3), fft_plan_information());
    auto &info = result.first->second;
    if (result.second) { // Just inserted: we need to create the plan!
        info.nLocal = 2 * FFTW3(mpi_local_size_3d_transposed)
                      (plans->n3, plans->n2, plans->n1 / 2 + 1, commMDL,
                       &info.nz, &info.sz, &info.ny, &info.sy);
        info.fplan = FFTW3(mpi_plan_dft_r2c_3d)(
                         plans->n3, plans->n2, plans->n1, plans->data, plans->kdata,
                         commMDL, FFTW_MPI_TRANSPOSED_OUT | (plans->data == NULL?FFTW_ESTIMATE:FFTW_MEASURE) );
        info.iplan = FFTW3(mpi_plan_dft_c2r_3d)(
                         plans->n3, plans->n2, plans->n1, plans->kdata, plans->data,
                         commMDL, FFTW_MPI_TRANSPOSED_IN  | (plans->kdata == NULL?FFTW_ESTIMATE:FFTW_MEASURE) );
    }
    plans->nLocal = info.nLocal;
    plans->nz = info.nz;
    plans->sz = info.sz;
    plans->ny = info.ny;
    plans->sy = info.sy;
    plans->fplan = info.fplan;
    plans->iplan = info.iplan;
    plans->sendBack();
}
#endif

void mpiClass::MessageAlltoallv(mdlMessageAlltoallv *a2a) {
    MPI_Datatype mpitype;
    MPI_Type_contiguous(a2a->dataSize, MPI_BYTE, &mpitype);
    MPI_Type_commit(&mpitype);
    MPI_Ialltoallv(
        a2a->sbuff, a2a->scount, a2a->sdisps, mpitype,
        a2a->rbuff, a2a->rcount, a2a->rdisps, mpitype,
        commMDL, newRequest(a2a));
    MPI_Type_free(&mpitype);
}

void mpiClass::MessageSendRequest(mdlMessageSendRequest *send) {
    int iProc = ThreadToProc(send->target);
    int iCore = send->target - ProcToThread(iProc);
    assert(iCore >= 0);
    int tag = send->tag + MDL_TAG_THREAD_OFFSET * iCore;

    /* Grab a free tag for the reply */
    auto it = std::find(pRequestTargets.begin()+iRequestTarget, pRequestTargets.end(), -1);
    if (it == pRequestTargets.end()) {
        it = std::find(pRequestTargets.begin(), pRequestTargets.begin()+iRequestTarget, -1);
        assert(it != pRequestTargets.begin()+iRequestTarget);
    }
    iRequestTarget = it - pRequestTargets.begin();
    assert(pRequestTargets[iRequestTarget] < 0);
    pRequestTargets[iRequestTarget] = iProc;
    send->header.replyTag = iRequestTarget;
    MPI_Aint disp[2];
    MPI_Get_address(&send->header, &disp[0]);
    MPI_Get_address(send->buf, &disp[1]);
    disp[1] -= disp[0]; disp[0] = 0;
    int blocklen[2] = {sizeof(ServiceHeader), send->header.nInBytes};
    MPI_Datatype type[3] = {MPI_BYTE, MPI_BYTE};
    MPI_Datatype datatype;
    MPI_Type_create_struct (2, blocklen, disp, type, &datatype);
    MPI_Type_commit (&datatype);
    MPI_Isend(&send->header, 1, datatype, iProc, tag, commMDL, newRequest(send));
    MPI_Type_free (&datatype);
}

void mpiClass::MessageSendReply(mdlMessageSendReply *send) {
    int iProc = ThreadToProc(send->iThreadTo);
    int iCore = send->iThreadTo - ProcToThread(iProc);
    assert(iCore >= 0);
    int tag = MDL_TAG_RPL + MDL_TAG_THREAD_OFFSET * send->header.replyTag;

    MPI_Aint disp[2];
    MPI_Get_address(&send->header, &disp[0]);
    MPI_Get_address(&send->Buffer.front(), &disp[1]);
    disp[1] -= disp[0]; disp[0] = 0;
    int blocklen[2] = {sizeof(ServiceHeader), send->header.nInBytes};
    MPI_Datatype type[3] = {MPI_BYTE, MPI_BYTE};
    MPI_Datatype datatype;
    MPI_Type_create_struct (2, blocklen, disp, type, &datatype);
    MPI_Type_commit (&datatype);
    MPI_Isend(&send->header, 1, datatype, iProc, tag, commMDL, newRequest(send));
    MPI_Type_free (&datatype);
}

void mpiClass::MessageSend(mdlMessageSend *send) {
    int iProc = ThreadToProc(send->target);
    int iCore = send->target - ProcToThread(iProc);
    assert(iCore >= 0);
    int tag = send->tag + MDL_TAG_THREAD_OFFSET * iCore;
    MPI_Issend(send->buf, send->count, MPI_BYTE, iProc, tag, commMDL, newRequest(send));
}

void mpiClass::MessageReceive(mdlMessageReceive *send) {
    int iProc = send->target == MPI_ANY_SOURCE ? MPI_ANY_SOURCE : ThreadToProc(send->target);
    int iCore = send->iCoreFrom;
    assert(iCore >= 0);
    int tag = send->tag + MDL_TAG_THREAD_OFFSET * iCore;
    MPI_Irecv(send->buf, send->count, MPI_BYTE, iProc, tag, commMDL, newRequest(send));
}

void mpiClass::MessageReceiveReply(mdlMessageReceiveReply *send) {
    assert(send->ta.get<pRequestTargets.size());
    int tag = MDL_TAG_RPL + MDL_TAG_THREAD_OFFSET * send->target;
    int iProc = pRequestTargets[send->target];

    if (send->buf) {
        MPI_Aint disp[2];
        MPI_Get_address(&send->header, &disp[0]);
        MPI_Get_address(send->buf, &disp[1]);
        disp[1] -= disp[0]; disp[0] = 0;
        int blocklen[2] = {sizeof(ServiceHeader), send->count};
        MPI_Datatype type[3] = {MPI_BYTE, MPI_BYTE};
        MPI_Datatype datatype;
        MPI_Type_create_struct (2, blocklen, disp, type, &datatype);
        MPI_Type_commit (&datatype);
        MPI_Irecv(&send->header, 1, datatype, iProc, tag, commMDL, newRequest(send));
        MPI_Type_free (&datatype);
    }
    else MPI_Irecv(&send->header, sizeof(ServiceHeader), MPI_BYTE, iProc, tag, commMDL, newRequest(send));
}

void mpiClass::FinishReceiveReply(mdlMessageReceiveReply *send, MPI_Request request, MPI_Status status) {
    assert(pRequestTargets[send->target]>=0);
    pRequestTargets[send->target] = -1;
    send->sendBack();
}

void mpiClass::finishRequests() {
    int nDone = 1; // Prime the loop; we keep trying as long as we get work
    // If there are MPI send / recieve message pending then check if they are complete
    while (!SendReceiveRequests.empty() && nDone) {
        // Here we keep track of entries that we can reuse in newRequest()
        assert(SendReceiveAvailable.size()==0);
        // Should be sufficent, but resize if necessary
        if (SendReceiveIndices.size() < SendReceiveRequests.size())  SendReceiveIndices.resize(SendReceiveRequests.size());
        if (SendReceiveStatuses.size() < SendReceiveRequests.size()) SendReceiveStatuses.resize(SendReceiveRequests.size());
        auto rc = MPI_Testsome(SendReceiveRequests.size(), &SendReceiveRequests.front(),
                               &nDone, &SendReceiveIndices.front(), &SendReceiveStatuses.front() );
        assert(rc == MPI_SUCCESS);
        assert(nDone != MPI_UNDEFINED);
        if (nDone) { // If one or more of the requests has completed
#ifndef NDEBUG
            nRequestsReaped += nDone;
#endif
            assert(SendReceiveMessages.size() == SendReceiveRequests.size()); // basic sanity
            for (int i = 0; i < nDone; ++i) {
                unsigned indx = SendReceiveIndices[i];
                assert(indx < SendReceiveMessages.size());
                mdlMessageMPI *message = SendReceiveMessages[indx];
                MPI_Request request = SendReceiveRequests[indx];
                // If request is not MPI_REQUEST_NULL then it means that we have a persistent communication channel.
                // It is the job of the "finish" routine to start or cancel the request and add it back with newRequest().
                // Elements marked as "NULL" can now be reused in newRequest() so we add them to SendReceiveAvailable
                SendReceiveAvailable.push_back(indx);
                SendReceiveMessages[indx] = nullptr;
                SendReceiveRequests[indx] = MPI_REQUEST_NULL;
                message->finish(this, request, SendReceiveStatuses[i]); // Finish request (which could create / add more requests)
            }
            // Now remove the "marked" elements. The "finish" routines above could have added new entries at the end (but that is valid).
            // By design the list of marked elements is in SendReceiveAvailable, and are in ascending order. We can remove them efficiently with:
            if (SendReceiveAvailable.size()) {
                SendReceiveAvailable.push_back(SendReceiveMessages.size());
                auto d = SendReceiveAvailable[0];
                for (int i = 1; i < SendReceiveAvailable.size(); ++i) {
                    auto s = SendReceiveAvailable[i - 1] + 1;
                    auto e = SendReceiveAvailable[i];
                    while (s < e) {
                        SendReceiveMessages[d] = SendReceiveMessages[s];
                        SendReceiveRequests[d] = SendReceiveRequests[s];
                        s++; d++;
                    }
                }
                SendReceiveMessages.erase(SendReceiveMessages.begin()+d, SendReceiveMessages.end());
                SendReceiveRequests.erase(SendReceiveRequests.begin()+d, SendReceiveRequests.end());
                SendReceiveAvailable.clear();
            }
        }
    }
}
void mpiClass::processMessages() {
    /* These are messages from other threads */
    while (!queueMPInew.empty()) {
        mdlMessage &M = queueMPInew.dequeue();
        M.action(this); // Action normally sends it back, but not always (see finish() below)
    }
}
/*
** This routine must be called often by the MPI thread. It will drain
** any requests from the thread queue, and track MPI completions.
*/
int mpiClass::checkMPI() {
    gpu.flushCompleted();
    /* Start any CUDA work packages */
#ifdef USE_CUDA
    cuda.launch();
#endif
#ifdef USE_METAL
    metal.launch();
#endif
    processMessages();
    finishRequests(); // Check for non-block MPI requests (send, receive, barrier, etc.)
    return nActiveCores;
}

static void TERM_handler(int signo) {
    MPI_Abort(MPI_COMM_WORLD, 130);
}

void mpiClass::KillAll(int signo) {
    for (auto i = 0; i < Cores(); ++i)
        if (i != Core()) pthread_kill(threadid[i], signo);
}

extern "C"
int mdlLaunch(int argc, char **argv, int (*fcnMaster)(MDL, void *), void *(*fcnWorkerInit)(MDL), void (*fcnWorkerDone)(MDL, void *)) {
    mpiClass mdl(fcnMaster, fcnWorkerInit, fcnWorkerDone, argc, argv);
    return mdl.Launch(fcnMaster, fcnWorkerInit, fcnWorkerDone);
}
int mpiClass::Launch(int (*fcnMaster)(MDL, void *), void *(*fcnWorkerInit)(MDL), void (*fcnWorkerDone)(MDL, void *)) {
    int i, j, n, bDiag, bDedicated, thread_support, rc, flag, *piTagUB;
    bool bThreads = false;
    char *p, ach[256];
    int exit_code = 0;
#ifdef USE_HWLOC
    hwloc_topology_t topology = NULL;
    hwloc_cpuset_t set_proc = NULL, set_thread = NULL;
    hwloc_obj_t t = NULL;
#endif

#ifdef USE_ITT
    __itt_domain *domain = __itt_domain_create("MyTraces.MyDomain");
    __itt_string_handle *shMyTask = __itt_string_handle_create("MDL Startup");
    __itt_task_begin(domain, __itt_null, __itt_null, shMyTask);
#endif

    pthread_key_create(&mdl_key, NULL);
    pthread_key_create(&worker_key, NULL);
    pthread_setspecific(mdl_key, static_cast<class mdlClass *>(this));

#ifdef _SC_NPROCESSORS_CONF /* from unistd.h */
    layout.nCores = sysconf(_SC_NPROCESSORS_CONF);
#endif

    /*
    ** Do some low level argument parsing for number of threads, and
    ** diagnostic flag!
    */
    bDiag = 0;
    bDedicated = -1;
    if (argv) {
        for (argc = 0; argv[argc]; argc++) {}
        ;
        for (i = j = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "-sz")) {
                if (argv[++i]) {
                    layout.nCores = atoi(argv[i]);
                    bThreads = true;
                }
            }
            else if (!strcmp(argv[i], "-dedicated")) {
                if (bDedicated < 1) bDedicated = 0;
            }
            else if (!strcmp(argv[i], "+dedicated")) {
                if (bDedicated < 1) bDedicated = 1;
            }
            else if (!strcmp(argv[i], "+sharedmpi")) {
                bDedicated = 2;
            }
            else if (!strcmp(argv[i], "+d") && !bDiag) {
                p = getenv("MDL_DIAGNOSTIC");
                if (!p) p = getenv("HOME");
                if (!p) snprintf(ach, sizeof(ach), "/tmp");
                else snprintf(ach, sizeof(ach), "%s", p);
                bDiag = 1;
            }
            else {
                if (i != j) argv[j] = argv[i];
                ++j;
            }
        }
        argc = j;
        argv[argc] = nullptr;
    }
    if (!bThreads) {
        if ( (p = getenv("SLURM_CPUS_PER_TASK")) != NULL ) layout.nCores = atoi(p);
        else if ( (p = getenv("OMP_NUM_THREADS")) != NULL ) layout.nCores = atoi(p);
    }
    assert(Cores()>0);

    /* MPI Initialization */
#ifdef USE_ITT
    __itt_string_handle *shMPITask = __itt_string_handle_create("MPI");
    __itt_task_begin(domain, __itt_null, __itt_null, shMPITask);
#endif
    commMDL = MPI_COMM_WORLD;
    rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
    if (rc != MPI_SUCCESS) {
        MPI_Error_string(rc, ach, &i);
        perror(ach);
        MPI_Abort(commMDL, rc);
    }
#ifdef HAVE_SIGNAL_H
    signal(SIGINT, TERM_handler);
#endif
#ifdef MDL_FFTW
    if (Cores()>1) FFTW3(init_threads)();
    FFTW3(mpi_init)();
    if (Cores()>1) FFTW3(plan_with_nthreads)(Cores());
#endif

#ifdef USE_ITT
    __itt_task_end(domain);
#endif
    MPI_Comm_size(commMDL, &layout.nProcs);
    MPI_Comm_rank(commMDL, &layout.iProc);

    /* Dedicate one of the threads for MPI, unless it would be senseless to do so */
    if (bDedicated == -1) {
        if (Procs()>0 && Cores()>3) bDedicated = 1;
        else bDedicated = 0;
    }
    if (bDedicated == 1) {
#ifndef USE_HWLOC
        /*if (Cores()==1 || Procs()==1) bDedicated = 0;
          else*/ --layout.nCores;
#else
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
        set_proc = hwloc_bitmap_alloc();
        set_thread = hwloc_bitmap_alloc();
        n = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
        if (n != Cores()) n = 1;
        else if (hwloc_get_cpubind(topology, set_proc, HWLOC_CPUBIND_PROCESS) != 0) n = 1;
        else {
            // Get the first core and reserve it for our MDL use
            t = hwloc_get_obj_inside_cpuset_by_type(topology, set_proc, HWLOC_OBJ_CORE, 0);
            if (t != NULL) {
                hwloc_bitmap_andnot(set_proc, set_proc, t->cpuset);
                n = hwloc_bitmap_weight(t->cpuset);
            }
            else n = 1;
        }
        layout.nCores -= n;
#endif
    }

    /* Construct the thread / processor map */
    iProcToThread.resize(Procs()+1);
    iProcToThread[0] = 0;
    MPI_Allgather(&layout.nCores, 1, MPI_INT32_T, &iProcToThread.front() + 1, 1, MPI_INT, commMDL);
    for (i = 1; i < Procs(); ++i) iProcToThread[i + 1] += iProcToThread[i];
    layout.nThreads = iProcToThread[Procs()];
    layout.idSelf = iProcToThread[Proc()];

    iCoreMPI = bDedicated ? -1 : 0;
    layout.iCore = iCoreMPI;

    /* We have an extra MDL context in case we want a dedicated MPI thread */
    pmdl = new mdlClass *[Cores()+1];
    assert(pmdl != NULL);
    pmdl++;
    pmdl[iCoreMPI] = this;

    /* Allocate the other MDL structures for any threads. */
    for (i = iCoreMPI + 1; i < Cores(); ++i) {
        pmdl[i] = new mdlClass(this, i);
    }

    /* All GPU work is funneled through the MPI thread */
#ifdef USE_CUDA
    cuda.initialize();
#endif
#ifdef USE_METAL
    metal.initialize();
#endif

    iRequestTarget = 0;

#ifndef DEBUG_COUNT_CACHE
    assert(sizeof(CacheHeader) == 16); /* Well, should be a multiple of 8 at least. */
#endif
    nOpenCaches = 0;
    iReplyBufSize = sizeof(CacheHeader) + MDL_CACHE_DATA_SIZE;
    iCacheBufSize = sizeof(CacheHeader) + MDL_FLUSH_DATA_SIZE;
    msgCacheReceive = std::make_unique<mdlMessageCacheReceive>(iCacheBufSize);

    n = Procs();
    if (n > 256) n = 256;
    else if (n < 64) n = 64;

    flushBuffersByRank.resize(Procs(), flushHeadBusy.end()); // For flushing to remote processes / nodes
    flushBuffersByCore.resize(Cores(), NULL); // For flushing to threads on this processor

    for (i = 0; i < 2 * Cores(); ++i) {
        localFlushBuffers.enqueue(new mdlMessageFlushToCore());
    }

    flushBuffCount = n;
    for (i = 0; i < n; ++i) {
        flushHeadFree.push_back(new mdlMessageFlushToRank());
    }

    /* Some bookeeping for the send / recv - 1 of each per thread */
    SendReceiveRequests.reserve(Cores()*2);
    SendReceiveMessages.reserve(Cores()*2);
    SendReceiveStatuses.resize(Cores()*2);
    SendReceiveIndices.resize(Cores()*2);
    CacheRequestMessages.resize(Cores(), nullptr); // Each core can have ONE request
    CacheRequestRendezvous.resize(Cores(), false);
#ifndef NDEBUG
    nRequestsCreated = nRequestsReaped = 0;
#endif

    /* Ring buffer of requests */
    iRequestTarget = 0;
    pRequestTargets.resize(2 * log2(1.0 * Threads()) + Cores(), -1);

    /* Make sure that MPI supports enough tags */
    rc = MPI_Comm_get_attr(commMDL, MPI_TAG_UB, &piTagUB, &flag);
    if (rc == MPI_SUCCESS && flag) {
        assert(Cores()*MDL_TAG_THREAD_OFFSET < *piTagUB);
        assert(pRequestTargets.size()*MDL_TAG_THREAD_OFFSET < *piTagUB);
    }

#ifdef USE_ITT
    __itt_thread_set_name("MPI");
#endif

    threadid.resize(Cores());

    /* Launch threads: if dedicated MPI thread then launch all worker threads. */
#ifdef USE_ITT
    __itt_string_handle *shPthreadTask = __itt_string_handle_create("pthread");
    __itt_task_begin(domain, __itt_null, __itt_null, shPthreadTask);
#endif
    nActiveCores = 0;
    if (Cores() > 1 || bDedicated) {
#ifdef USE_HWLOC
        int icpu = -1;
#endif
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
        pthread_barrier_init(&barrier, NULL, Cores()+(bDedicated?1:0));
        for (i = iCoreMPI + 1; i < Cores(); ++i) {
            pthread_create(&threadid[i], &attr,
                           mdlWorkerThread,
                           pmdl[i]);
#ifdef USE_HWLOC
            if (t) { // If requested, bind the new thread to the specific core
                icpu = hwloc_bitmap_next(set_proc, icpu);
                hwloc_bitmap_only(set_thread, icpu);
                hwloc_set_thread_cpubind(topology, threadid[i], set_thread, HWLOC_CPUBIND_THREAD);
            }
#endif
            ++nActiveCores;
        }
        pthread_attr_destroy(&attr);
    }
#ifdef USE_ITT
    __itt_task_end(domain);
    __itt_task_end(domain);
#endif
#ifdef USE_HWLOC
    if (bDedicated) {
        hwloc_bitmap_free(set_thread);
        hwloc_bitmap_free(set_proc);
        hwloc_topology_destroy(topology);
    }
#endif
#ifdef USE_BT
    register_backtrace();
#endif
    if (!bDedicated) {
        delete queueReceive.release();
        queueReceive = std::make_unique<mdlMessageQueue[]>(Cores());
        worker_ctx = (*fcnWorkerInit)(static_cast<MDL>(static_cast<mdlClass *>(this)));
        pthread_setspecific(worker_key, worker_ctx);
        CommitServices();
        if (Self()) Handler();
        else exit_code = run_master();
        (*fcnWorkerDone)(static_cast<MDL>(static_cast<mdlClass *>(this)), worker_ctx);
    }
    drainMPI();

    if (Cores() > 1 || bDedicated) {
        pthread_barrier_destroy(&barrier);
    }
    for (i = iCoreMPI + 1; i < Cores(); ++i) {
        int *result = NULL;
        pthread_join(threadid[i], reinterpret_cast<void **>(&result));
        if (result) exit_code = *result;
        delete pmdl[i];
    }
    delete[] (pmdl - 1);

    // Deallocate cache reply buffers
    for (auto pReply : freeCacheReplies) { delete pReply; }
    freeCacheReplies.clear();

    while (!localFlushBuffers.empty())
        delete &static_cast<mdlMessageFlushToCore &>(localFlushBuffers.dequeue());
    for (auto i : flushHeadFree) delete i;

#ifdef MDL_FFTW
    // Cleanup for FFTW
    for (auto &plan : fft_plans) {
        auto &info = plan.second;
        FFTW3(destroy_plan)(info.fplan);
        FFTW3(destroy_plan)(info.iplan);
    }
    fft_plans.clear();
    if (Cores()>1) FFTW3(cleanup_threads)();
    FFTW3(cleanup)();
#endif

    MPI_Barrier(commMDL);
    MPI_Finalize();
    return exit_code;
}

#ifdef USE_METAL
void mpiClass::enqueue(const metal::metalMessage &M, basicQueue &replyTo) {
    metal.enqueue(M, replyTo);
}
#endif

void mpiClass::enqueue(mdlMessage &M) {
    queueMPInew.enqueue(M);
}

void mpiClass::enqueue(const mdlMessage &M, basicQueue &replyTo, bool bWait) {
    queueMPInew.enqueue(M, replyTo);
    if (bWait) waitQueue(replyTo);
}

void mdlAbort(MDL mdl) {
    abort();
}

mdlClass::~mdlClass() {
    while (!coreFlushBuffers.empty())
        delete &static_cast<mdlMessageFlushFromCore &>(coreFlushBuffers.dequeue());
    delete coreFlushBuffer;
    // Close Diagnostic file.
    if (bDiag) fclose(fpDiag);
}

/*****************************************************************************\
\*****************************************************************************/

// This routine is overridden for the MPI thread.
int mdlClass::checkMPI() { return 0; }

/* Accept pending combine requests, and call the combine function for each. */
void mdlClass::combine_all_incoming() {
    if (Core()<0) return; /* Dedicated MPI thread combines nothing */
    while (!wqCacheFlush.empty()) {
        mdlMessage &M = wqCacheFlush.dequeue();
        M.result(this);
    }
}

void mdlClass::bookkeeping() {
    combine_all_incoming();
}

extern "C" void mdlCompleteAllWork(MDL mdl) { static_cast<mdlClass *>(mdl)->CompleteAllWork(); }
void mdlClass::CompleteAllWork() {
    while (gpu.flushCompleted()) {} // Wait for any / all GPU operations to return / completed
}

basicMessage &mdlClass::waitQueue(basicQueue &wait) {
    while (wait.empty()) {
        checkMPI(); // Only does something on the MPI thread
        bookkeeping();
        yield();
    }
    return wait.dequeue();
}

#ifdef USE_METAL
void mdlClass::enqueue(const metal::metalMessage &M) {
    mpi->enqueue(M, gpu);
}

void mdlClass::enqueue(const metal::metalMessage &M, basicQueue &replyTo) {
    mpi->enqueue(M, replyTo);
}

// Send the message to the MPI thread and wait for the response
void mdlClass::enqueueAndWait(const metal::metalMessage &M) {
    metal::metalMessageQueue wait;
    enqueue(M, wait);
    waitQueue(wait);
}
#endif

void mdlClass::enqueue(mdlMessage &M) {
    mpi->enqueue(M);
}

void mdlClass::enqueue(const mdlMessage &M, basicQueue &replyTo, bool bWait) {
    mpi->enqueue(M, replyTo, false);
    if (bWait) waitQueue(replyTo);
}

// Send the message to the MPI thread and wait for the response
void mdlClass::enqueueAndWait(const mdlMessage &M) {
    mdlMessageQueue wait;
    enqueue(M, wait, true);
}

/* Synchronize threads */
extern "C" void mdlThreadBarrier(MDL mdl) { static_cast<mdlClass *>(mdl)->ThreadBarrier(); }
int mdlClass::ThreadBarrier(bool bGlobal, int iVote) {
    mdlMessageVote barrier(iVote);
    int i;

    if (Core()) {
        // Send barrier message to thread 0 and wait for it back
        pmdl[0]->threadBarrierQueue.enqueue(barrier, threadBarrierQueue);
        waitQueue(threadBarrierQueue);
        iVote = barrier.vote();
    }
    else {
        mdlMessageQueue pending;
        // Wait for all of the above messages, then send them back
        for (i = 1; i < Cores(); ++i) {
            mdlMessageVote &M = dynamic_cast<mdlMessageVote &>(waitQueue(threadBarrierQueue));
            iVote += M.vote();
            pending.enqueue(M);
        }
        if (bGlobal && Procs()>1) enqueueAndWait(mdlMessageBarrierMPI());
        for (i = 1; i < Cores(); ++i) {
            mdlMessageVote &M = dynamic_cast<mdlMessageVote &>(waitQueue(pending));
            M.vote(iVote);
            M.sendBack();
        }
    }
    return iVote;
}

void mdlClass::mdl_start_MPI_Ssend(mdlMessageSend &M, mdlMessageQueue &replyTo) {
    int iCore = M.target - mpi->Self();
    int bOnNode = (iCore >= 0 && iCore < Cores());
    if (bOnNode) pmdl[iCore]->queueReceive[Core()].enqueue(M, replyTo);
    else enqueue(M, replyTo);
}

void mdlClass::mdl_MPI_Ssend(void *buf, int count, int dest, int tag) {
    mdlMessageQueue wait;
    mdlMessageSend send(buf, count, dest, tag);
    mdl_start_MPI_Ssend(send, wait);
    waitQueue(wait);
}

/*
**
*/
int mdlClass::mdl_MPI_Recv(void *buf, int count, int source, int tag, int *nBytes) {
    int iCore = source - mpi->Self();
    int bOnNode = (iCore >= 0 && iCore < Cores());

    assert(source != Self());

    /* If the sender is on - node, we just wait for the send to appear and make a copy */
    if (bOnNode) {
        mdlMessageSend &M = dynamic_cast<mdlMessageSend &>(waitQueue(queueReceive[iCore]));
        assert(M.tag == tag);
        memcpy(buf, M.buf, M.count);
        *nBytes = M.count;
        M.sendBack();
    }

    /* Off - node: We ask the MPI thread to post a receive for us. */
    else {
        mdlMessageReceive receive(buf, count, source, tag, Core());
        enqueueAndWait(receive);
        *nBytes = receive.getCount();
        return MPI_SUCCESS;
    }
    return MPI_SUCCESS;
}

/*
** Here we do a bidirectional message exchange between two threads.
*/
int mdlClass::mdl_MPI_Sendrecv(
    void *sendbuf, int sendcount,
    int dest, int sendtag, void *recvbuf, int recvcount,
    int source, int recvtag, int *nReceived) {
    mdlMessageQueue wait;
    mdlMessageSend send(sendbuf, sendcount, dest, sendtag);
    mdl_start_MPI_Ssend(send, wait);
    mdl_MPI_Recv(recvbuf, recvcount, source, recvtag, nReceived);
    waitQueue(wait);

    return MPI_SUCCESS;
}

void mdlClass::init(bool bDiag) {
    /*
    ** Set default "maximums" for structures. These are NOT hard
    ** maximums, as the structures will be realloc'd when these
    ** values are exceeded.
    */
    nMaxSrvBytes = -1;
    /*
    ** Allocate service buffers.
    */
    /*
     ** Allocate swapping transfer buffer. This buffer remains fixed.
     */
    pszTrans.resize(MDL_TRANS_SIZE);
    /*
    ** Allocate initial cache spaces.
    */
    cacheSize = mdl_cache_size;
    cache.reserve(MDL_DEFAULT_CACHEIDS);
    while (cache.size() < MDL_DEFAULT_CACHEIDS)
        cache.emplace_back(std::make_unique<CACHE>(this, cache.size()));
    nFlushOutBytes = 0;

    for (int i = 0; i < 9; ++i) coreFlushBuffers.enqueue(new mdlMessageFlushFromCore);
    coreFlushBuffer = new mdlMessageFlushFromCore;

    queueReceive = std::make_unique<mdlMessageQueue[]>(Cores());

    this->bDiag = bDiag;
    if (bDiag) {
        char achDiag[256], ach[256];
        const char *tmp = strrchr(argv[0], '/');
        if (!tmp) tmp = argv[0];
        else ++tmp;
        if (snprintf(achDiag, sizeof(achDiag), "%s/%s.%d", ach, tmp, Self())>=sizeof(achDiag)) abort();
        fpDiag = fopen(achDiag, "w");
        assert(fpDiag != NULL);
    }
}

mdlClass::mdlClass(class mpiClass *mdl, int iMDL)
    : mdlBASE(mdl->argc, mdl->argv), mpi(mdl) {
    layout.iCore = iMDL;
    layout.idSelf = mdl->Self() + iMDL;
    layout.nThreads = mdl->Threads();
    layout.nCores = mdl->Cores();
    layout.nProcs = mdl->layout.nProcs;
    layout.iProc = mdl->layout.iProc;
    iProcToThread = mdl->iProcToThread;

    iCoreMPI = mdl->iCoreMPI;
    pmdl = mdl->pmdl;

    fcnWorkerInit = mdl->fcnWorkerInit;
    fcnWorkerDone = mdl->fcnWorkerDone;
    fcnMaster = mdl->fcnMaster;
    init();
}

mdlClass::mdlClass(class mpiClass *mdl, int (*fcnMaster)(MDL, void *), void *(*fcnWorkerInit)(MDL), void (*fcnWorkerDone)(MDL, void *), int argc, char **argv)
    : mdlBASE(argc, argv), mpi(mdl), fcnWorkerInit(fcnWorkerInit), fcnWorkerDone(fcnWorkerDone), fcnMaster(fcnMaster) {
    init();
}

void mdlClass::drainMPI() {
    while (checkMPI()) {
        yield();
    }
}

int mdlClass::run_master() {
    auto rc = (*fcnMaster)(static_cast<MDL>(this), worker_ctx);
    int id;
    for (id = 1; id < Threads(); ++id) {
        int rID = ReqService(id, SRV_STOP, NULL, 0);
        GetReply(rID);
    }
    return rc;
}

void *mdlClass::WorkerThread() {
    void *result = NULL;
#ifdef USE_ITT
    char szName[20];
    sprintf(szName, "ID %d", Core());
    __itt_thread_set_name(szName);
#endif
    pthread_setspecific(mdl_key, this);
#ifdef USE_BT
    register_backtrace();
#endif
    worker_ctx = (*fcnWorkerInit)(static_cast<MDL>(this));
    pthread_setspecific(worker_key, worker_ctx);
    CommitServices();
    if (Self()) Handler();
    else {
        exit_code = run_master();
        result = &exit_code;
    }
    (*fcnWorkerDone)(static_cast<MDL>(this), worker_ctx);

    if (Core() != iCoreMPI) {
        enqueueAndWait(mdlMessageSTOP());
    }
    else {
        drainMPI();
    }
    return result;
}
void *mdlClass::mdlWorkerThread(void *vmdl) {
    mdlClass *mdl = static_cast<mdlClass *>(vmdl);
    return mdl->WorkerThread();
}

/*
** This function will transfer a block of data using a pack function.
** The corresponding node must call mdlRecv.
*/

#define SEND_BUFFER_SIZE (1 * 1024 * 1024)

extern "C" void mdlSend(MDL mdl, int id, mdlPack pack, void *ctx) { static_cast<mdlClass *>(mdl)->Send(id, pack, ctx); }
void mdlClass::Send(int id, mdlPack pack, void *ctx) {
    size_t nBuff;
    char *vOut;

    vOut = new char[SEND_BUFFER_SIZE];

    do {
        nBuff = (*pack)(ctx, &id, SEND_BUFFER_SIZE, vOut);
        mdl_MPI_Ssend(vOut, nBuff, id, MDL_TAG_SEND);
    } while (nBuff != 0);
    delete[] vOut;
}

extern "C" void mdlRecv(MDL mdl, int id, mdlPack unpack, void *ctx) { static_cast<mdlClass *>(mdl)->Recv(id, unpack, ctx); }
void mdlClass::Recv(int id, mdlPack unpack, void *ctx) {
    char *vIn;
    size_t nUnpack;
    int nBytes;
    int inid;

    if (id < 0) id = MPI_ANY_SOURCE;

    vIn = new char[SEND_BUFFER_SIZE];

    do {
        mdl_MPI_Recv(vIn, SEND_BUFFER_SIZE, id, MDL_TAG_SEND, &nBytes);
        inid = id; //status.MPI_SOURCE;
        nUnpack = (*unpack)(ctx, &inid, nBytes, vIn);
    } while (nUnpack > 0 && nBytes > 0);

    delete[] vIn;
}

/*
 ** This is a tricky function. It initiates a bilateral transfer between
 ** two threads. Both threads MUST be expecting this transfer. The transfer
 ** occurs between idSelf <---> 'id' or 'id' <---> idSelf as seen from the
 ** opposing thread. It is designed as a high performance non-local memory
 ** swapping primitive and implementation will vary in non-trivial ways
 ** between differing architectures and parallel paradigms (eg. message
 ** passing and shared address space). A buffer is specified by 'pszBuf'
 ** which is 'nBufBytes' in size. Of this buffer the LAST 'nOutBytes' are
 ** transfered to the opponent, in turn, the opponent thread transfers his
 ** nBufBytes to this thread's buffer starting at 'pszBuf'.
 ** If the transfer completes with no problems the function returns 1.
 ** If the function returns 0 then one of the players has not received all
 ** of the others memory, however he will have successfully transfered all
 ** of his memory.
 */
extern "C" int mdlSwap(MDL mdl, int id, size_t nBufBytes, void *vBuf, size_t nOutBytes, size_t *pnSndBytes, size_t *pnRcvBytes) {
    return static_cast<mdlClass *>(mdl)->Swap(id, nBufBytes, vBuf, nOutBytes, pnSndBytes, pnRcvBytes);
}
int mdlClass::Swap(int id, size_t nBufBytes, void *vBuf, size_t nOutBytes, size_t *pnSndBytes, size_t *pnRcvBytes) {
    size_t nInBytes, nOutBufBytes;
    int nInMax, nOutMax, nBytes;
    char *pszBuf = CAST(char *, vBuf);
    char *pszIn, *pszOut;
    struct swapInit {
        size_t nOutBytes;
        size_t nBufBytes;
    } swi, swo;

    *pnRcvBytes = 0;
    *pnSndBytes = 0;
    /*
     ** Send number of rejects to target thread amount of free space
     */
    swi.nOutBytes = nOutBytes;
    swi.nBufBytes = nBufBytes;
    mdl_MPI_Sendrecv(&swi, sizeof(swi), id, MDL_TAG_SWAPINIT,
                     &swo, sizeof(swo), id, MDL_TAG_SWAPINIT, &nBytes);
    assert(nBytes == sizeof(swo));
    nInBytes = swo.nOutBytes;
    nOutBufBytes = swo.nBufBytes;
    /*
     ** Start bilateral transfers. Note: One processor is GUARANTEED to
     ** complete all its transfers.
     */
    assert(nBufBytes >= nOutBytes);
    pszOut = &pszBuf[nBufBytes - nOutBytes];
    pszIn = pszBuf;
    while (nOutBytes && nInBytes) {
        /*
         ** nOutMax is the maximum number of bytes allowed to be sent
         ** nInMax is the number of bytes which will be received.
         */
        nOutMax = size_t_to_int((nOutBytes < MDL_TRANS_SIZE)?nOutBytes:MDL_TRANS_SIZE);
        nOutMax = size_t_to_int((nOutMax < nOutBufBytes)?nOutMax:nOutBufBytes);
        nInMax = size_t_to_int((nInBytes < MDL_TRANS_SIZE)?nInBytes:MDL_TRANS_SIZE);
        nInMax = size_t_to_int((nInMax < nBufBytes)?nInMax:nBufBytes);
        /*
         ** Copy to a temp buffer to be safe.
         */
        memcpy(&pszTrans.front(), pszOut, nOutMax);
        mdl_MPI_Sendrecv(&pszTrans.front(), nOutMax, id, MDL_TAG_SWAP,
                         pszIn, nInMax, id, MDL_TAG_SWAP, &nBytes);
        assert(nBytes == nInMax);
        /*
         ** Adjust pointers and counts for next itteration.
         */
        pszOut = &pszOut[nOutMax];
        nOutBytes -= nOutMax;
        nOutBufBytes -= nOutMax;
        *pnSndBytes += nOutMax;
        pszIn = &pszIn[nInMax];
        nInBytes -= nInMax;
        nBufBytes -= nInMax;
        *pnRcvBytes += nInMax;
    }

    /*
     ** At this stage we perform only unilateral transfers, and here we
     ** could exceed the opponent's storage capacity.
     ** Note: use of Ssend is mandatory here, also because of this we
     ** don't need to use the intermediate buffer mdl->pszTrans.
     */
    while (nOutBytes && nOutBufBytes) {
        nOutMax = size_t_to_int((nOutBytes < MDL_TRANS_SIZE)?nOutBytes:MDL_TRANS_SIZE);
        nOutMax = size_t_to_int((nOutMax < nOutBufBytes)?nOutMax:nOutBufBytes);
        mdl_MPI_Ssend(pszOut, nOutMax, id, MDL_TAG_SWAP);
        //ON - NODE not handled:enqueueAndWait(mdlMessageSend(pszOut, nOutMax, MPI_BYTE, id, MDL_TAG_SWAP));
        pszOut = &pszOut[nOutMax];
        nOutBytes -= nOutMax;
        nOutBufBytes -= nOutMax;
        *pnSndBytes += nOutMax;
    }
    while (nInBytes && nBufBytes) {
        nInMax = size_t_to_int((nInBytes < MDL_TRANS_SIZE)?nInBytes:MDL_TRANS_SIZE);
        nInMax = size_t_to_int((nInMax < nBufBytes)?nInMax:nBufBytes);
        mdl_MPI_Recv(pszIn, nInMax, id, MDL_TAG_SWAP, &nBytes);
        assert(nBytes == nInMax);
        pszIn = &pszIn[nInMax];
        nInBytes -= nInMax;
        nBufBytes -= nInMax;
        *pnRcvBytes += nInMax;
    }
    if (nOutBytes) return (0);
    else if (nInBytes) return (0);
    else return (1);
}

void mdlClass::CommitServices() {
    int nMaxBytes;
    nMaxBytes = (nMaxInBytes > nMaxOutBytes) ? nMaxInBytes : nMaxOutBytes;
    if (nMaxBytes > nMaxSrvBytes) {
        nMaxSrvBytes = nMaxBytes;
        input_buffer.resize(nMaxSrvBytes + sizeof(SRVHEAD));
    }
    /* We need a thread barrier here because we share these buffers */
    ThreadBarrier();
}

void mdlAddService(MDL cmdl, int sid, void *p1,
                   fcnService_t *fcnService,
                   int nInBytes, int nOutBytes) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    mdl->AddService(sid, p1, fcnService, nInBytes, nOutBytes);
}

extern "C" int mdlReqService(MDL mdl, int id, int sid, void *vin, int nInBytes) { return static_cast<mdlClass *>(mdl)->ReqService(id, sid, vin, nInBytes); }
int mdlClass::ReqService(int id, int sid, void *vin, int nInBytes) {
    mdlMessageSendRequest request(Self(), sid, id, vin, nInBytes);
    enqueueAndWait(request);
    return request.header.replyTag;
}

extern "C" void mdlGetReply(MDL mdl, int rID, void *vout, int *pnOutBytes) {
    auto nOutBytes = static_cast<mdlClass *>(mdl)->GetReply(rID, vout);
    if (pnOutBytes) *pnOutBytes = nOutBytes;
}
int mdlClass::GetReply(int rID, void *vout) {
    mdlMessageReceiveReply receive(vout, nMaxSrvBytes, rID, Core());
    enqueueAndWait(receive);
    return receive.getCount();
}

void mdlClass::Handler() {
    SRVHEAD *phi = (SRVHEAD *)(&input_buffer.front());
    char *pszIn = (char *)(phi + 1);
    int sid, id, nOutBytes, nBytes;
    mdlMessageSendReply reply(nMaxSrvBytes);

    do {
        /* We ALWAYS use MPI to send requests. */
        mdlMessageReceive receive(phi, nMaxSrvBytes + sizeof(SRVHEAD), MPI_ANY_SOURCE, MDL_TAG_REQ, Core());
        enqueueAndWait(receive);
        nBytes = receive.getCount();
        assert(nBytes == phi->nInBytes + sizeof(SRVHEAD));
        id = phi->idFrom;
        sid = phi->sid;

        nOutBytes = RunService(sid, phi->nInBytes, pszIn, &reply.Buffer.front());
        enqueueAndWait(reply.makeReply(Self(), phi->replyTag, sid, id, nOutBytes));
    } while (sid != SRV_STOP);
}

/*
 ** Special MDL memory allocation functions for allocating memory
 ** which must be visible to other processors thru the MDL cache
 ** functions.
 ** mdlMalloc() is defined to return a pointer to AT LEAST iSize bytes
 ** of memory. This pointer will be passed to either mdlROcache or
 ** mdlCOcache as the pData parameter.
 ** For PVM and most machines these functions are trivial, but on the
 ** T3D and perhaps some future machines these functions are required.
 */
void *mdlMalloc(MDL mdl, size_t iSize) {
    /*    void *ptr;
        int rc = MPI_Alloc_mem(iSize, MPI_INFO_NULL, &ptr);
        if (rc) return NULL;
        return ptr;*/
    return (malloc(iSize));
}

void mdlFree(MDL mdl, void *p) {
    /*    MPI_Free_mem(p);*/
    free(p);
}

void mdlClass::delete_shared_array(void *p, uint64_t nBytes) {
    if (Core()==0) munmap(p, nBytes);
}

uint64_t mdlClass::new_shared_array(void **p, int nSegments, uint64_t *nElements, uint64_t *nBytesPerElement, uint64_t nMinTotalStore) {
    auto gcd = [](int a, int b) {                      // Greatest common divisor
        while (a != 0) {
            auto c = a;
            a = b % a;
            b = c;
        }
        return b;
    };
    uint64_t nPageSize = sysconf(_SC_PAGESIZE);
    auto align = [&gcd, nPageSize](auto &N, auto b) {     // Align elements to a page size boundary
        auto n = nPageSize / gcd(nPageSize, b);
        N += n - 1;
        N -= N % n;
        assert(N * b % nPageSize == 0);
        return N * b;
    };

    // Align each segment to a page boundary
    uint64_t nBytes[nSegments];
    nBytesSegment = nBytes;
    pSegments = p;
    nMessageData = 0;
    for (auto i = 0; i < nSegments; ++i) {
        p[i] = nullptr;
        nMessageData += (nBytesSegment[i] = align(nElements[i], nBytesPerElement[i]));
    }
    ThreadBarrier();
    uint64_t nTotalBytes = 0;
    for (auto t = 0; t < Cores(); ++t)  {
        assert(pmdl[t]->nMessageData == nMessageData);
        nTotalBytes += pmdl[t]->nMessageData;
    }
    // If we don't have enough storage, then we need to increase the number of elements in the last segment
    if (nTotalBytes < nMinTotalStore) {
        ThreadBarrier();
        auto nExtra = nMinTotalStore - nTotalBytes;
        auto nPerThread = ((nExtra + Cores() - 1) / Cores() + nBytesPerElement[nSegments - 1] - 1) / nBytesPerElement[nSegments - 1];
        nElements[nSegments - 1] += nPerThread;
        nMessageData = 0;
        for (auto i = 0; i < nSegments; ++i) {
            nMessageData += (nBytesSegment[i] = align(nElements[i], nBytesPerElement[i]));
        }
        ThreadBarrier();
    }

    if (Core()==0) {
        nTotalBytes = 0;
        for (auto t = 0; t < Cores(); ++t) nTotalBytes += pmdl[t]->nMessageData;
        assert(nTotalBytes >= nMinTotalStore);
        auto region = mmap(nullptr, nTotalBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region == MAP_FAILED) return 0;   // This likely means that there isn't enough memory.
#ifdef __linux__
        madvise(region, nTotalBytes, MADV_DONTDUMP);
#endif
        auto p = static_cast<char *>(region);
        auto q = p;
        for (auto i = 0; i < nSegments; ++i) {
            for (auto t = 0; t < Cores(); ++t) {
                auto o = pmdl[t];
                o->pSegments[i] = p;
                p += o->nBytesSegment[i];
            }
        }
        assert(p - q == nTotalBytes);
    }
    ThreadBarrier();
#ifdef HAVE_NUMA
    for (auto i = 0; i < nSegments; ++i) {
        numa_setlocal_memory(pSegments[i], nElements[i]*nBytesPerElement[i]);
    }
#endif

    return nTotalBytes;
}

/* This is a "thread collective" call. */
void *mdlSetArray(MDL cmdl, size_t nmemb, size_t size, void *vdata) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    char *data = CAST(char *, vdata);
    mdl->nMessageData = nmemb * size;
    mdl->ThreadBarrier();
    if (mdlCore(mdl)==0) {
        int i;
        for (i = 0; i < mdlCores(mdl); ++i) {
            mdl->pmdl[i]->pvMessageData = data;
            data += mdl->pmdl[i]->nMessageData;
        }
    }
    mdl->ThreadBarrier();
    return mdl->pvMessageData;
}

/*
** New collective form. All cores must call this at the same time.
** Normally "size" is identical on all cores, while nmemb may differ
** but this is not strictly a requirement.
*/
void *mdlMallocArray(MDL cmdl, size_t nmemb, size_t size, size_t minSize) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    char *data;
    size_t iSize;
    mdl->nMessageData = nmemb * size;
    mdl->ThreadBarrier();
    if (mdlCore(mdl)==0) {
        int i;
        iSize = 0;
        for (i = 0; i < mdlCores(mdl); ++i) iSize += mdl->pmdl[i]->nMessageData;
        if (iSize < minSize) iSize = minSize;
        data = CAST(char *, mdlMalloc(cmdl, iSize));
        for (i = 0; i < mdlCores(mdl); ++i) {
            mdl->pmdl[i]->pvMessageData = data;
            data += mdl->pmdl[i]->nMessageData;
        }
    }
    mdl->ThreadBarrier();
    data = CAST(char *, mdl->pvMessageData);
    iSize = nmemb * size;
    if (iSize > 4096) iSize -= 4096; /* Cheesy page hack */
    memset(data, 0, iSize); /* First touch */
    mdl->ThreadBarrier();
    return data;
}
void mdlFreeArray(MDL mdl, void *p) {
    if (mdlCore(mdl)==0) free(p);
}

void mdlSetCacheSize(MDL cmdl, int cacheSize) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    mdl->cacheSize = cacheSize;
}

extern "C" void mdlCacheCheck(MDL mdl) { static_cast<mdlClass *>(mdl)->CacheCheck(); }
void mdlClass::CacheCheck() {
    checkMPI(); // Only does something on the MPI thread
    bookkeeping();
}

int mdlCacheStatus(MDL cmdl, int cid) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    assert(cid >= 0);
    if (cid >= mdl->cache.size()) return false;
    auto c = mdl->cache[cid].get();
    return c->isActive();
}

void mdlRelease(MDL cmdl, int cid, void *p) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    auto c = mdl->cache[cid].get();
    c->release(p);
}

double mdlNumAccess(MDL cmdl, int cid) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    auto c = mdl->cache[cid].get();
    return (c->nAccess);
}


double mdlMissRatio(MDL cmdl, int cid) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    auto c = mdl->cache[cid].get();
    double dAccess = c->nAccess;

    if (dAccess > 0.0) return (c->nMiss / dAccess);
    else return (0.0);
}

/*
** GRID Geometry information.  The basic process is as follows:
** - Initialize: Create a MDLGRID giving the global geometry information (total grid size)
** - SetLocal:   Set the local grid geometry (which slabs are on this processor)
** - GridShare:  Share this information between processors
** - Malloc:     Allocate one or more grid instances
** - Free:       Free the memory for all grid instances
** - Finish:     Free the GRID geometry information.
*/
void mdlGridInitialize(MDL cmdl, MDLGRID *pgrid, int n1, int n2, int n3, int a1) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    MDLGRID grid;
    assert(n1 > 0&&n2 > 0&&n3 > 0);
    assert(n1 <= a1);
    *pgrid = grid = CAST(mdlGridContext *, malloc(sizeof(struct mdlGridContext))); assert(grid != NULL);
    grid->n1 = n1;
    grid->n2 = n2;
    grid->n3 = n3;
    grid->a1 = a1;

    /* This will be shared later (see mdlGridShare) */
    grid->id = CAST(uint32_t *, malloc(sizeof(*grid->id)*(grid->n3)));    assert(grid->id != NULL);
    grid->rs = CAST(uint32_t *, mdlMalloc(cmdl, sizeof(*grid->rs)*mdl->Procs())); assert(grid->rs != NULL);
    grid->rn = CAST(uint32_t *, mdlMalloc(cmdl, sizeof(*grid->rn)*mdl->Procs())); assert(grid->rn != NULL);

    /* The following need to be set to appropriate values still. */
    grid->sSlab = grid->nSlab = 0;
    grid->nLocal = 0;
}

void mdlGridFinish(MDL cmdl, MDLGRID grid) {
    //mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    if (grid->rs) free(grid->rs);
    if (grid->rn) free(grid->rn);
    if (grid->id) free(grid->id);
    free(grid);
}

void mdlGridSetLocal(MDL cmdl, MDLGRID grid, int s, int n, uint64_t nLocal) {
    //mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    assert( s >= 0 && s < grid->n3);
    assert( n >= 0 && s + n <= grid->n3);
    grid->sSlab = s;
    grid->nSlab = n;
    grid->nLocal = nLocal;
}

/*
** Share the local GRID information with other processors by,
**   - finding the starting slab and number of slabs on each processor
**   - building a mapping from slab to processor id.
*/
extern "C" void mdlGridShare(MDL cmdl, MDLGRID grid) {return static_cast<mdlClass *>(cmdl)->GridShare(grid); }
void mdlClass::GridShare(MDLGRID grid) {
    int i, id;

    enqueueAndWait(mdlMessageGridShare(grid));

    /* Calculate on which processor each slab can be found. */
    for (id = 0; id < Procs(); id++ ) {
        for (i = grid->rs[id]; i < grid->rs[id]+grid->rn[id]; i++) grid->id[i] = id;
    }
}

/*
** Retrieve the first and last element for the calling thread.
*/
void mdlGridCoordFirstLast(MDL cmdl, MDLGRID grid, mdlGridCoord *f, mdlGridCoord *l, int bCacheAlign) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    uint64_t nPerCore, nThisCore;
    uint64_t nLocal = (uint64_t)(grid->a1) * grid->n2 * grid->nSlab;

    /* Number on each core with multiples of "a1" elments (complete pencils). */
    /* This needs to change to be complete MDL "cache lines" at some point. */
    uint64_t nAlign;
    if (bCacheAlign) nAlign = 1 << (int)(log2(MDL_CACHE_DATA_SIZE / sizeof(float)));
    else nAlign = grid->a1;

    /*nPerCore = nLocal / mdlCores(mdl) + nAlign - 1;*/
    nPerCore = (nLocal - 1) / mdlCores(mdl) + nAlign;
    nPerCore -= nPerCore % nAlign;

    if ( mdlCore(mdl)*nPerCore >= nLocal) nThisCore = 0;
    else if (mdlCore(mdl) == mdlCores(mdl)-1) nThisCore = nLocal - (mdlCores(mdl)-1)*nPerCore;
    else if ( (1 + mdlCore(mdl))*nPerCore < nLocal) nThisCore = nPerCore;
    else nThisCore = nLocal - mdlCore(mdl)*nPerCore;

    /* Calculate global x, y, z coordinates, and local "i" coordinate. */
    f->II = nPerCore * mdlCore(mdl);
    if (f->II > nLocal) f->II = nLocal;
    l->II = f->II + nThisCore;
    f->i = 0;
    l->i = nThisCore;
    f->z = f->II/(grid->a1 * grid->n2);
    f->y = f->II / grid->a1 - f->z * grid->n2;
    f->x = f->II - grid->a1 * (f->z * grid->n2 + f->y);
    assert(bCacheAlign || f->x == 0); // MDL depends on this at the moment
    f->z += grid->sSlab;
    f->grid = grid;

    l->z = l->II/(grid->a1 * grid->n2);
    l->y = l->II / grid->a1 - l->z * grid->n2;
    l->x = l->II - grid->a1 * (l->z * grid->n2 + l->y);
    assert(bCacheAlign || l->x == 0); // MDL depends on this at the moment
    l->z += grid->sSlab;
    l->grid = grid;
}

/*
** Allocate the local elements.  The size of a single element is
** given and the local GRID information is consulted to determine
** how many to allocate.
*/
void *mdlGridMalloc(MDL mdl, MDLGRID grid, int nEntrySize) {
    return mdlMallocArray(mdl, mdlCore(mdl)?0:grid->nLocal, nEntrySize, 0);
}

void mdlGridFree(MDL mdl, MDLGRID grid, void *p) {
    mdlFree(mdl, p);
}

#ifdef MDL_FFTW

extern "C" size_t mdlFFTlocalCount(MDL cmdl, int n1, int n2, int n3, int *nz, int *sz, int *ny, int *sy) {
    return static_cast<mdlClass *>(cmdl)->FFTlocalCount(n1, n2, n3, nz, sz, ny, sy);
}
size_t mdlClass::FFTlocalCount(int n1, int n2, int n3, int *nz, int *sz, int *ny, int *sy) {
    mdlMessageFFT_Sizes sizes(n1, n2, n3);
    enqueueAndWait(sizes);
    if (nz != NULL) *nz = sizes.nz;
    if (sz != NULL) *sz = sizes.sz;
    if (ny != NULL) *ny = sizes.ny;
    if (sy != NULL) *sy = sizes.sy;
    return sizes.nLocal;
}

extern "C" MDLFFT mdlFFTNodeInitialize(MDL cmdl, int n1, int n2, int n3, int bMeasure, FFTW3(real) *data) {
    return static_cast<mdlClass *>(cmdl)->FFTNodeInitialize(n1, n2, n3, bMeasure, data);
}
MDLFFT mdlClass::FFTNodeInitialize(int n1, int n2, int n3, int bMeasure, FFTW3(real) *data) {
    MDLFFT fft = CAST(mdlFFTContext *, malloc(sizeof(struct mdlFFTContext)));
    assert( fft != NULL);
    assert(Core() == 0);

    mdlMessageFFT_Plans plans(n1, n2, n3);
    enqueueAndWait(plans);

    fft->fplan = plans.fplan;
    fft->iplan = plans.iplan;

    /*
    ** Dimensions of k-space and r-space grid.  Note transposed order.
    ** Note also that the "actual" dimension 1 side of the r-space array
    ** can be (and usually is) larger than "n1" because of the inplace FFT.
    */
    mdlGridInitialize(this, &fft->rgrid, n1, n2, n3, 2*(n1 / 2 + 1));
    mdlGridInitialize(this, &fft->kgrid, n1 / 2 + 1, n3, n2, n1 / 2 + 1);
    mdlGridSetLocal(this, fft->rgrid, plans.sz, plans.nz, plans.nLocal);
    mdlGridSetLocal(this, fft->kgrid, plans.sy, plans.ny, plans.nLocal / 2);
    mdlGridShare(this, fft->rgrid);
    mdlGridShare(this, fft->kgrid);
    return fft;
}

MDLFFT mdlFFTInitialize(MDL cmdl, int n1, int n2, int n3, int bMeasure, FFTW3(real) *data) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    MDLFFT fft;
    if (mdlCore(mdl) == 0) {
        mdl->pvMessageData = mdlFFTNodeInitialize(cmdl, n1, n2, n3, bMeasure, data);
    }
    mdl->ThreadBarrier(); /* To synchronize pvMessageData */
    fft = CAST(MDLFFT, mdl->pmdl[0]->pvMessageData);
    mdl->ThreadBarrier(); /* In case we reuse pvMessageData */
    return fft;
}

void mdlFFTNodeFinish(MDL cmdl, MDLFFT fft) {
    //mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    mdlGridFinish(cmdl, fft->kgrid);
    mdlGridFinish(cmdl, fft->rgrid);
    free(fft);
}
void mdlFFTFinish(MDL cmdl, MDLFFT fft) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    mdl->ThreadBarrier();
    if (mdlCore(mdl) == 0) {
        mdlFFTNodeFinish(cmdl, fft);
    }
}

FFTW3(real) *mdlFFTMalloc(MDL mdl, MDLFFT fft) {
    return CAST(FFTW3(real) *, mdlGridMalloc(mdl, fft->rgrid, sizeof(FFTW3(real))));
}

void mdlFFTFree(MDL mdl, MDLFFT fft, void *p) {
    mdlGridFree(mdl, fft->rgrid, p);
}

void mdlFFT( MDL cmdl, MDLFFT fft, FFTW3(real) *data ) { static_cast<mdlClass *>(cmdl)->FFT(fft, data); }
void mdlClass::FFT( MDLFFT fft, FFTW3(real) *data ) {
    mdlMessageDFT_R2C trans(fft, data, (FFTW3(complex) *)data);
    ThreadBarrier();
    if (Core() == iCoreMPI) {
        FFTW3(execute_dft_r2c)(fft->fplan, data, (FFTW3(complex) *)(data));
    }
    else if (Core() == 0) {
        // NOTE: we do not receive a "reply" to this message, rather the synchronization
        // is done with a barrier_wait. This is so there is no spinning (in principle)
        // and the FFTW threads can make full use of the CPU.
        enqueue(trans);
    }
    if (Cores()>1) mpi->pthreadBarrierWait();
}

void mdlIFFT( MDL cmdl, MDLFFT fft, FFTW3(complex) *kdata ) { static_cast<mdlClass *>(cmdl)->IFFT(fft, kdata); }
void mdlClass::IFFT( MDLFFT fft, FFTW3(complex) *kdata ) {
    mdlMessageDFT_C2R trans(fft, (FFTW3(real) *)kdata, kdata);
    ThreadBarrier();
    if (Core() == iCoreMPI) {
        FFTW3(execute_dft_c2r)(fft->iplan, kdata, (FFTW3(real) *)(kdata));
    }
    else if (Core() == 0) {
        // NOTE: we do not receive a "reply" to this message, rather the synchronization
        // is done with a barrier_wait. This is so there is no spinning (in principle)
        // and the FFTW threads can make full use of the CPU.
        enqueue(trans);
    }
    if (Cores()>1) mpi->pthreadBarrierWait();
}
#endif

void mdlAlltoallv(MDL cmdl, int dataSize, void *sbuff, int *scount, int *sdisps, void *rbuff, int *rcount, int *rdisps) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    mdl->Alltoallv(dataSize, sbuff, scount, sdisps, rbuff, rcount, rdisps);
}

void mdlClass::Alltoallv(int dataSize, void *sbuff, int *scount, int *sdisps, void *rbuff, int *rcount, int *rdisps) {
    enqueueAndWait(mdlMessageAlltoallv(dataSize, sbuff, scount, sdisps, rbuff, rcount, rdisps));
}
#if defined(USE_CUDA)
void mdlSetCudaBufferSize(MDL cmdl, int inBufSize, int outBufSize) {
//    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
//    if (mdl->inCudaBufSize < inBufSize) mdl->inCudaBufSize = inBufSize;
//    if (mdl->outCudaBufSize < outBufSize) mdl->outCudaBufSize = outBufSize;
}
#endif

int mdlProcToThread(MDL cmdl, int iProc) {
    return static_cast<mdlClass *>(cmdl)->ProcToThread(iProc);
}
int mdlThreadToProc(MDL cmdl, int iThread) {
    return static_cast<mdlClass *>(cmdl)->ThreadToProc(iThread);
}

int mdlThreads(void *mdl) {  return static_cast<mdlClass *>(mdl)->Threads(); }
int mdlSelf(void *mdl)    {  return static_cast<mdlClass *>(mdl)->Self(); }
int mdlCore(void *mdl)    {  return static_cast<mdlClass *>(mdl)->Core(); }
int mdlCores(void *mdl)   {  return static_cast<mdlClass *>(mdl)->Cores(); }
int mdlProc(void *mdl)    {  return static_cast<mdlClass *>(mdl)->Proc(); }
int mdlProcs(void *mdl)   {  return static_cast<mdlClass *>(mdl)->Procs(); }
int mdlGetArgc(void *mdl) {  return static_cast<mdlClass *>(mdl)->argc; }
char **mdlGetArgv(void *mdl) {  return static_cast<mdlClass *>(mdl)->argv; }


void mdlTimeReset(MDL mdl)             {        static_cast<mdlClass *>(mdl)->TimeReset(); }
double mdlTimeComputing(MDL mdl)       { return static_cast<mdlClass *>(mdl)->TimeComputing(); }
double mdlTimeSynchronizing(MDL mdl)   { return static_cast<mdlClass *>(mdl)->TimeSynchronizing(); }
double mdlTimeWaiting(MDL mdl)         { return static_cast<mdlClass *>(mdl)->TimeWaiting(); }

#ifdef _MSC_VER
double mdlWallTime(void *mdl) {
    FILETIME ft;
    uint64_t clock;
    GetSystemTimeAsFileTime(&ft);
    clock = ft.dwHighDateTime;
    clock <<= 32;
    clock |= ft.dwLowDateTime;
    /* clock is in 100 nano - second units */
    return clock / 10000000.0;
}
#else
double mdlWallTime(void *mdl) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec+(tv.tv_usec * 1e-6));
}
#endif

void mdlprintf(MDL cmdl, const char *format, ...) {
    mdlClass *mdl = static_cast<mdlClass *>(cmdl);
    va_list args;
    va_start(args, format);
    mdl->mdl_vprintf(format, args);
    va_end(args);
}

int mdlClass::numGPUs() {return mpi->numGPUs(); }
bool mdlClass::isMetalActive() {return mpi->isMetalActive(); }
bool mdlClass::isCudaActive() {return mpi->isCudaActive(); }
int mdlCudaActive(MDL mdl) {
#ifdef USE_CUDA
    return static_cast<mdlClass *>(mdl)->isCudaActive();
#else
    return 0;
#endif
}

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

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_UNISTD_H
    #include <unistd.h> /* for unlink() */
#endif
#ifdef _MSC_VER
    #define mkdir _mkdir
    #define unlink _unlink
#endif
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <cinttypes>
#include <limits.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#ifdef HAVE_SYS_TIME_H
    #include <sys/time.h>
#endif
#ifdef HAVE_SYS_STAT_H
    #include <sys/stat.h>
#endif
#include <math.h>
#if defined(HAVE_WORDEXP) && defined(HAVE_WORDFREE)
    #include <wordexp.h>
#elif defined(HAVE_GLOB) && defined(HAVE_GLOBFREE)
    #include <glob.h>
#endif
#include <sys/stat.h>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <filesystem>
#ifdef HAVE_SYS_PARAM_H
    #include <sys/param.h> /* for MAXHOSTNAMELEN, if available */
#endif
#include "fmt/format.h"  // This will be part of c++20
#include "fmt/ostream.h"
using namespace fmt::literals; // Gives us ""_a and ""_format literals

#include "master.h"
#include "core/illinois.h"
#include "io/outtype.h"
#include "smooth/smoothfcn.h"
#include "io/fio.h"
#include "SPH/SPHOptions.h"
#define CYTHON_EXTERN_C extern "C++"
#include "modules/checkpoint.h"

#include "core/memory.h"
#include "core/setadd.h"
#include "core/swapall.h"
#include "core/hostname.h"
#include "core/calcroot.h"
#include "core/select.h"
#include "core/fftsizes.h"
#include "io/restore.h"
#include "initlightcone.h"

#ifdef HAVE_ROCKSTAR
#include "analysis/rshalocount.h"
#include "analysis/rshaloloadids.h"
namespace rockstar {
#include "rockstar/io/io_internal.h"
#include "rockstar/halo.h"
}
#endif

#include "analysis/rsloadids.h"
#include "analysis/rssaveids.h"
#include "analysis/rsextract.h"

#include "domains/distribtoptree.h"
#include "domains/distribroot.h"
#include "domains/dumptrees.h"
#include "domains/olddd.h"
#include "domains/reorder.h"
#include "gravity/setsoft.h"
#include "gravity/activerung.h"
#include "gravity/countrungs.h"
#include "gravity/zeronewrung.h"
#ifdef STELLAR_EVOLUTION
    #include "stellarevolution/stellarevolution.h"
#endif
#include "eEOS/eEOS.h"

#define LOCKFILE ".lockfile"    /* for safety lock */
#define STOPFILE "STOP"         /* for user interrupt */
#define CHECKFILE "CHECKPOINT"      /* for user interrupt */

void MSR::msrprintf(const char *Format, ... ) const {
    va_list ap;
    if (bVDetails) {
        va_start(ap, Format);
        vprintf(Format, ap);
        va_end(ap);
    }
}

#ifdef _MSC_VER
double MSR::Time() {
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
double MSR::Time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec+(tv.tv_usec * 1e-6));
}
#endif

void MSR::TimerStart(int iTimer) {
    ti[iTimer].sec = Time();
}

void MSR::TimerStop(int iTimer) {
    ti[iTimer].sec = Time() - ti[iTimer].sec;
    ti[iTimer].acc += ti[iTimer].sec;
}

// Query the timing of the last call, given that TimerStop was called
// for that iTimer
double MSR::TimerGet(int iTimer) {
    return ti[iTimer].sec;
}

double MSR::TimerGetAcc(int iTimer) {
    return ti[iTimer].acc;
}

// The order should be the same than in the enumerate above!
static const char *timer_names[TOTAL_TIMERS] = {
    "Gravity",  "IO", "Tree", "DomainDecom",  "KickOpen", "KickClose",
    "Density", "EndTimeStep",  "Gradient", "Flux", "TimeStep", "Drift", "FoF",
#ifdef FEEDBACK
    "Feedback",
#endif
#ifdef STAR_FORMATION
    "StarForm",
#endif
#ifdef BLACKHOLES
    "BHs",
#endif
#ifdef STELLAR_EVOLUTION
    "Stev",
#endif
    "Others"
};
static_assert(sizeof(timer_names) / sizeof(timer_names[0]) == TOTAL_TIMERS);

void MSR::TimerHeader() {
    std::string achFile(OutName());
    achFile += ".timing";
    auto fpLog = fopen(achFile.c_str(), "a");
    fprintf(fpLog, "# Step");
    for (int iTimer = 0; iTimer < TOTAL_TIMERS; iTimer++) {
        fprintf(fpLog, " %s", timer_names[iTimer] );
    }
    fprintf(fpLog, "\n");
    fclose(fpLog);
}

void MSR::TimerDump(int iStep) {
    std::string achFile(OutName());
    achFile += ".timing";
    auto fpLog = fopen(achFile.c_str(), "a");

    fprintf(fpLog, "%d", iStep);
    for (int iTimer = 0; iTimer < TOTAL_TIMERS; iTimer++) {
        fprintf(fpLog, " %f", TimerGetAcc(iTimer) );
    }
    fprintf(fpLog, "\n");
    fclose(fpLog);
}

void MSR::TimerRestart() {
    for (int iTimer = 0; iTimer < TOTAL_TIMERS; iTimer++) {
        ti[iTimer].acc = 0.0;
        ti[iTimer].sec = 0.0;
    }
}

void MSR::Leader(void) {
    puts("pkdgrav" PACKAGE_VERSION " Joachim Stadel & Doug Potter Sept 2015");
    puts("USAGE: pkdgrav3 [SETTINGS | FLAGS] [SIM_FILE]");
    puts("SIM_FILE: Configuration file of a particular simulation, which");
    puts("          includes desired settings and relevant input and");
    puts("          output files. Settings specified in this file override");
    puts("          the default settings.");
    puts("SETTINGS");
    puts("or FLAGS: Command line settings or flags for a simulation which");
    puts("          will override any defaults and any settings or flags");
    puts("          specified in the SIM_FILE.");
}
void MSR::Trailer(void) {
    puts("(see man page for more information)");
}

void MSR::Exit(int status) {
    exit(status);
}

static void make_directories(std::string name) {
    auto i = name.rfind('/');
    if (i > 0) {
        if (i != std::string::npos) {
            name = name.substr(0, i);
            make_directories(name);
            mkdir(name.c_str(), 0755);
        }
    }
}

std::string MSR::BuildName(std::string_view path, int iStep, const char *type) {
    if (!path.length()) path = "{name}.{step:05d}{type}";
    auto name = fmt::format(path, "name"_a = OutName(), "step"_a = iStep, "type"_a = type);
    make_directories(name);
    return name;
}
std::string MSR::BuildName(int iStep, const char *type) {
    return BuildName(parameters.get_achOutPath(), iStep, type);
}
std::string MSR::BuildIoName(int iStep, const char *type) {
    auto achIoPath = parameters.get_achIoPath();
    if ( achIoPath.length() )
        return BuildName(achIoPath, iStep, type);
    else return BuildName(iStep, type);
}
std::string MSR::BuildCpName(int iStep, const char *type) {
    auto achCheckpointPath = parameters.get_achCheckpointPath();
    if ( achCheckpointPath.length() )
        return BuildName(achCheckpointPath, iStep, type);
    else return BuildName(iStep, type);
}

void MSR::MakePath(std::string_view dir, std::string_view base, char *path) {
    /*
    ** Prepends "dir" to "base" and returns the result in "path". It is the
    ** caller's responsibility to ensure enough memory has been allocated
    ** for "path".
    */

    if (!path) return;
    if (dir.length()) {
        memcpy(path, dir.data(), dir.length());
        path += dir.length();
        *path++ = '/';
    }
    if (base.length()) {
        memcpy(path, base.data(), base.length());
        path += base.length();
    }
    *path = 0;
}

uint64_t MSR::getMemoryModel() {
    uint64_t mMemoryModel = 0;
    /*
    ** Figure out what memory models are in effect.  Configuration flags
    ** can be used to request a specific model, but certain operations
    ** will force these flags to be on.
    */
    if (parameters.get_bFindGroups()) {
        mMemoryModel |= PKD_MODEL_GROUPS | PKD_MODEL_VELOCITY;
        if (parameters.get_bMemGlobalGid()) {
            mMemoryModel |= PKD_MODEL_GLOBALGID;
        }
    }
    if (DoGravity()) {
        mMemoryModel |= PKD_MODEL_VELOCITY | PKD_MODEL_NODE_MOMENT;
        if (!parameters.get_bNewKDK()) mMemoryModel |= PKD_MODEL_ACCELERATION;
    }
    if (parameters.get_bDoDensity())       mMemoryModel |= PKD_MODEL_DENSITY;
    if (parameters.get_bMemIntegerPosition()) mMemoryModel |= PKD_MODEL_INTEGER_POS;
    if (parameters.get_bMemUnordered()&&parameters.get_bNewKDK()) mMemoryModel |= PKD_MODEL_UNORDERED;
    if (parameters.get_bMemParticleID())   mMemoryModel |= PKD_MODEL_PARTICLE_ID;
    if (parameters.get_bMemAcceleration() || parameters.get_bDoAccOutput()) mMemoryModel |= PKD_MODEL_ACCELERATION;
    if (parameters.get_bMemVelocity())     mMemoryModel |= PKD_MODEL_VELOCITY;
    if (parameters.get_bMemPotential() || parameters.get_bDoPotOutput())    mMemoryModel |= PKD_MODEL_POTENTIAL;
    if (parameters.get_bFindHopGroups())   mMemoryModel |= PKD_MODEL_GROUPS | PKD_MODEL_DENSITY | PKD_MODEL_BALL;
    if (parameters.get_bMemGroups())       mMemoryModel |= PKD_MODEL_GROUPS;
    if (parameters.get_bMemMass())         mMemoryModel |= PKD_MODEL_MASS;
    if (parameters.get_bMemSoft())         mMemoryModel |= PKD_MODEL_SOFTENING;
    if (parameters.get_bMemVelSmooth())    mMemoryModel |= PKD_MODEL_VELSMOOTH;

    if (parameters.get_bMemNodeAcceleration()) mMemoryModel |= PKD_MODEL_NODE_ACCEL;
    if (parameters.get_bMemNodeVelocity())     mMemoryModel |= PKD_MODEL_NODE_VEL;
    if (parameters.get_bMemNodeMoment())       mMemoryModel |= PKD_MODEL_NODE_MOMENT;
    if (parameters.get_bMemNodeSphBounds())    mMemoryModel |= PKD_MODEL_NODE_SPHBNDS;

    if (parameters.get_bMemNodeBnd())          mMemoryModel |= PKD_MODEL_NODE_BND;
    if (parameters.get_bMemNodeVBnd())         mMemoryModel |= PKD_MODEL_NODE_VBND;
    if (MeshlessHydro())                       mMemoryModel |= (PKD_MODEL_SPH | PKD_MODEL_ACCELERATION);
#if defined(STAR_FORMATION) || defined(FEEDBACK) || defined(STELLAR_EVOLUTION)
    mMemoryModel |= PKD_MODEL_STAR;
#endif
#if BLACKHOLES
    mMemoryModel |= PKD_MODEL_BALL;
    mMemoryModel |= PKD_MODEL_BH;
#endif

    if (parameters.get_bMemBall())             mMemoryModel |= PKD_MODEL_BALL;

    return mMemoryModel;
}

std::pair < int, int> MSR::InitializePStore(uint64_t *nSpecies, uint64_t mMemoryModel, uint64_t nEphemeral) {
    struct inInitializePStore ps;
    double dStorageAmount = (1.0 + parameters.get_dExtraStore());
    int i;
    for ( i = 0; i <= FIO_SPECIES_LAST; ++i) ps.nSpecies[i] = nSpecies[i];
    ps.nStore = ceil( dStorageAmount * ps.nSpecies[FIO_SPECIES_ALL] / mdlThreads(mdl));
    ps.nTreeBitsLo = parameters.get_nTreeBitsLo();
    ps.nTreeBitsHi = parameters.get_nTreeBitsHi();
    ps.iCacheSize  = parameters.get_iCacheSize();
    ps.iCacheMaxInflight = parameters.get_iCacheMaxInflight();
    ps.iWorkQueueSize  = parameters.get_iWorkQueueSize();
    ps.fPeriod = parameters.get_dPeriod();
    ps.mMemoryModel = mMemoryModel | PKD_MODEL_VELOCITY;

#define SHOW(m) ((ps.mMemoryModel & PKD_MODEL_##m)?" " #m:"")
    printf("Memory Models:%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
           parameters.get_bMemIntegerPosition() ? " INTEGER_POSITION" : " DOUBLE_POSITION",
           SHOW(UNORDERED), SHOW(VELOCITY), SHOW(ACCELERATION), SHOW(POTENTIAL),
           SHOW(GROUPS), SHOW(MASS), SHOW(DENSITY),
           SHOW(BALL), SHOW(SOFTENING), SHOW(VELSMOOTH), SHOW(SPH), SHOW(NEW_SPH),
           SHOW(STAR), SHOW(PARTICLE_ID), SHOW(BH), SHOW(GLOBALGID),
           SHOW(NODE_MOMENT), SHOW(NODE_ACCEL), SHOW(NODE_VEL), SHOW(NODE_SPHBNDS),
           SHOW(NODE_BND), SHOW(NODE_VBND), SHOW(NODE_BOB));
#undef SHOW

    // Calculate the Ephemeris memory requirements
    EphemeralMemory e(nEphemeral);
    if (parameters.get_bFindGroups()    ) e |= EphemeralMemory(4);
    if (parameters.get_bFindHopGroups() ) e |= EphemeralMemory(8);
    if (parameters.get_iPkInterval()    ) e |= EphemeralMemory(4);
    if (parameters.get_bGravStep()      ) e |= EphemeralMemory(8);
    if (DoGas()                         ) e |= EphemeralMemory(8);
    if (parameters.get_bMemBall()       ) e |= EphemeralMemory(8);
    if (parameters.get_bDoDensity()     ) e |= EphemeralMemory(12);
#ifdef BLACKHOLES
    e |= EphemeralMemory(8);
#endif
    // We need one grid to measure P(k); two if we are interlacing
    e |= EphemeralMemory(mdl, parameters.get_nGridPk(), parameters.get_bPkInterlace() ? 2 : 1);
    // Add some ephemeral memory (if needed) for the linGrid. 3 grids are stored : forceX, forceY, forceZ
    e |= EphemeralMemory(mdl, parameters.get_nGridLin(), 3);

    // Calculate constraint for generating initial conditions. We need 10 grids for the initial conditions
    EphemeralMemory ic_memory(mdl, parameters.get_nGrid(), 10);

    ps.nEphemeralBytes = e.per_particle;
    ps.nMinEphemeral = e.per_process;
    ps.nMinTotalStore = ic_memory.per_process;

    // Check all registered Python analysis routines and account for their memory requirements
    for ( msr_analysis_callback &i : analysis_callbacks) {
        auto attr_per_node = PyObject_GetAttrString(i.memory, "bytes_per_node");
        auto attr_per_part = PyObject_GetAttrString(i.memory, "bytes_per_particle");
        auto per_node = PyLong_AsSize_t(attr_per_node);
        auto per_part = PyLong_AsSize_t(attr_per_part);
        if (ps.nEphemeralBytes < per_part) ps.nEphemeralBytes = per_part;
        if (ps.nMinEphemeral < per_node) ps.nMinEphemeral = per_node;
        Py_DECREF(attr_per_node);
        Py_DECREF(attr_per_part);
    }
    outInitializePStore pout;
    pstInitializePStore(pst, &ps, sizeof(ps), &pout, sizeof(pout));
    PKD pkd = pst->plcl->pkd;
    printf("Allocated %lu MB for particle store on each processor.\n",
           pkd->ParticleMemory()/(1024 * 1024));
    printf("Particles: %lu bytes (persistent) + %d bytes (ephemeral), Nodes: %lu bytes\n",
           pkd->particles.ParticleSize(), ps.nEphemeralBytes, pkd->NodeSize());
    if (pkd->particles.ParticleSize() > MDL_CACHE_DATA_SIZE) {
        printf("ERROR! MDL_CACHE_DATA_SIZE (%d bytes) is too small for the given particle size, please increasing it\n", MDL_CACHE_DATA_SIZE);
        abort();
    }
    if (ps.nMinEphemeral)
        printf("Ephemeral will be at least %" PRIu64 " MB per node.\n", ps.nMinEphemeral/(1024 * 1024));
    return std::make_pair(pout.nSizeParticle, pout.nSizeNode);
}

/// @brief Return the number of threads to use for parallel reading / writing
/// @param bParallel True if parallel reading / writing is enabled
/// @param nParallel number of threads to use for parallel reading / writing
/// @return Actual number of threads to use for parallel reading / writing
int64_t MSR::parallel_count(bool bParallel, int64_t nParallel) {
    if (!bParallel) return 1; // Parallel reading / writing is disabled, use 1 thread
    else if (nParallel < 1) return nThreads; // Use maximum number of threads
    else return std::min(nParallel, int64_t(nThreads)); // Use specified number of threads (if less than maximum)
}

/// @brief Return the number of threads to use for parallel reading
/// @return Number of threads to use for parallel reading
int64_t MSR::parallel_read_count() {
    return parallel_count(parameters.get_bParaRead(), parameters.get_nParaRead());
}

/// @brief Return the number of threads to use for parallel writing
/// @return Number of threads to use for parallel writing
int64_t MSR::parallel_write_count() {
    return parallel_count(parameters.get_bParaWrite(), parameters.get_nParaWrite());
}

void MSR::stat_files(std::vector<uint64_t> &counts, const std::string_view &filename_template, uint64_t element_size) {
    ServiceFileSizes::input hdr;

    strncpy(hdr.filename, filename_template.data(), sizeof(hdr.filename));
    hdr.nSimultaneous = hdr.nTotalActive = parallel_read_count();
    hdr.iReaderWriter = 0;
    hdr.nElementSize = 1;

    auto out = new ServiceFileSizes::output[ServiceFileSizes::max_files];
    auto n = mdl->RunService(PST_FILE_SIZES, sizeof(hdr), &hdr, out);
    n /= sizeof(ServiceFileSizes::output);
    counts.resize(n);

    for (auto i = 0; i < n; ++i) {
        assert(out[i].iFileIndex < n);
        counts[out[i].iFileIndex] = out[i].nFileBytes / element_size;
    }
    delete [] out;
}

void MSR::Restore(const std::string &baseName, int nSizeParticle) {
    std::vector<uint64_t> counts;
    std::string filename_template = baseName + ".{i}";
    TimerStart(TIMER_NONE);
    printf("Scanning Checkpoint files...\n");
    stat_files(counts, filename_template, nSizeParticle);
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);
    printf("... identified %" PRIu64 " particles in %d files, Wallclock: %f secs.\n",
           std::accumulate(counts.begin(), counts.end(), uint64_t(0)),
           int(counts.size()), dsec);

    using mdl::ServiceBuffer;
    ServiceBuffer msg {
        ServiceBuffer::Field<ServiceInput::input>(),
        ServiceBuffer::Field<ServiceInput::io_elements>(counts.size())
    };
    auto hdr = static_cast<ServiceInput::input *>(msg.data(0));
    auto elements = static_cast<ServiceInput::io_elements *>(msg.data(1));
    hdr->nFiles = counts.size();
    hdr->nElements = std::accumulate(counts.begin(), counts.end(), uint64_t(0));
    std::copy(counts.begin(), counts.end(), elements);
    hdr->iBeg = 0;
    hdr->iEnd = hdr->nElements;
    strncpy(hdr->io.filename, filename_template.c_str(), sizeof(hdr->io.filename));
    hdr->io.nSimultaneous = parallel_read_count();
    hdr->io.nSegment = hdr->io.iThread = 0; // setup later
    hdr->io.iReaderWriter = 0;
    mdl->RunService(PST_RESTORE, msg);
}

template<>
PyObject *MSR::restore(PyObject *file) {
    auto result = PyObject_CallOneArg(pDill_load, file);
    if (!result) { PyErr_Print(); abort(); }
    return result;
}

// Specialization for int
template<>
int MSR::restore(PyObject *file) {
    auto result = restore < PyObject *>(file);
    auto value = PyLong_AsLong(result);
    Py_DECREF(result);
    return value;
}

// Specialization for double
template<>
double MSR::restore(PyObject *file) {
    auto result = restore < PyObject *>(file);
    auto value = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return value;
}

void MSR::Restart(const char *filename, PyObject *kwargs) {
    auto sec = MSR::Time();

    std::string pkl_filename = filename;
    pkl_filename += ".pkl";

    bVDetails = parameters.get_bVDetails();
    if (parameters.get_bVStart())
        printf("Restoring from checkpoint\n");
    TimerStart(TIMER_IO);

    auto pFile = PyObject_CallFunction(PyDict_GetItemString(PyEval_GetBuiltins(), "open"), "ss", pkl_filename.c_str(), "rb");
    if (!pFile) {
        PyErr_Print();
        abort();
    }

    // // Checkpoint the important variables
    auto version = restore < int>(pFile);
    if (version != 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid checkpoint file version");
        PyErr_Print();
        abort();
    }

    auto species_list = restore < PyObject *>(pFile);
    fioSpeciesList nSpecies;
    for (auto i = 0; i < FIO_SPECIES_LAST; ++i) {
        nSpecies[i] = PyLong_AsUnsignedLongLong(PyList_GetItem(species_list, i));
    }
    this->nDark = nSpecies[FIO_SPECIES_DARK];
    this->nGas  = nSpecies[FIO_SPECIES_SPH];
    this->nStar = nSpecies[FIO_SPECIES_STAR];
    this->nBH   = nSpecies[FIO_SPECIES_BH];
    this->N     = nSpecies[FIO_SPECIES_ALL] = nDark + nGas + nStar + nBH;
    nMaxOrder = N - 1; // iOrder goes from 0 to N - 1

    auto classes_list = restore < PyObject *>(pFile);
    static_assert(PKD_MAX_CLASSES <= 256); // Hopefully nobody will be mean to us (we use the stack)
    PARTCLASS aCheckpointClasses[PKD_MAX_CLASSES];
    auto nCheckpointClasses = PyList_Size(classes_list);
    for (int i = 0; i < nCheckpointClasses; ++i) {
        auto class_list = PyList_GetItem(classes_list, i); // Borrowed reference, no need to Py_DECREF
        auto eSpeciesObj = PyList_GetItem(class_list, 0);
        auto fMassObj = PyList_GetItem(class_list, 1);
        auto fSoftObj = PyList_GetItem(class_list, 2);
        auto iMatObj = PyList_GetItem(class_list, 3);
        aCheckpointClasses[i].eSpecies = FIO_SPECIES(PyLong_AsLong(eSpeciesObj));
        aCheckpointClasses[i].fMass = PyFloat_AsDouble(fMassObj);
        aCheckpointClasses[i].fSoft = PyFloat_AsDouble(fSoftObj);
        aCheckpointClasses[i].iMat = PyLong_AsLong(iMatObj);
    }

    auto iStep = restore < int>(pFile);
    auto nSteps = restore < int>(pFile);
    auto dTime = restore < double>(pFile);
    auto dDelta = restore < double>(pFile);
    this->dEcosmo = restore < double>(pFile);
    this->dUOld = restore < double>(pFile);
    this->dTimeOld = restore < double>(pFile);

    auto arguments = restore < PyObject *>(pFile);
    auto specified = restore < PyObject *>(pFile);
    parameters.merge(pkd_parameters(arguments, specified));
    parameters.update(kwargs, false);
    ValidateParameters(); // Should be okay, but other stuff happens here (cosmo is setup for example)

    // Restore the interpreter state
    PyObject *args = PyTuple_Pack(1, pFile);
    PyObject *result = PyObject_CallObject(pDill_load_module, args);
    Py_DECREF(args);
    if (!result) { PyErr_Print(); abort(); }
    Py_DECREF(result);

    PyObject_CallMethod(pFile, "close", NULL);
    Py_DECREF(pFile);

    uint64_t mMemoryModel = 0;
    mMemoryModel = getMemoryModel();
    if (nGas && !parameters.has_bDoGas()) parameters.set_bDoGas(true);
    if (NewSPH()) mMemoryModel |= (PKD_MODEL_NEW_SPH | PKD_MODEL_ACCELERATION | PKD_MODEL_VELOCITY | PKD_MODEL_DENSITY | PKD_MODEL_BALL | PKD_MODEL_NODE_BOB);
    auto [nSizeParticle, nSizeNode] = InitializePStore(nSpecies, mMemoryModel, parameters.get_nMemEphemeral());

    Restore(filename, nSizeParticle);
    pstSetClasses(pst, aCheckpointClasses, nCheckpointClasses * sizeof(PARTCLASS), NULL, 0);
    CalcBound();
    CountRungs(NULL);

    TimerStop(TIMER_IO);
    auto dsec = TimerGet(TIMER_IO);
    double dExp = csmTime2Exp(csm, dTime);
    if (dsec > 0.0) {
        double rate = N * nSizeParticle / dsec;
        const char *units = "B";
        if (rate > 10000) { rate /= 1024;   units = "KB"; }
        if (rate > 10000) { rate /= 1024;   units = "MB"; }
        if (rate > 10000) { rate /= 1024;   units = "GB"; }
        msrprintf("Checkpoint Restart Complete @ a=%g, Wallclock: %f secs (%.2f %s/s)\n\n", dExp, dsec, rate, units);
    }
    else msrprintf("Checkpoint Restart Complete @ a=%g, Wallclock: %f secs\n\n", dExp, dsec);

    /* We can indicate that the DD was already done at rung 0 */
    iLastRungRT = 0;
    iLastRungDD = 0;

    InitCosmology(csm);

    SetDerivedParameters(true);

    if (parameters.has_dSoft()) SetSoft(Soft());

    if (NewSPH()) {
        /*
        ** Initialize kernel target with either the mean mass or nSmooth
        */
        sec = MSR::Time();
        printf("Initializing Kernel target ...\n");
        {
            SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
            if (SPHoptions.useNumDen) {
                parameters.set_fKernelTarget(parameters.get_nSmooth());
            }
            else {
                double Mtot;
                uint64_t Ntot;
                CalcMtot(&Mtot, &Ntot);
                parameters.set_fKernelTarget(Mtot / Ntot * parameters.get_nSmooth());
            }
        }
        dsec = MSR::Time() - sec;
        printf("Initializing Kernel target complete, Wallclock: %f secs.\n", dsec);
        SetSPHoptions();
        InitializeEOS();
    }
    if (parameters.get_bAddDelete()) GetNParts();
    if (parameters.has_achOutTimes()) {
        nSteps = ReadOuts(dTime);
    }

    if (bAnalysis) return; // Very cheeserific
    Simulate(dTime, dDelta, iStep, nSteps, true);
}

// This is the old style restart, which is not used anymore
// It is kept here for old checkpoint files, but will be removed soon
void MSR::Restart(int n, const char *baseName, int iStep, int nSteps, double dTime, double dDelta,
                  size_t nDark, size_t nGas, size_t nStar, size_t nBH,
                  double dEcosmo, double dUOld, double dTimeOld,
                  std::vector<PARTCLASS> &aClasses, PyObject *arguments, PyObject *specified) {
    auto sec = MSR::Time();

    parameters.merge(pkd_parameters(arguments, specified));

    this->nDark = nDark;
    this->nGas  = nGas;
    this->nStar = nStar;
    this->nBH   = nBH;
    this->N     = nDark + nGas + nStar + nBH;
    this->dEcosmo = dEcosmo;
    this->dUOld   = dUOld;
    this->dTimeOld= dTimeOld;

    if (parameter_overrides) {
        if (!parameters.update(parameter_overrides)) Exit(1);
        parameter_overrides = nullptr; // This is not owned by us
    }

    ValidateParameters(); // Should be okay, but other stuff happens here (cosmo is setup for example)

    bVDetails = parameters.get_bVDetails();
    if (parameters.get_bVStart())
        printf("Restoring from checkpoint\n");
    TimerStart(TIMER_IO);

    nMaxOrder = N - 1; // iOrder goes from 0 to N - 1

    fioSpeciesList nSpecies;
    for ( auto i = 0; i <= FIO_SPECIES_LAST; ++i) nSpecies[i] = 0;
    nSpecies[FIO_SPECIES_ALL]  = N;
    nSpecies[FIO_SPECIES_SPH]  = nGas;
    nSpecies[FIO_SPECIES_DARK] = nDark;
    nSpecies[FIO_SPECIES_STAR] = nStar;
    nSpecies[FIO_SPECIES_BH]   = nBH;
    uint64_t mMemoryModel = 0;
    mMemoryModel = getMemoryModel();
    if (nGas && !parameters.has_bDoGas()) parameters.set_bDoGas(true);
    if (NewSPH()) mMemoryModel |= (PKD_MODEL_NEW_SPH | PKD_MODEL_ACCELERATION | PKD_MODEL_VELOCITY | PKD_MODEL_DENSITY | PKD_MODEL_BALL | PKD_MODEL_NODE_BOB);
    auto [nSizeParticle, nSizeNode] = InitializePStore(nSpecies, mMemoryModel, parameters.get_nMemEphemeral());

    Restore(baseName, nSizeParticle);
    pstSetClasses(pst, aClasses.data(), aClasses.size()*sizeof(PARTCLASS), NULL, 0);
    CalcBound();
    CountRungs(NULL);

    TimerStop(TIMER_IO);
    auto dsec = TimerGet(TIMER_IO);
    double dExp = csmTime2Exp(csm, dTime);
    if (dsec > 0.0) {
        double rate = N * nSizeParticle / dsec;
        const char *units = "B";
        if (rate > 10000) { rate /= 1024;   units = "KB"; }
        if (rate > 10000) { rate /= 1024;   units = "MB"; }
        if (rate > 10000) { rate /= 1024;   units = "GB"; }
        msrprintf("Checkpoint Restart Complete @ a=%g, Wallclock: %f secs (%.2f %s/s)\n\n", dExp, dsec, rate, units);
    }
    else msrprintf("Checkpoint Restart Complete @ a=%g, Wallclock: %f secs\n\n", dExp, dsec);

    /* We can indicate that the DD was already done at rung 0 */
    iLastRungRT = 0;
    iLastRungDD = 0;

    InitCosmology(csm);

    SetDerivedParameters(true);

    if (parameters.has_dSoft()) SetSoft(Soft());

    if (DoGas() && NewSPH()) {
        /*
        ** Initialize kernel target with either the mean mass or nSmooth
        */
        sec = MSR::Time();
        printf("Initializing Kernel target ...\n");
        {
            SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
            if (SPHoptions.useNumDen) {
                parameters.set_fKernelTarget(parameters.get_nSmooth());
            }
            else {
                double Mtot;
                uint64_t Ntot;
                CalcMtot(&Mtot, &Ntot);
                parameters.set_fKernelTarget(Mtot / Ntot * parameters.get_nSmooth());
            }
        }
        dsec = MSR::Time() - sec;
        printf("Initializing Kernel target complete, Wallclock: %f secs.\n", dsec);
        SetSPHoptions();
        InitializeEOS();
    }
    if (parameters.get_bAddDelete()) GetNParts();
    if (parameters.has_achOutTimes()) {
        nSteps = ReadOuts(dTime);
    }

    if (bAnalysis) return; // Very cheeserific
    Simulate(dTime, dDelta, iStep, nSteps, true);
}

void MSR::persist(PyObject *file, PyObject *obj) {
    auto args = PyTuple_Pack(3, obj, file, Py_True);
    auto result = PyObject_CallObject(pDill_dump, args);
    Py_DECREF(args);
    if (!result) { PyErr_Print(); abort(); }
    Py_DECREF(result);
}
void MSR::persist(PyObject *file, int n) {
    auto number = PyLong_FromLongLong(n);
    persist(file, number);
    Py_DECREF(number);
}
void MSR::persist(PyObject *file, double d) {
    auto number = PyFloat_FromDouble(d);
    persist(file, number);
    Py_DECREF(number);
}

void MSR::writeParameters(const std::string &baseName, int iStep, int nSteps, double dTime, double dDelta) {
    fioSpeciesList nSpecies;
    int i;
    int nBytes;

    // ******************************************************************
    // Collect the information to checkpoint
    // ******************************************************************
    static_assert(PKD_MAX_CLASSES <= 256); // Hopefully nobody will be mean to us (we use the stack)
    PARTCLASS aCheckpointClasses[PKD_MAX_CLASSES];
    nBytes = pstGetClasses(pst, NULL, 0, aCheckpointClasses, PKD_MAX_CLASSES * sizeof(PARTCLASS));
    int nCheckpointClasses = nBytes / sizeof(PARTCLASS);
    assert(nCheckpointClasses * sizeof(PARTCLASS)==nBytes);

    for (i = 0; i <= FIO_SPECIES_LAST; ++i) nSpecies[i] = 0;
    nSpecies[FIO_SPECIES_ALL]  = N;
    nSpecies[FIO_SPECIES_SPH]  = nGas;
    nSpecies[FIO_SPECIES_DARK] = nDark;
    nSpecies[FIO_SPECIES_STAR] = nStar;
    nSpecies[FIO_SPECIES_BH]   = nBH;

    // Create a list with a count of each species
    auto species_list = PyList_New(FIO_SPECIES_LAST);
    for (int i = 0; i < FIO_SPECIES_LAST; ++i) {
        PyList_SetItem(species_list, i, PyLong_FromUnsignedLongLong(nSpecies[i])); // PyList_SetItem steals a reference to num
    }

    // Create a list with the checkpoint classes
    auto classes_list = PyList_New(nCheckpointClasses);
    for (int i = 0; i < nCheckpointClasses; ++i) {
        auto class_list = PyList_New(4); // Each inner list has 4 elements

        // Convert structure members to Python objects and add them to the class_list
        PyObject *eSpecies = PyLong_FromLong(aCheckpointClasses[i].eSpecies);
        PyObject *fMass = PyFloat_FromDouble(aCheckpointClasses[i].fMass);
        PyObject *fSoft = PyFloat_FromDouble(aCheckpointClasses[i].fSoft);
        PyObject *iMat = PyLong_FromLong(aCheckpointClasses[i].iMat);

        // Note: PyList_SetItem steals a reference to the item
        PyList_SetItem(class_list, 0, eSpecies);
        PyList_SetItem(class_list, 1, fMass);
        PyList_SetItem(class_list, 2, fSoft);
        PyList_SetItem(class_list, 3, iMat);

        // Add the inner list to the outer list
        PyList_SetItem(classes_list, i, class_list); // This also steals a reference
    }
    auto a = parameters.arguments();
    auto s = parameters.specified();

    // ******************************************************************
    // Write the interpreter state and the checkpoint variables to a file
    // ******************************************************************
    // Write the interpreter and the checkpoint variables to a file
    auto achOutName = baseName + ".pkl";
    auto pFile = PyObject_CallFunction(PyDict_GetItemString(PyEval_GetBuiltins(), "open"), "ss", achOutName.c_str(), "wb");
    if (!pFile) {
        PyErr_Print();
        abort();
    }

    // Checkpoint the important variables
    persist(pFile, 1);            // checkpoint version
    persist(pFile, species_list); // 1: species
    persist(pFile, classes_list); // 1: classes
    persist(pFile, iStep);        // 1: step
    persist(pFile, nSteps);       // 1: steps
    persist(pFile, dTime);        // 1: time
    persist(pFile, dDelta);       // 1: delta
    persist(pFile, dEcosmo);      // 1: E
    persist(pFile, dUOld);        // 1: U
    persist(pFile, dTimeOld);     // 1: Utime
    persist(pFile, a);            // 1: arguments
    persist(pFile, s);            // 1: specified

    // 1: Persist the interpreter state
    PyObject *args = PyTuple_Pack(3, pFile, Py_None, Py_True);
    PyObject *result = PyObject_CallObject(pDill_dump_module, args);
    Py_DECREF(args);
    if (!result) { PyErr_Print(); abort(); }
    Py_DECREF(result);

    PyObject_CallMethod(pFile, "close", NULL);
    Py_DECREF(pFile);

    Py_DECREF(species_list);
    Py_DECREF(classes_list);

    // ******************************************************************
    // Write the restart file
    // ******************************************************************
    std::ofstream restart_file(baseName);
    if (!restart_file) {
        perror(baseName.c_str());
        abort();
    }
    restart_file << "import PKDGRAV as msr\n";
    restart_file << "msr.restore(__file__)\n";
    restart_file.close();
    auto par_name = baseName + ".par";
    // This is temporary. We support the old naming convention for now,
    // but we will remove it soon
    std::error_code ec; // We will ignore the error code (creating a symlink is not critical)
    std::filesystem::remove(par_name, ec);
    std::filesystem::create_symlink(baseName, par_name, ec);
}

void MSR::Checkpoint(int iStep, int nSteps, double dTime, double dDelta) {
    struct inWrite in;
    double dsec;

    auto filename = BuildCpName(iStep, ".chk");
    assert(filename.size() < sizeof(in.achOutFile));
    strcpy(in.achOutFile, filename.c_str());
    in.nProcessors = parallel_write_count();
    if (csm->val.bComove) {
        double dExp = csmTime2Exp(csm, dTime);
        msrprintf("Writing checkpoint for Step: %d Time:%g Redshift:%g\n",
                  iStep, dTime, (1.0 / dExp - 1.0));
    }
    else {
        msrprintf("Writing checkpoint for Step: %d Time:%g\n", iStep, dTime);
    }

    TimerStart(TIMER_IO);

    pstCheckpoint(pst, &in, sizeof(in), NULL, 0);

    writeParameters(filename, iStep, nSteps, dTime, dDelta);

    /* This is not necessary, but it means the bounds will be identical upon restore */
    CalcBound();

    TimerStop(TIMER_IO);
    dsec = TimerGet(TIMER_IO);
    msrprintf("Checkpoint has been successfully written, Wallclock: %f secs.\n", dsec);
}

void MSR::SetDerivedParameters(bool bRestart) {
    /**********************************************************************\
    * The following "parameters" are derived from real parameters.
    \**********************************************************************/

    /* Temperature-to-specific - internal - energy conversion factors for different values
     * of mean molecular weight (mu) */
    units = UNITS(parameters, csm->val.h);
    const double dTuPrefac = units.dGasConst / (parameters.get_dConstGamma() - 1.);
    dTuFac = dTuPrefac / parameters.get_dMeanMolWeight();

    const double dInvPrimNeutralMu = 0.25 + 0.75 * parameters.get_dInitialH();
    const double dInvPrimIonisedMu = 0.75 + 1.25 * parameters.get_dInitialH();
    dTuFacPrimNeutral = dTuPrefac * dInvPrimNeutralMu;
    dTuFacPrimIonised = dTuPrefac * dInvPrimIonisedMu;

#ifdef COOLING
    SetCoolingParam();
#endif
#ifdef STAR_FORMATION
    SetStarFormationParam();
#endif
#ifdef FEEDBACK
    SetFeedbackParam();
#endif
#if defined(EEOS_POLYTROPE) || defined(EEOS_JEANS)
    SetEOSParam();
#endif
#ifdef BLACKHOLES
    SetBlackholeParam();
#endif
#ifdef STELLAR_EVOLUTION
    SetStellarEvolutionParam();
#endif
}

void MSR::Initialize() {
    char ach[256];

    lcl.pkd = NULL;
    nThreads = mdlThreads(mdl);
    lStart = time(0);
    fCenter = 0; // Center is at (0, 0, 0)
    /* Storage for output times*/
    dOutTimes.reserve(100); // Reasonable number
    dOutTimes.push_back(INFINITY); // Sentinal node
    iOut = 0;

    iCurrMaxRung = 0;
    iRungDD = 0;
    iRungDT = 0;
    iLastRungRT = -1;
    iLastRungDD = -1;  /* Domain decomposition is not done */
    nRung.resize(MAX_RUNG + 1, 0);
    csmInitialize(&csm);

    /*
    ** Create the processor subset tree.
    */
    if (nThreads > 1) {
        msrprintf("Adding %d through %d to the PST\n", 1, nThreads);
        ServiceSetAdd::input inAdd(nThreads);
        mdl->RunService(PST_SETADD, sizeof(inAdd), &inAdd);
    }

}

int MSR::GetLock() {
    /*
    ** Attempts to lock run directory to prevent overwriting. If an old lock
    ** is detected with the same achOutName, an abort is signaled. Otherwise
    ** a new lock is created. The bOverwrite parameter flag can be used to
    ** suppress lock checking.
    */
    auto bOverwrite = parameters.get_bOverwrite();

    FILE *fp = NULL;
    char achTmp[256], achFile[256];

    MakePath(parameters.get_achDataSubPath(), LOCKFILE, achFile);
    if (!bOverwrite && (fp = fopen(achFile, "r"))) {
        if (fscanf(fp, "%s", achTmp) != 1) achTmp[0] = '\0';
        (void) fclose(fp);
        if (parameters.get_achOutName() == achTmp) {
            (void) printf("ABORT: %s detected.\nPlease ensure data is safe to "
                          "overwrite. Delete lockfile and try again.\n", achFile);
            return 0;
        }
    }
    if (!(fp = fopen(achFile, "w"))) {
        if (bOverwrite && parameters.get_bVWarnings()) {
            (void) printf("WARNING: Unable to create %s...ignored.\n", achFile);
            return 1;
        }
        else {
            (void) printf("Unable to create %s\n", achFile);
            return 0;
        }
    }
    (void) fprintf(fp, "%s", parameters.get_achOutName().data());
    (void) fclose(fp);
    return 1;
}

int MSR::CheckForStop(const char *achStopFile) {
    /*
    ** Checks for existence of STOPFILE in run directory. If found, the file
    ** is removed and the return status is set to 1, otherwise 0.
    */

    char achFile[256];
    FILE *fp = NULL;
    MakePath(parameters.get_achDataSubPath(), achStopFile, achFile);
    if ((fp = fopen(achFile, "r"))) {
        (void) printf("User interrupt detected.\n");
        (void) fclose(fp);
        (void) unlink(achFile);
        return 1;
    }
    return 0;
}

MSR::MSR(MDL mdl, PST pst) : pst(pst), mdl(static_cast<mdl::mdlClass *>(mdl)), bVDetails(false) {
}

MSR::~MSR() {
    csmFinish(csm);
    if (Py_IsInitialized()) Py_Finalize();
}

void MSR::SetClasses() {
    std::vector<PARTCLASS> classes(PKD_MAX_CLASSES);
    auto nClass = pstGetClasses(pst, NULL, 0, classes.data(), classes.size()*sizeof(PARTCLASS));
    auto n = nClass / sizeof(PARTCLASS);
    assert(n * sizeof(PARTCLASS)==nClass);
    classes.resize(n);
    std::sort(classes.begin(), classes.end());
    pstSetClasses(pst, classes.data(), nClass, NULL, 0);
}

void MSR::SwapClasses(int id) {
    LCL *plcl = pst->plcl;
    PST pst0 = pst;
    int n;
    int rID;

    std::unique_ptr<PARTCLASS[]> pClass {new PARTCLASS[PKD_MAX_CLASSES]};
    n = plcl->pkd->particles.getClasses( PKD_MAX_CLASSES, pClass.get() );
    rID = mdlReqService(pst0->mdl, id, PST_SWAPCLASSES, pClass.get(), n * sizeof(PARTCLASS));
    mdlGetReply(pst0->mdl, rID, pClass.get(), &n);
    n = n / sizeof(PARTCLASS);
    plcl->pkd->particles.setClasses( n, pClass.get(), 0 );
}

void MSR::OneNodeRead(struct inReadFile *in, FIO fio) {
    int id;
    uint64_t nStart;
    PST pst0;
    LCL *plcl;
    int nid;
    ServiceSwapAll::input inswap;
    int rID;

    std::unique_ptr<int[]> nParts {new int[nThreads]};
    for (id = 0; id < nThreads; ++id) {
        nParts[id] = -1;
    }

    nid = pstOneNodeReadInit(pst, in, sizeof(*in), nParts.get(), nThreads * sizeof(nParts[0]));
    assert((size_t)nid == nThreads * sizeof(nParts[0]));
    for (id = 0; id < nThreads; ++id) {
        assert(nParts[id] > 0);
    }

    pst0 = pst;
    while (pst0->nLeaves > 1)
        pst0 = pst0->pstLower;
    plcl = pst0->plcl;

    nStart = nParts[0];
    for (id = 1; id < nThreads; ++id) {
        /*
         * Read particles into the local storage.
         */
        assert(plcl->pkd->FreeStore() >= nParts[id]);
        pkdReadFIO(plcl->pkd, fio, nStart, nParts[id], in->dvFac, in->dTuFac);
        nStart += nParts[id];
        /*
         * Now shove them over to the remote processor.
         */
        SwapClasses(id);
        inswap.idSwap = 0;
        rID = mdl->ReqService(id, PST_SWAPALL, &inswap, sizeof(inswap));
        //rID = mdlReqService(pst0->mdl, id, PST_SWAPALL, &inswap, sizeof(inswap));
        pkdSwapAll(plcl->pkd, id);
        mdlGetReply(pst0->mdl, rID, NULL, NULL);
    }
    assert(nStart == N);
    /*
     * Now read our own particles.
     */
    pkdReadFIO(plcl->pkd, fio, 0, nParts[0], in->dvFac, in->dTuFac);
}

double MSR::SwitchDelta(double dTime, double dDelta, int iStep, int nSteps) {
    if (csm->val.bComove && parameters.has_dRedTo()
            && parameters.has_nSteps() && parameters.has_nStepsSync()) {
        double aTo, tTo;
        const auto nStepsSync = parameters.get_nStepsSync();
        if (iStep < nStepsSync) {
            aTo = 1.0 / (parameters.get_dRedSync() + 1.0);
            nSteps = nStepsSync - iStep;
        }
        else {
            aTo = 1.0/(parameters.get_dRedTo() + 1.0);
            nSteps = nSteps - iStep;
        }
        assert(nSteps > 0);
        tTo = csmExp2Time(csm, aTo);
        dDelta = (tTo - dTime) / nSteps;
        if (iStep == nStepsSync && bVDetails)
            printf("dDelta changed to %g at z = 10\n", dDelta);
    }
    else if ( dOutTimes.size()>1) {
        dDelta = dOutTimes[iStep + 1]-dOutTimes[iStep];
        printf("Changing dDelta to %e \n", dDelta);
    }

    return dDelta;
}

double MSR::getTime(double dExpansion) {
    if (csm->val.bComove) {
        if (csm->val.dHubble0 == 0.0) {
            printf("No hubble constant specified\n");
            Exit(1);
        }
        return csmExp2Time(csm, dExpansion);
    }
    else return dExpansion;
}

double MSR::getVfactor(double dExpansion) {
    return csm->val.bComove ? dExpansion * dExpansion : 1.0;
}

void MSR::RecvArray(void *vBuffer, PKD_FIELD field, int iUnitSize, double dTime, bool bMarked) {
    PKD pkd = pst->plcl->pkd;
    inSendArray in;
    in.field = field;
    in.iUnitSize = iUnitSize;
    in.bMarked = bMarked;
    if (csm->val.bComove) {
        auto dExp = csmTime2Exp(csm, dTime);
        in.dvFac = 1.0/(dExp * dExp);
    }
    else in.dvFac = 1.0;
    int iIndex = 0;
    vBuffer = pkdPackArray(pkd, iUnitSize * pkd->Local(), vBuffer, &iIndex, pkd->Local(), field, iUnitSize, in.dvFac, in.bMarked);
    assert (iIndex == pkd->Local());
    for (auto i = 1; i < nThreads; ++i) {
        in.iTo = 0;
        auto rID = mdlReqService(pkd->mdl, i, PST_SENDARRAY, &in, sizeof(in));
        vBuffer = pkdRecvArray(pkd, i, vBuffer, iUnitSize);
        mdlGetReply(pkd->mdl, rID, NULL, NULL);
    }
}

/*
** This function makes some potentially problematic assumptions!!!
** Main problem is that it calls pkd level routines, bypassing the
** pst level. It uses plcl pointer which is not desirable.
*/
void MSR::AllNodeWrite(const char *pszFileName, double dTime, double dvFac, int bDouble) {
    int nProcessors;
    struct inWrite in;

    /*
    ** Add Data Subpath for local and non-local names.
    */
    MSR::MakePath(parameters.get_achDataSubPath(), pszFileName, in.achOutFile);

    in.bStandard = parameters.get_bStandard();
    nProcessors = parallel_write_count();
    in.iIndex = 0;

    in.dTime = dTime;
    if (csm->val.bComove) {
        in.dExp = csmTime2Exp(csm, dTime);
        in.dvFac = 1.0/(in.dExp * in.dExp);
    }
    else {
        in.dExp = 1.0;
        in.dvFac = 1.0;
    }

    /* We need to enforce periodic boundaries (when applicable) */
    auto period = parameters.get_dPeriod();
    if (parameters.get_bPeriodic() && blitz::all(period < FLOAT_MAXVAL)) {
        Bound::coord_type offset(0.5 * period);
        in.bnd = Bound(fCenter - offset, fCenter + offset);
    }
    else {
        in.bnd = Bound(Bound::coord_type(-std::numeric_limits<double>::max()), Bound::coord_type(std::numeric_limits<double>::max()));
    }

    in.dEcosmo    = dEcosmo;
    in.dTimeOld   = dTimeOld;
    in.dUOld      = dUOld;
    in.dTuFac     = dTuFac;
    in.units      = units;
    in.dBoxSize   = in.units.dKpcUnit * 1e-3 * csm->val.h;
    in.Omega0     = csm->val.dOmega0;
    in.OmegaLambda= csm->val.dLambda;
    in.HubbleParam= csm->val.h;

    in.nDark = nDark;
    in.nGas  = nGas;
    in.nStar = nStar;
    in.nBH = nBH;

    in.bHDF5 = parameters.get_bHDF5();
    in.mFlags = FIO_FLAG_POTENTIAL | FIO_FLAG_DENSITY
                | (bDouble?FIO_FLAG_CHECKPOINT:0)
                | (parameters.get_bDoublePos()?FIO_FLAG_DOUBLE_POS:0)
                | (parameters.get_bDoubleVel()?FIO_FLAG_DOUBLE_VEL:0)
                | ((parameters.get_bMemParticleID()||MeshlessHydro())?FIO_FLAG_ID:0)
                | (parameters.get_bMemMass()?0:FIO_FLAG_COMPRESS_MASS)
                | (parameters.get_bMemSoft()?0:FIO_FLAG_COMPRESS_SOFT)
                | (parameters.has_dSoft()?FIO_FLAG_GLOBAL_SOFT:0)
                | (parameters.get_bMemUnordered()?FIO_FLAG_UNORDERED:0);

    if (!in.bHDF5 && strstr(in.achOutFile, "&I")==0) {
        FIO fio;
        fio = fioTipsyCreate(in.achOutFile,
                             in.mFlags & FIO_FLAG_CHECKPOINT,
                             in.bStandard, NewSPH() ? in.dTime : in.dExp,
                             in.nGas, in.nDark, in.nStar);
        fioClose(fio);
    }
    in.iLower = 0;
    in.iUpper = nThreads;
    in.iIndex = 0;
    in.nProcessors = nProcessors;
    pstWrite(pst, &in, sizeof(in), NULL, 0);
}

uint64_t MSR::CalcWriteStart() {
    struct outSetTotal out;
    struct inSetWriteStart in;

    pstSetTotal(pst, NULL, 0, &out, sizeof(out));
    assert(out.nTotal <= N);
    in.nWriteStart = 0;
    pstSetWriteStart(pst, &in, sizeof(in), NULL, 0);
    return out.nTotal;
}

void MSR::Write(const std::string &pszFileName, double dTime, int bCheckpoint) {
    char achOutFile[PST_FILENAME_SIZE];
    int nProcessors;
    double dvFac, dExp;
    double dsec;

#ifdef NAIVE_DOMAIN_DECOMP
    Reorder();
#else
    if (iLastRungRT >= 0) Reorder();
#endif

    /*
    ** Calculate where to start writing.
    ** This sets plcl->nWriteStart.
    */
    /*uint64_t N =*/ CalcWriteStart();
    /*
    ** Add Data Subpath for local and non-local names.
    */
    MSR::MakePath(parameters.get_achDataSubPath(), pszFileName.c_str(), achOutFile);

    nProcessors = parallel_write_count();
    auto bHDF5 = parameters.get_bHDF5();

    if (csm->val.bComove) {
        dExp = csmTime2Exp(csm, dTime);
        dvFac = 1.0/(dExp * dExp);
    }
    else {
        dExp = dTime;
        dvFac = 1.0;
    }
    if (nProcessors == 1) {
        msrprintf("Writing %s in %s format serially ...\n",
                  achOutFile, (bHDF5?"HDF5":"Tipsy"));
    }
    else {
        msrprintf("Writing %s in %s format in parallel (but limited to %d processors) ...\n",
                  achOutFile, (bHDF5?"HDF5":"Tipsy"), nProcessors);
    }

    if (csm->val.bComove)
        msrprintf("Time:%g Redshift:%g\n", dTime, (1.0 / dExp - 1.0));
    else
        msrprintf("Time:%g\n", dTime);

    TimerStart(TIMER_IO);
    AllNodeWrite(achOutFile, dTime, dvFac, bCheckpoint);
    TimerStop(TIMER_IO);
    dsec = TimerGet(TIMER_IO);

    msrprintf("Output file has been successfully written, Wallclock: %f secs.\n", dsec);
}

void MSR::SetSoft(double dSoft) {
    msrprintf("Set Softening...\n");
    ServiceSetSoft::input in(dSoft);
    mdl->RunService(PST_SETSOFT, sizeof(in), &in);
}

// Use the smSmooth routine with an empty smooth function to initialize fBall
void MSR::InitBall() {
    printf("Computing a first guess for the smoothing length\n");

    bUpdateBall = 1;
    Smooth(1., 0.0, SMX_NULL, 0, parameters.get_nSmooth());
    bUpdateBall = 0;
}

void MSR::DomainDecompOld(int iRung) {
    OldDD::ServiceDomainDecomp::input in;
    uint64_t nActive;
    const uint64_t nDT = (N * parameters.get_dFracDualTree());
    const uint64_t nDD = (N * parameters.get_dFracNoDomainDecomp());
    const uint64_t nRT = (N * parameters.get_dFracNoDomainRootFind());
    const uint64_t nSD = (N * parameters.get_dFracNoDomainDimChoice());
    double dsec;
    int iRungDT, iRungDD = 0, iRungRT, iRungSD;
    int i;
    int bRestoreActive = 0;

    in.bDoRootFind = 1;
    in.bDoSplitDimFind = 1;
    if (iRung > 0) {
        /*
        ** All of this could be calculated once for the case that the number
        ** of particles don't change. Or calculated every time the number of
        ** particles does change.
        */
        nActive = 0;
        iRungDT = 0;
        iRungDD = 0;
        iRungRT = 0;
        iRungSD = 0;
        for (i = iCurrMaxRung; i >= 0; --i) {
            nActive += nRung[i];
            if (nActive > nDT && !iRungDT) iRungDT = i;
            if (nActive > nDD && !iRungDD) iRungDD = i;
            if (nActive > nRT && !iRungRT) iRungRT = i;
            if (nActive > nSD && !iRungSD) iRungSD = i;
        }
        assert(iRungDD >= iRungRT);
        assert(iRungRT >= iRungSD);
#ifdef NAIVE_DOMAIN_DECOMP
        if (iLastRungRT < 0) {
            /*
            ** We need to do a full domain decompotition with iRungRT particles being active.
            ** However, since I am not sure what the exact state of the domains can be at this point
            ** I had better do a split dim find as well.
            */
            iLastRungRT = 0;
            ActiveRung(iLastRungRT, 1);
            bRestoreActive = 1;
            in.bDoRootFind = 1;
            in.bDoSplitDimFind = 1;
            if (parameters.get_bVRungStat()) {
                printf("Doing Domain Decomposition (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungRT:%d)\n",
                       nActive, N, iRung, iRungRT);
            }
        }
        else if (iRung <= iRungRT) {
            /*
            ** We need to do a full domain decomposition with *ALL* particles being active.
            */
            in.bDoRootFind = 1;
            if (iRung <= iRungSD) {
                if (parameters.get_bVRungStat()) {
                    printf("Doing Domain Decomposition (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungRT:%d)\n",
                           nActive, N, iRung, iRungRT);
                }
                in.bDoSplitDimFind = 1;
            }
            else {
                if (parameters.get_bVRungStat()) {
                    printf("Skipping Domain Dim Choice (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungSD:%d)\n",
                           nActive, N, iRung, iRungSD);
                }
                in.bDoSplitDimFind = 0;
            }
            ActiveRung(0, 1); /* Here we activate all particles. */
            bRestoreActive = 1;
        }
        else if (iRung <= iRungDD) {
            if (parameters.get_bVRungStat()) {
                printf("Skipping Root Finder (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungRT:%d iRungDD:%d)\n",
                       nActive, N, iRung, iRungRT, iRungDD);
            }
            in.bDoRootFind = 0;
            in.bDoSplitDimFind = 0;
            bRestoreActive = 0;
        }
        else {
            if (parameters.get_bVRungStat()) {
                printf("Skipping Domain Decomposition (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungDD:%d)\n",
                       nActive, N, iRung, iRungDD);
            }
            return; /* do absolutely nothing! */
        }
#else
        if (iLastRungRT < 0) {
            /*
            ** We need to do a full domain decompotition with iRungRT particles being active.
            ** However, since I am not sure what the exact state of the domains can be at this point
            ** I had better do a split dim find as well.
            */
            iLastRungRT = iRungRT;
            ActiveRung(iRungRT, 1);
            bRestoreActive = 1;
            in.bDoRootFind = 1;
            in.bDoSplitDimFind = 1;
        }
        else if (iRung == iLastRungDD) {
            if (parameters.get_bVRungStat()) {
                printf("Skipping Domain Decomposition (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungDD:%d iLastRungRT:%d)\n",
                       nActive, N, iRung, iRungDD, iLastRungRT);
            }
            return;  /* do absolutely nothing! */
        }
        else if (iRung >= iRungDD && !bSplitVA) {
            if (iLastRungRT < iRungRT) {
                iLastRungRT = iRungRT;
                ActiveRung(iRungRT, 1);
                bRestoreActive = 1;
                in.bDoRootFind = 1;
                in.bDoSplitDimFind = 0;
            }
            else {
                if (parameters.get_bVRungStat()) {
                    printf("Skipping Domain Decomposition (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungDD:%d iLastRungRT:%d)\n",
                           nActive, N, iRung, iRungDD, iLastRungRT);
                }
                return;  /* do absolutely nothing! */
            }
        }
        else if (iRung > iRungRT) {
            if (iLastRungRT < iRungRT) {
                iLastRungRT = iRungRT;
                ActiveRung(iRungRT, 1);
                bRestoreActive = 1;
                in.bDoRootFind = 1;
                in.bDoSplitDimFind = 0;
            }
            else {
                if (parameters.get_bVRungStat()) {
                    printf("Skipping Root Finder (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungRT:%d iRungDD:%d iLastRungRT:%d)\n",
                           nActive, N, iRung, iRungRT, iRungDD, iLastRungRT);
                }
                in.bDoRootFind = 0;
                in.bDoSplitDimFind = 0;
            }
        }
        else if (iRung > iRungSD) {
            if (iLastRungRT == iRung) {
                if (parameters.get_bVRungStat()) {
                    printf("Skipping Root Finder (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungRT:%d iRungDD:%d iLastRungRT:%d)\n",
                           nActive, N, iRung, iRungRT, iRungDD, iLastRungRT);
                }
                in.bDoRootFind = 0;
                in.bDoSplitDimFind = 0;
            }
            else {
                if (parameters.get_bVRungStat()) {
                    printf("Skipping Domain Dim Choice (nActive = %" PRIu64 "/%" PRIu64 ", iRung:%d iRungSD:%d iLastRungRT:%d)\n",
                           nActive, N, iRung, iRungSD, iLastRungRT);
                }
                iLastRungRT = iRung;
                in.bDoRootFind = 1;
                in.bDoSplitDimFind = 0;
            }
        }
        else {
            if (iLastRungRT == iRung) {
                in.bDoRootFind = 0;
                in.bDoSplitDimFind = 0;
            }
            else {
                iLastRungRT = iRung;
                in.bDoRootFind = 1;
                in.bDoSplitDimFind = 1;
            }
        }
#endif
    }
    else nActive = N;
    iLastRungDD = iLastRungRT;
    in.nActive = nActive;
    in.nTotal = N;

    in.nBndWrap[0] = 0;
    in.nBndWrap[1] = 0;
    in.nBndWrap[2] = 0;
    /*
    ** If we are dealing with a nice periodic volume in all
    ** three dimensions then we can set the initial bounds
    ** instead of calculating them.
    */
    auto period = parameters.get_dPeriod();
    if (parameters.get_bPeriodic() && blitz::all(period < FLOAT_MAXVAL)) {
        Bound::coord_type offset(0.5 * period);
        in.bnd = Bound(fCenter - offset, fCenter + offset);
        mdl->RunService(PST_ENFORCEPERIODIC, sizeof(in.bnd), &in.bnd);
    }
    else {
        mdl->RunService(PST_COMBINEBOUND, &in.bnd);
    }
    /* We make sure that the classes are synchronized among all the domains,
     * otherwise a new class type being moved to another DD region could cause
     * very nasty bugs!
     */
    SetClasses();

    msrprintf("Domain Decomposition: nActive (Rung %d) %" PRIu64 "\n",
              iLastRungRT, nActive);
    msrprintf("Domain Decomposition... \n");
    TimerStart(TIMER_DOMAIN);

    mdl->RunService(PST_DOMAINDECOMP, sizeof(in), &in);
    TimerStop(TIMER_DOMAIN);
    dsec = TimerGet(TIMER_DOMAIN);
    printf("Domain Decomposition complete, Wallclock: %f secs\n\n", dsec);
    if (bRestoreActive) {
        /* Restore Active data */
        ActiveRung(iRung, 1);
    }
}

void MSR::DomainDecomp(int iRung) {
    DomainDecompOld(iRung);
}

/*
** This the meat of the tree build, but will be called by differently named
** functions in order to implement special features without recoding...
*/
void MSR::BuildTree(int bNeedEwald, uint32_t uRoot, uint32_t utRoot) {
    struct inBuildTree in;
    const double ddHonHLimit = parameters.get_ddHonHLimit();
    PST pst0;
    LCL *plcl;
    PKD pkd;
    double dsec;

    pst0 = pst;
    while (pst0->nLeaves > 1)
        pst0 = pst0->pstLower;
    plcl = pst0->plcl;
    pkd = plcl->pkd;

    auto nTopTree = pkd->NodeSize() * (2 * nThreads - 1);
    auto nMsgSize = sizeof(ServiceDistribTopTree::input) + nTopTree;

    std::unique_ptr<char[]> buffer {new char[nMsgSize]};
    auto pDistribTop = new (buffer.get()) ServiceDistribTopTree::input;
    auto pkdn = pkd->tree[reinterpret_cast<KDN *>(pDistribTop + 1)];
    pDistribTop->uRoot = uRoot;
    pDistribTop->allocateMemory = 1;

    in.nBucket = parameters.get_nBucket();
    in.nGroup = parameters.get_nGroup();
    in.uRoot = uRoot;
    in.utRoot = utRoot;
    in.ddHonHLimit = ddHonHLimit;
    TimerStart(TIMER_TREE);
    nTopTree = pstBuildTree(pst, &in, sizeof(in), pkdn, nTopTree);
    pDistribTop->nTop = nTopTree / pkd->NodeSize();
    assert(pDistribTop->nTop == (2 * nThreads - 1));
    mdl->RunService(PST_DISTRIBTOPTREE, nMsgSize, pDistribTop);
    TimerStop(TIMER_TREE);
    dsec = TimerGet(TIMER_TREE);
    printf("Tree built, Wallclock: %f secs\n\n", dsec);

    if (bNeedEwald) {
        /*
        ** For simplicity we will skip calculating the Root for all particles
        ** with exclude very active since there are missing particles which
        ** could add to the mass and because it probably is not important to
        ** update the root so frequently.
        */
        ServiceCalcRoot::input calc;
        ServiceCalcRoot::output root;
        calc.com = pkdn->position();
        calc.uRoot = uRoot;

        mdl->RunService(PST_CALCROOT, sizeof(calc), &calc, &root);
        momTreeRoot[uRoot] = root.momc;
        momTreeCom[uRoot][0] = calc.com[0];
        momTreeCom[uRoot][1] = calc.com[1];
        momTreeCom[uRoot][2] = calc.com[2];
    }
}

void MSR::BuildTree(int bNeedEwald) {
    msrprintf("Building local trees...\n\n");

    ServiceDumpTrees::input dump(IRUNGMAX);
    mdl->RunService(PST_DUMPTREES, sizeof(dump), &dump);
    BuildTree(bNeedEwald, ROOT, 0);

    if (bNeedEwald) {
        ServiceDistribRoot::input droot;
        droot.momc = momTreeRoot[ROOT];
        droot.r[0] = momTreeCom[ROOT][0];
        droot.r[1] = momTreeCom[ROOT][1];
        droot.r[2] = momTreeCom[ROOT][2];
        mdl->RunService(PST_DISTRIBROOT, sizeof(droot), &droot);
    }
#ifdef OPTIM_REORDER_IN_NODES
    if (MeshlessHydro()) {
        ReorderWithinNodes();
    }
#endif
}

/*
** Separates the particles into two trees, and builds the "fixed" tree.
*/
void MSR::BuildTreeFixed(int bNeedEwald, uint8_t uRungDD) {
    msrprintf("Building fixed local trees...\n\n");
    BuildTree(bNeedEwald, FIXROOT, 0);
}

void MSR::BuildTreeActive(int bNeedEwald, uint8_t uRungDD) {
    /*
     ** The trees reset / removed. This does the following:
     **   1. Closes any open cell cache (it will be subsequently invalid)
     **   2. Resets the number of used nodes to zero (or more if we keep the main tree)
     **   3. Sets up the ROOT and FIXROOT node (either of which may have zero particles).
     */

    msrprintf("Building active local trees...\n\n");

    ServiceDumpTrees::input dump(uRungDD, true);
    mdl->RunService(PST_DUMPTREES, sizeof(dump), &dump);

    /* New build the very active tree */
    BuildTree(bNeedEwald, ROOT, FIXROOT);

    /* For ewald we have to shift and combine the individual tree moments */
    if (bNeedEwald) {
        ServiceDistribRoot::input droot;
        MOMC momc;
        double *com1 = momTreeCom[FIXROOT];
        double    m1 = momTreeRoot[FIXROOT].m;
        double *com2 = momTreeCom[ROOT];
        double    m2 = momTreeRoot[ROOT].m;
        double ifMass = 1.0 / (m1 + m2);
        double x, y, z;
        int j;

        /* New Center of Mass, then shift and scale the moments */
        for (j = 0; j < 3; ++j) droot.r[j] = ifMass*(m1 * com1[j] + m2 * com2[j]);

        droot.momc = momTreeRoot[FIXROOT];
        x = com1[0] - droot.r[0];
        y = com1[1] - droot.r[1];
        z = com1[2] - droot.r[2];
        momShiftMomc(&droot.momc, x, y, z);

        momc = momTreeRoot[ROOT];
        x = com2[0] - droot.r[0];
        y = com2[1] - droot.r[1];
        z = com2[2] - droot.r[2];
        momShiftMomc(&momc, x, y, z);

        momAddMomc(&droot.momc, &momc);

        mdl->RunService(PST_DISTRIBROOT, sizeof(droot), &droot);
    }
}

void MSR::BuildTreeMarked(int bNeedEwald) {
    ServiceDumpTrees::input dump(IRUNGMAX);
    mdl->RunService(PST_DUMPTREES, sizeof(dump), &dump);

    pstTreeInitMarked(pst, NULL, 0, NULL, 0);
    BuildTree(bNeedEwald, FIXROOT, 0);
    BuildTree(bNeedEwald, ROOT, FIXROOT);

    /* For ewald we have to shift and combine the individual tree moments */
    if (bNeedEwald) {
        ServiceDistribRoot::input droot;
        MOMC momc;
        double *com1 = momTreeCom[FIXROOT];
        double    m1 = momTreeRoot[FIXROOT].m;
        double *com2 = momTreeCom[ROOT];
        double    m2 = momTreeRoot[ROOT].m;
        double ifMass = 1.0 / (m1 + m2);
        double x, y, z;
        int j;

        /* New Center of Mass, then shift and scale the moments */
        for (j = 0; j < 3; ++j) droot.r[j] = ifMass*(m1 * com1[j] + m2 * com2[j]);

        droot.momc = momTreeRoot[FIXROOT];
        x = com1[0] - droot.r[0];
        y = com1[1] - droot.r[1];
        z = com1[2] - droot.r[2];
        momShiftMomc(&droot.momc, x, y, z);

        momc = momTreeRoot[ROOT];
        x = com2[0] - droot.r[0];
        y = com2[1] - droot.r[1];
        z = com2[2] - droot.r[2];
        momShiftMomc(&momc, x, y, z);

        momAddMomc(&droot.momc, &momc);

        mdl->RunService(PST_DISTRIBROOT, sizeof(droot), &droot);
    }
}

void MSR::Reorder() {
    if (!parameters.get_bMemUnordered()) {
        double sec, dsec;

        msrprintf("Ordering...\n");
        sec = Time();
#ifdef NEW_REORDER
        // Start by dividing the particles by processor; cores will follow
        auto nPerProc = (N + mdl->Procs() - 1) / mdl->Procs();
        printf("Divided %llu particles into %d domains (%llu)\n", N, mdl->Procs(), nPerProc);
        NewDD::ServiceReorder::input indomain(nPerProc, MaxOrder());
        mdl->RunService(PST_REORDER, sizeof(indomain), &indomain);
#else
        OldDD::ServiceDomainOrder::input indomain(MaxOrder());
        mdl->RunService(PST_DOMAINORDER, sizeof(indomain), &indomain);
        OldDD::ServiceLocalOrder::input inlocal(MaxOrder());
        mdl->RunService(PST_LOCALORDER, sizeof(inlocal), &inlocal);
#endif
        dsec = Time() - sec;
        msrprintf("Order established, Wallclock: %f secs\n\n", dsec);

        /*
        ** Mark domain decomp as not done.
        */
        iLastRungRT = -1;
        iLastRungDD = -1;
    }
}

void MSR::OutASCII(const char *pszFile, int iType, int nDims, int iFileType) {

    char achOutFile[PST_FILENAME_SIZE];
    LCL *plcl;
    PST pst0;
    int id, iDim;
    ServiceSwapAll::input inswap;
    PKDOUT pkdout;
    const char *arrayOrVector;
    struct outSetTotal total;
    int rID;

    switch (nDims) {
    case 1:
        arrayOrVector = "array";
        break;
    case 3:
        arrayOrVector = "vector";
        break;
    default:
        arrayOrVector = NULL;
        assert(nDims == 1 || nDims == 3);
    }

    pst0 = pst;
    while (pst0->nLeaves > 1)
        pst0 = pst0->pstLower;
    plcl = pst0->plcl;

    pstSetTotal(pst, NULL, 0, &total, sizeof(total));

    if (pszFile) {
        /*
        ** Add Data Subpath for local and non-local names.
        */
        MSR::MakePath(parameters.get_achDataSubPath(), pszFile, achOutFile);

        switch (iFileType) {
#ifdef HAVE_LIBBZ2
        case PKDOUT_TYPE_BZIP2:
            strcat(achOutFile, ".bz2");
            break;
#endif
#ifdef HAVE_LIBZ
        case PKDOUT_TYPE_ZLIB:
            strcat(achOutFile, ".gz");
            break;
#endif
        default:
            break;
        }

        msrprintf("Writing %s to %s\n", arrayOrVector, achOutFile);
    }
    else {
        printf("No %s Output File specified\n", arrayOrVector);
        Exit(1);
        return;
    }

    if (parallel_write_count()>1 && iFileType > 1) {
        struct inCompressASCII in;
        struct outCompressASCII out;
        struct inWriteASCII inWrite;
        FILE *fp;

        fp = fopen(achOutFile, "wb");
        if ( fp == NULL) {
            printf("Could not create %s Output File:%s\n", arrayOrVector, achOutFile);
            Exit(1);
        }
        fclose(fp);

        inWrite.nFileOffset = 0;
        for (iDim = 0; iDim < nDims; iDim++) {
            in.nTotal = total.nTotal;
            in.iFile = iFileType;
            in.iType = iType;
            in.iDim = iDim;
            pstCompressASCII(pst, &in, sizeof(in), &out, sizeof(out));
            strcpy(inWrite.achOutFile, achOutFile);
            pstWriteASCII(pst, &inWrite, sizeof(inWrite), NULL, 0);
            inWrite.nFileOffset += out.nBytes;
        }
    }
    else {
        pkdout = pkdOpenOutASCII(plcl->pkd, achOutFile, "wb", iFileType, iType);
        if (!pkdout) {
            printf("Could not open %s Output File:%s\n", arrayOrVector, achOutFile);
            Exit(1);
        }

        pkdOutHdr(plcl->pkd, pkdout, total.nTotal);

        /*
         * First write our own particles.
         */
        for (iDim = 0; iDim < nDims; ++iDim) {
            pkdOutASCII(plcl->pkd, pkdout, iType, iDim);
            for (id = 1; id < nThreads; ++id) {
                /*
                 * Swap particles with the remote processor.
                 */
                inswap.idSwap = 0;
                rID = mdl->ReqService(id, PST_SWAPALL, &inswap, sizeof(inswap));
                //rID = mdlReqService(pst0->mdl, id, PST_SWAPALL, &inswap, sizeof(inswap));
                pkdSwapAll(plcl->pkd, id);
                mdlGetReply(pst0->mdl, rID, NULL, NULL);
                /*
                 * Write the swapped particles.
                 */
                pkdOutASCII(plcl->pkd, pkdout, iType, iDim);
                /*
                 * Swap them back again.
                 */
                inswap.idSwap = 0;
                rID = mdl->ReqService(id, PST_SWAPALL, &inswap, sizeof(inswap));
                //rID = mdlReqService(pst0->mdl, id, PST_SWAPALL, &inswap, sizeof(inswap));
                pkdSwapAll(plcl->pkd, id);
                mdlGetReply(pst0->mdl, rID, NULL, NULL);
            }
        }
        pkdCloseOutASCII(plcl->pkd, pkdout);
    }
}

void MSR::OutArray(const char *pszFile, int iType, int iFileType) {
    OutASCII(pszFile, iType, 1, iFileType);
}
void MSR::OutArray(const char *pszFile, int iType) {
    OutArray(pszFile, iType, parameters.get_iCompress());
}

void MSR::OutVector(const char *pszFile, int iType, int iFileType) {
    OutASCII(pszFile, iType, 3, iFileType);
}
void MSR::OutVector(const char *pszFile, int iType) {
    OutVector(pszFile, iType, parameters.get_iCompress());
}

void MSR::SmoothSetSMF(SMF *smf, double dTime, double dDelta, int nSmooth) {
    smf->nSmooth = nSmooth;
    smf->dTime = dTime;
    smf->bDoGravity = parameters.get_bDoGravity();
    smf->iMaxRung = parameters.get_iMaxRung();
    smf->units = units;
    if (Comove()) {
        smf->bComove = 1;
        smf->H = csmTime2Hub(csm, dTime);
        smf->a = csmTime2Exp(csm, dTime);
    }
    else {
        smf->bComove = 0;
        smf->H = 0.0;
        smf->a = 1.0;
    }
    smf->gamma = parameters.get_dConstGamma();
    smf->dDelta = dDelta;
    smf->dEtaCourant = parameters.get_dEtaCourant();
    smf->bMeshlessHydro = MeshlessHydro();
    smf->bIterativeSmoothingLength = parameters.get_bIterativeSmoothingLength();
    smf->bStricterSlopeLimiter = parameters.get_bStricterSlopeLimiter();
    smf->bUpdateBall = bUpdateBall;
    smf->nBucket = parameters.get_nBucket();
    smf->dCFLacc = parameters.get_dCFLacc();
    smf->dConstGamma = parameters.get_dConstGamma();
    smf->dhMinOverSoft = parameters.get_dhMinOverSoft();
    smf->dNeighborsStd = parameters.get_dNeighborsStd();
#if defined(EEOS_POLYTROPE) || defined(EEOS_JEANS)
    smf->eEOS = eEOSparam(parameters, calc);
#endif
#ifdef FEEDBACK
    smf->dSNFBDu = calc.dSNFBDu;
    smf->dCCSNFBDelay = calc.dCCSNFBDelay;
    smf->dCCSNFBSpecEnergy = calc.dCCSNFBSpecEnergy;
    smf->dSNIaFBDelay = calc.dSNIaFBDelay;
    smf->dSNIaFBSpecEnergy = calc.dSNIaFBSpecEnergy;
#endif
#ifdef BLACKHOLES
    smf->dBHFBEff = calc.dBHFBEff;
    smf->dBHFBEcrit = calc.dBHFBEcrit;
    smf->dBHAccretionEddFac = calc.dBHAccretionEddFac;
    smf->dBHAccretionAlpha = parameters.get_dBHAccretionAlpha();
    smf->dBHAccretionCvisc = parameters.get_dBHAccretionCvisc();
    smf->bBHFeedback = parameters.get_bBHFeedback();
    smf->bBHAccretion = parameters.get_bBHAccretion();
#endif
#ifdef STELLAR_EVOLUTION
    smf->dWindSpecificEkin = calc.dWindSpecificEkin;
    smf->dSNIaNorm = calc.dSNIaNorm;
    smf->dSNIaScale = calc.dSNIaScale;
    smf->dCCSNMinMass = parameters.get_dCCSNMinMass();
#endif
}

void MSR::Smooth(double dTime, double dDelta, int iSmoothType, int bSymmetric, int nSmooth) {
    struct inSmooth in;

    in.nSmooth = nSmooth;
    in.bPeriodic = parameters.get_bPeriodic();
    in.bSymmetric = bSymmetric;
    in.iSmoothType = iSmoothType;
    SmoothSetSMF(&(in.smf), dTime, dDelta, nSmooth);
    if (parameters.get_bVStep()) {
        double sec, dsec;
        printf("Smoothing...\n");
        sec = MSR::Time();
        pstSmooth(pst, &in, sizeof(in), NULL, 0);
        dsec = MSR::Time() - sec;
        printf("Smooth Calculated, Wallclock: %f secs\n\n", dsec);
    }
    else {
        pstSmooth(pst, &in, sizeof(in), NULL, 0);
    }
}

int MSR::ReSmooth(double dTime, double dDelta, int iSmoothType, int bSymmetric) {
    struct inSmooth in;
    struct outSmooth out;

    in.nSmooth = parameters.get_nSmooth();
    in.bPeriodic = parameters.get_bPeriodic();
    in.bSymmetric = bSymmetric;
    in.iSmoothType = iSmoothType;
    SmoothSetSMF(&(in.smf), dTime, dDelta, parameters.get_nSmooth());
    if (parameters.get_bVStep()) {
        double sec, dsec;
        printf("ReSmoothing...\n");
        sec = MSR::Time();
        pstReSmooth(pst, &in, sizeof(in), &out, sizeof(struct outSmooth));
        dsec = MSR::Time() - sec;
        printf("ReSmooth Calculated, Wallclock: %f secs\n\n", dsec);
    }
    else {
        pstReSmooth(pst, &in, sizeof(in), &out, sizeof(struct outSmooth));
    }
    return out.nSmoothed;
}

#ifdef OPTIM_SMOOTH_NODE
int MSR::ReSmoothNode(double dTime, double dDelta, int iSmoothType, int bSymmetric) {
    struct inSmooth in;
    struct outSmooth out;

    in.nSmooth = parameters.get_nSmooth();
    in.bPeriodic = parameters.get_bPeriodic();
    in.bSymmetric = bSymmetric;
    in.iSmoothType = iSmoothType;
    SmoothSetSMF(&(in.smf), dTime, dDelta, parameters.get_nSmooth());

    pstReSmoothNode(pst, &in, sizeof(in), &out, sizeof(struct outSmooth));

#if defined(INSTRUMENT) && defined(DEBUG_FLUX_INFO)
    if (iSmoothType == SMX_THIRDHYDROLOOP) {
        printf("  (cache access statistics are given per active particle)\n");
        msrPrintStat(&out.sPartNumAccess, "  P - cache access:", 1);
        msrPrintStat(&out.sCellNumAccess, "  C - cache access:", 1);
        msrPrintStat(&out.sPartMissRatio, "  P - cache miss %:", 2);
        msrPrintStat(&out.sCellMissRatio, "  C - cache miss %:", 2);
        msrPrintStat(&out.sComputing,     "     % computing:", 3);
        msrPrintStat(&out.sWaiting,       "     %   waiting:", 3);
        msrPrintStat(&out.sSynchronizing, "     %   syncing:", 3);
    }
#endif
    return out.nSmoothed;
}
#endif

#ifdef OPTIM_REORDER_IN_NODES
void MSR::ReorderWithinNodes() {
    double dsec;
    TimerStart(TIMER_TREE);
    pstReorderWithinNodes(pst, NULL, 0, NULL, 0);

    TimerStop(TIMER_TREE);
    dsec = TimerGet(TIMER_TREE);
    printf("Reordering nodes took %e secs \n", dsec);

}
#endif

void MSR::UpdateSoft(double dTime) {
    const auto dSoft = parameters.get_dSoft();
    const auto dSoftMax = parameters.get_dSoftMax();
    const auto dMaxPhysicalSoft = parameters.get_dMaxPhysicalSoft();
    if (parameters.get_bPhysicalSoft()) {
        struct inPhysicalSoft in;

        in.dFac = 1./csmTime2Exp(csm, dTime);
        in.bSoftMaxMul = parameters.get_bSoftMaxMul();
        in.dSoftMax = dSoftMax;

        if (in.bSoftMaxMul && in.dFac > in.dSoftMax) in.dFac = in.dSoftMax;

        pstPhysicalSoft(pst, &in, sizeof(in), NULL, 0);
    }
    if (dMaxPhysicalSoft > 0) {
        double dFac = csmTime2Exp(csm, dTime);
        if (dSoft > 0.0) {
            // Change the global softening
            if (dSoft * dFac > dMaxPhysicalSoft)
                SetSoft(dMaxPhysicalSoft / dFac);
        }
        else {
            // Individual (either particle or classes) softening are used
            struct inPhysicalSoft in;

            in.bSoftMaxMul = parameters.get_bSoftMaxMul();
            in.dSoftMax = dSoftMax;

            if (in.bSoftMaxMul) {
                const float fPhysFac = dFac; // Factor to convert to physical
                if (fPhysFac < dMaxPhysicalSoft) {
                    // nothing happens, still in the comoving softening regime
                    in.dFac = 1.;
                }
                else {
                    // late-times, softening limited by physical
                    in.dFac = dMaxPhysicalSoft / dFac;
                }
            }

            pstPhysicalSoft(pst, &in, sizeof(in), NULL, 0);

        }
    }
}

#define PRINTGRID(w, FRM, VAR) {                      \
    printf("      % *d % *d % *d % *d % *d % *d % *d % *d % *d % *d\n", \
       w, 0, w, 1, w, 2, w, 3, w, 4, w, 5, w, 6, w, 7, w, 8, w, 9);               \
    for (i = 0;i < nThreads / 10;++i) {\
    printf("%4d: " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR, out[i * 10 + 4].VAR, \
           out[i * 10 + 5].VAR, out[i * 10 + 6].VAR, out[i * 10 + 7].VAR, out[i * 10 + 8].VAR, out[i * 10 + 9].VAR);\
    }\
    switch (nThreads % 10) {\
    case 0: break;\
    case 1: printf("%4d: " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR); break;\
    case 2: printf("%4d: " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR); break;\
    case 3: printf("%4d: " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR); break;\
    case 4: printf("%4d: " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR); break;\
    case 5: printf("%4d: " FRM " " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR, out[i * 10 + 4].VAR); break;\
    case 6: printf("%4d: " FRM " " FRM " " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR, out[i * 10 + 4].VAR, \
           out[i * 10 + 5].VAR); break;\
    case 7: printf("%4d: " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR, out[i * 10 + 4].VAR, \
           out[i * 10 + 5].VAR, out[i * 10 + 6].VAR); break;\
    case 8: printf("%4d: " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR, out[i * 10 + 4].VAR, \
           out[i * 10 + 5].VAR, out[i * 10 + 6].VAR, out[i * 10 + 7].VAR); break;\
    case 9: printf("%4d: " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM " " FRM "\n", i * 10, \
           out[i * 10 + 0].VAR, out[i * 10 + 1].VAR, out[i * 10 + 2].VAR, out[i * 10 + 3].VAR, out[i * 10 + 4].VAR, \
           out[i * 10 + 5].VAR, out[i * 10 + 6].VAR, out[i * 10 + 7].VAR, out[i * 10 + 8].VAR); break;\
    }\
}

void MSR::Hostname() {
    int i;
    std::unique_ptr<ServiceHostname::output[]> out {new ServiceHostname::output[nThreads]};
    mdl->RunService(PST_HOSTNAME, out.get());
    printf("Host Names:\n");
    PRINTGRID(12, "%12.12s", szHostname);
    printf("MPI Rank:\n");
    PRINTGRID(8, "% 8d", iMpiID);
}

void MSR::MemStatus() {
    int i;
    if (bVDetails) {
        std::unique_ptr<struct outMemStatus[]> out {new struct outMemStatus[nThreads]};
        pstMemStatus(pst, 0, 0, out.get(), nThreads * sizeof(struct outMemStatus));
#ifdef __linux__
        printf("Resident (MB):\n");
        PRINTGRID(8, "%8" PRIu64, rss);
        printf("Free Memory (MB):\n");
        PRINTGRID(8, "%8" PRIu64, freeMemory);
#endif
        printf("Tree size (MB):\n");
        PRINTGRID(8, "%8" PRIu64, nBytesTree / 1024 / 1024);
        printf("Checklist size (KB):\n");
        PRINTGRID(8, "%8" PRIu64, nBytesCl / 1024);
        printf("Particle List size (KB):\n");
        PRINTGRID(8, "%8" PRIu64, nBytesIlp / 1024);
        printf("Cell List size (KB):\n");
        PRINTGRID(8, "%8" PRIu64, nBytesIlc / 1024);
    }
}

void msrPrintStat(STAT *ps, char const *pszPrefix, int p) {
    double dSum = ps->dSum;
    double dMax = ps->dMax;
    const char *minmax = "max";

    if (dSum < 0) {
        dSum = -dSum;
        dMax = -dMax;
        minmax = "min";
    }
    if (ps->n > 1) {
        printf("%s %s=%8.*f @%5d avg=%8.*f of %5d std-dev=%8.*f\n", pszPrefix, minmax,
               p, dMax, ps->idMax, p, dSum / ps->n, ps->n, p, sqrt((ps->dSum2 - ps->dSum * ps->dSum / ps->n)/(ps->n - 1)));
    }
    else if (ps->n == 1) {
        printf("%s %s=%8.*f @%5d\n", pszPrefix, minmax, p, dMax, ps->idMax);
    }
    else {
        printf("%s no data\n", pszPrefix);
    }
}

uint8_t MSR::Gravity(uint8_t uRungLo, uint8_t uRungHi, int iRoot1, int iRoot2,
                     double dTime, double dDelta, double dStep, double dTheta,
                     int bKickClose, int bKickOpen, int bEwald, int bGravStep,
                     int nPartRhoLoc, int iTimeStepCrit) {
    SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
    SPHoptions.doGravity = parameters.get_bDoGravity();
    SPHoptions.nPredictRung = uRungLo;
    return Gravity(uRungLo, uRungHi, iRoot1, iRoot2, dTime, dDelta, dStep, dTheta,
                   bKickClose, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit,
                   SPHoptions);
}

uint8_t MSR::Gravity(uint8_t uRungLo, uint8_t uRungHi, int iRoot1, int iRoot2,
                     double dTime, double dDelta, double dStep, double dTheta,
                     int bKickClose, int bKickOpen, int bEwald, int bGravStep,
                     int nPartRhoLoc, int iTimeStepCrit, SPHOptions SPHoptions) {
    struct inGravity in;
    uint64_t nRungSum[IRUNGMAX + 1];
    int i;
    double dsec, dTotFlop, dt, a;
    double dTimeLCP;
    uint8_t uRungMax = 0;
    char c;

    a = csmTime2Exp(csm, dTime);

    if (parameters.get_bVStep()) {
        if (SPHoptions.doDensity && SPHoptions.useDensityFlags) printf("Calculating Density using FastGas, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doDensity && !SPHoptions.useDensityFlags) printf("Calculating Density without FastGas, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doDensityCorrection && SPHoptions.useDensityFlags) printf("Calculating Density Correction using FastGas, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doDensityCorrection && !SPHoptions.useDensityFlags) printf("Calculating Density Correction without FastGas, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doGravity && SPHoptions.doSPHForces) printf("Calculating Gravity and SPH forces, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doGravity && !SPHoptions.doSPHForces) printf("Calculating Gravity, Step:%f (rung %d)", dStep, uRungLo);
        if (!SPHoptions.doGravity && SPHoptions.doSPHForces) printf("Calculating SPH forces, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doSetDensityFlags) printf("Marking Neighbors for FastGas, Step:%f (rung %d)", dStep, uRungLo);
        if (SPHoptions.doSetNNflags) printf("Marking Neighbors of Neighbors for FastGas, Step:%f (rung %d)", dStep, uRungLo);
        if (csm->val.bComove)
            printf(", Redshift:%f\n", 1. / a - 1.);
        else
            printf(", Time:%f\n", dTime);
    }
    in.dTime = dTime;
    in.iRoot1 = iRoot1;
    in.iRoot2 = iRoot2;

    in.dTheta = dTheta;

    in.bPeriodic = parameters.get_bPeriodic();
    in.bEwald = bEwald;
    in.bGPU = parameters.get_bGPU();
    in.dEwCut = parameters.get_dEwCut();
    in.dEwhCut = parameters.get_dEwhCut();
    in.nReps = in.bPeriodic ? parameters.get_nReplicas() : 0;

    // Parameters related to timestepping
    in.ts.iTimeStepCrit = iTimeStepCrit;
    in.ts.nPartRhoLoc = nPartRhoLoc;
    in.ts.nPartColl = parameters.get_nPartColl();
    in.ts.dEccFacMax = parameters.get_dEccFacMax();
    if (bGravStep) in.ts.dRhoFac = 1.0/(a * a * a);
    else in.ts.dRhoFac = 0.0;
    in.ts.dTime = dTime;
    in.ts.dDelta = dDelta;
    in.ts.dEta = Eta();
    in.ts.dPreFacRhoLoc = parameters.get_dPreFacRhoLoc();
    in.ts.bGravStep = bGravStep;
    in.ts.uRungLo = uRungLo;
    in.ts.uRungHi = uRungHi;
    in.ts.uMaxRung = parameters.get_iMaxRung();
    if (csm->val.bComove) in.ts.dAccFac = 1.0/(a * a * a);
    else in.ts.dAccFac = 1.0;

    CalculateKickParameters(&in.kick, uRungLo, dTime, dDelta, dStep, bKickClose, bKickOpen, SPHoptions);

    in.lc.bLightConeParticles = parameters.get_bLightConeParticles();
    in.lc.dBoxSize = parameters.get_dBoxSize();
    if (parameters.get_bLightCone()) {
        in.lc.dLookbackFac = csmComoveKickFac(csm, dTime, (csmExp2Time(csm, 1.0) - dTime));
        dTimeLCP = csmExp2Time(csm, 1.0/(1.0 + parameters.get_dRedshiftLCP()));
        in.lc.dLookbackFacLCP = csmComoveKickFac(csm, dTimeLCP, (csmExp2Time(csm, 1.0) - dTimeLCP));
        auto sqdegLCP = parameters.get_sqdegLCP();
        if (sqdegLCP <= 0 || sqdegLCP >= 4 * M_1_PI * 180.0 * 180.0 ) {
            in.lc.tanalpha_2 = -1; // indicates we want an all sky lightcone, a bit weird but this is the flag.
        }
        else {
            double alpha = sqrt(sqdegLCP * M_1_PI)*(M_PI / 180.0);
            in.lc.tanalpha_2 = tan(0.5 * alpha);  // it is tangent of the half angle that we actually need!
        }
    }
    else {
        in.lc.dLookbackFac = 0.0;
        in.lc.dLookbackFacLCP = 0.0;
    }
    in.lc.hLCP = parameters.get_hLCP();

    /*
    ** Note that in this loop we initialize dt with the full step, not a half step!
    */
    for (i = 0, dt = dDelta; i <= parameters.get_iMaxRung(); ++i, dt *= 0.5) {
        in.lc.dtLCDrift[i] = 0.0;
        in.lc.dtLCKick[i] = 0.0;
        if (i >= uRungLo) {
            if (csm->val.bComove) {
                in.lc.dtLCDrift[i] = csmComoveDriftFac(csm, dTime, dt);
                in.lc.dtLCKick[i] = csmComoveKickFac(csm, dTime, dt);
            }
            else {
                in.lc.dtLCDrift[i] = dt;
                in.lc.dtLCKick[i] = dt;
            }
        }
    }

    in.SPHoptions = SPHoptions;

    outGravityReduct outr;

    TimerStart(TIMER_GRAVITY);

    pstGravity(pst, &in, sizeof(in), &outr, sizeof(outr));

    TimerStop(TIMER_GRAVITY);
    dsec = TimerGet(TIMER_GRAVITY);

    if (bKickOpen) {
        for (i = IRUNGMAX; i >= uRungLo; --i) {
            if (outr.nRung[i]) break;
        }
        assert(i >= uRungLo);
        uRungMax = i;
        iCurrMaxRung = uRungMax;   /* this assignment shouldn't be needed */
        /*
        ** Update only the active rung counts in the master rung counts.
        ** We need to go all the way to IRUNGMAX to clear any prior counts at rungs
        ** deeper than the current uRungMax!
        */
        for (i = uRungLo; i <= IRUNGMAX; ++i) nRung[i] = outr.nRung[i];

        const uint64_t nDT = (N * parameters.get_dFracDualTree());
        const uint64_t nDD = (N * parameters.get_dFracNoDomainDecomp());
        uint64_t nActive = 0;
        iRungDD = 0;
        iRungDT = 0;
        for (i = iCurrMaxRung; i >= 0; --i) {
            nActive += nRung[i];
            if (nActive > nDT && !iRungDT) iRungDT = i;
            if (nActive > nDD && !iRungDD) iRungDD = i;
        }
    }
    if (parameters.get_bVStep()) {
        /*
        ** Output some info...
        */
        dTotFlop = outr.sFlop.dSum;
        if (dsec > 0.0) {
            double dGFlops = dTotFlop / dsec;
            printf("Gravity Calculated, Wallclock: %f secs, Gflops:%.1f, Total Gflop:%.3g\n",
                   dsec, dGFlops, dTotFlop);
            printf("  Gflops: CPU:%.1f, %.1f GPU:%.1f, %.1f",
                   outr.dFlopSingleCPU / dsec, outr.dFlopDoubleCPU / dsec,
                   outr.dFlopSingleGPU / dsec, outr.dFlopDoubleGPU / dsec);
        }
        else {
            printf("Gravity Calculated, Wallclock: %f secs, Gflops:unknown, Total Gflop:%.3g\n",
                   dsec, dTotFlop);
        }
        printf("  Gflop: CPU:%.3g, %.3g GPU:%.3g, %.3g\n",
               outr.dFlopSingleCPU, outr.dFlopDoubleCPU,
               outr.dFlopSingleGPU, outr.dFlopDoubleGPU);
        msrPrintStat(&outr.sLocal,         "  particle  load:", 0);
        msrPrintStat(&outr.sActive,        "  actives   load:", 0);
        msrPrintStat(&outr.sFlop,          "  Gflop     load:", 1);
        msrPrintStat(&outr.sPart,          "  P - P per active:", 2);
        msrPrintStat(&outr.sCell,          "  P - C per active:", 2);
#ifdef INSTRUMENT
        msrPrintStat(&outr.sComputing,     "     % computing:", 3);
        msrPrintStat(&outr.sWaiting,       "     %   waiting:", 3);
        msrPrintStat(&outr.sSynchronizing, "     %   syncing:", 3);
#endif
#ifdef __linux__
        msrPrintStat(&outr.sFreeMemory,    "free memory (GB):", 3);
        msrPrintStat(&outr.sRSS,           "   resident size:", 3);
#endif
        printf("  (cache access statistics are given per active particle)\n");
        msrPrintStat(&outr.sPartNumAccess, "  P - cache access:", 1);
        msrPrintStat(&outr.sCellNumAccess, "  C - cache access:", 1);
        msrPrintStat(&outr.sPartMissRatio, "  P - cache miss %:", 2);
        msrPrintStat(&outr.sCellMissRatio, "  C - cache miss %:", 2);
    }
    if (outr.nTilesTotal > 0) {
        printf("Total tiles processed: %.5e, on the GPU: %.5e, ratio: %2.2f %%\n", (double)outr.nTilesTotal, (double)(outr.nTilesTotal - outr.nTilesCPU), 100.0 - ((double)outr.nTilesCPU)/((double)outr.nTilesTotal)*100.0);
    }
    if (parameters.get_bVRungStat() && bKickOpen) {
        printf("Rung distribution:\n");
        printf("\n");
        nRungSum[uRungMax] = nRung[uRungMax];
        for (i = uRungMax - 1; i >= 0; --i) {
            nRungSum[i] = nRungSum[i + 1] + nRung[i];
        }
        for (i = 0; i <= uRungMax; ++i) {
            if (nRung[i]) break;
        }
        if (nRungSum[0]>0) for (; i <= uRungMax; ++i) {
                c = ' ';
                printf(" %c rung:%d %14" PRIu64 "    %14" PRIu64 "  %3.0f %%\n",
                       c, i, nRung[i], nRungSum[i],
                       ceil(100.0 * nRungSum[i] / nRungSum[0]));
            }
        printf("\n");
    }
    return (uRungMax);
}

void MSR::CalcEandL(int bFirst, double dTime, double *E, double *T, double *U, double *Eth, double *L, double *F, double *W) {
    struct outCalcEandL out;
    double a;
    int k;

    pstCalcEandL(pst, NULL, 0, &out, sizeof(out));
    *T = out.T;
    *U = out.U;
    *Eth = out.Eth;
    for (k = 0; k < 3; k++) L[k] = out.L[k];
    for (k = 0; k < 3; k++) F[k] = out.F[k];
    *W = out.W;
    /*
    ** Do the comoving coordinates stuff.
    ** Currently L is not adjusted for this. Should it be?
    */
    a = csmTime2Exp(csm, dTime);
    if (!csm->val.bComove) *T *= pow(a, 4.0);
    /*
     * Estimate integral (\dot a * U * dt) over the interval.
     * Note that this is equal to integral (W * da) and the latter
     * is more accurate when a is changing rapidly.
     */
    if (csm->val.bComove && !bFirst) {
        dEcosmo += 0.5*(a - csmTime2Exp(csm, dTimeOld))
                   *((*U) + dUOld);
    }
    else {
        dEcosmo = 0.0;
    }
    dTimeOld = dTime;
    dUOld = *U;
    *U *= a;
    *E = (*T) + (*U) - dEcosmo + a * a*(*Eth);
}

void MSR::Drift(double dTime, double dDelta, int iRoot) {
    struct inDrift in;
    double dsec;

#if defined(BLACKHOLES) and !defined(DEBUG_BH_NODRIFT)
    TimerStart(TIMER_BHS);
    BHGasPin(dTime, dDelta);
    TimerStop(TIMER_BHS);
#endif

    TimerStart(TIMER_DRIFT);

    if (csm->val.bComove) {
        in.dDelta = csmComoveDriftFac(csm, dTime, dDelta);
        in.dDeltaVPred = csmComoveKickFac(csm, dTime, dDelta);
    }
    else {
        in.dDelta = dDelta;
        in.dDeltaVPred = dDelta;
    }
    in.dTime = dTime;
    in.dDeltaUPred = dDelta;
    in.bDoGas = DoGas();
    in.iRoot = iRoot;

    pstDrift(pst, &in, sizeof(in), NULL, 0);

    TimerStop(TIMER_DRIFT);
    dsec = TimerGet(TIMER_DRIFT);
    printf("Drift took %.5f seconds \n", dsec);

#if defined(BLACKHOLES) and !defined(DEBUG_BH_NODRIFT)
    TimerStart(TIMER_BHS);
    BHReposition();
    TimerStop(TIMER_BHS);
#endif
}

void MSR::OutputFineStatistics(double dStep, double dTime) {
    if (!parameters.get_bOutFineStatistics())
        return;
    if (dTime==-1) {
        std::string achFile(OutName());
        achFile += ".finelog";
        fpFineLog = fopen(achFile.c_str(), "a");
        assert(fpFineLog != NULL);
        setbuf(fpFineLog, (char *) NULL); /* no buffering */

        /* Write the header */
        fprintf(fpFineLog, "#First line:\n#dStep dTime ");
#ifdef STAR_FORMATION
        fprintf(fpFineLog, "starsFormed massFormed ");
#endif
        fprintf(fpFineLog, "\n");

        fprintf(fpFineLog, "#Second line:\n#nRung[0] nRung[1] nRung[...] nRung[iCurrMaxRung]\n");
    }
    else {
        /* First, we add a line with relevant statistics */
        fprintf(fpFineLog, "%e %e ", dStep, dTime);
#ifdef STAR_FORMATION
        fprintf(fpFineLog, "%d %e ", starFormed, massFormed);
#endif
        fprintf(fpFineLog, "\n");

        /* Second, we add the rung distribution */
        for (int i = 0; i < iCurrMaxRung; i++)
            fprintf(fpFineLog, "%" PRIu64 " ", nRung[i]);
        fprintf(fpFineLog, "\n");

    }
}

void MSR::EndTimestepIntegration(double dTime, double dDelta) {
    struct inEndTimestep in;
    in.units = UNITS(parameters, csm->val.h);
#ifdef GRACKLE
    strcpy(in.achCoolingTable, parameters.get_achCoolingTables().data());
#endif
    in.dTime = dTime;
    in.dDelta = dDelta;
    in.dConstGamma = parameters.get_dConstGamma();
    in.dTuFac = dTuFacPrimNeutral;
#ifdef STAR_FORMATION
    in.dSFThresholdOD = calc.dSFThresholdOD;
#endif
#if defined(EEOS_POLYTROPE) || defined(EEOS_JEANS)
    in.eEOS = eEOSparam(parameters, calc);
#endif
#ifdef BLACKHOLES
    in.dBHRadiativeEff = parameters.get_dBHRadiativeEff();
#endif
#ifdef STELLAR_EVOLUTION
    in.bChemEnrich = parameters.get_bChemEnrich();
#endif
    double dsec;

    ComputeSmoothing(dTime, dDelta);

    printf("Computing primitive variables... ");
    TimerStart(TIMER_ENDINT);
    pstEndTimestepIntegration(pst, &in, sizeof(in), NULL, 0);

    TimerStop(TIMER_ENDINT);
    dsec = TimerGet(TIMER_ENDINT);
    printf("took %.5f seconds\n", dsec);
}

/*
 * For gas, updates predicted velocities to beginning of timestep.
 */
void MSR::KickKDKOpen(double dTime, double dDelta, uint8_t uRungLo, uint8_t uRungHi) {
    struct inKick in;
    TimerStart(TIMER_KICKO);

    in.dTime = dTime;
    if (csm->val.bComove) {
        in.dDelta = csmComoveKickFac(csm, dTime, dDelta);
        in.dDeltaVPred = 0;
    }
    else {
        in.dDelta = dDelta;
        in.dDeltaVPred = 0;
    }
    in.dDeltaU = dDelta;
    in.dDeltaUPred = 0;
    in.uRungLo = uRungLo;
    in.uRungHi = uRungHi;
    in.bDoGas = DoGas();
    pstKick(pst, &in, sizeof(in), NULL, 0);
    TimerStop(TIMER_KICKO);
}

/*
 * For gas, updates predicted velocities to end of timestep.
 */
void MSR::KickKDKClose(double dTime, double dDelta, uint8_t uRungLo, uint8_t uRungHi) {
    struct inKick in;
    TimerStart(TIMER_KICKC);

    in.dTime = dTime;
    if (csm->val.bComove) {
        in.dDelta = csmComoveKickFac(csm, dTime, dDelta);
        in.dDeltaVPred = in.dDelta;
    }
    else {
        in.dDelta = dDelta;
        in.dDeltaVPred = in.dDelta;
    }
    in.dDeltaU = dDelta;
    in.dDeltaUPred = in.dDeltaU;
    in.uRungLo = uRungLo;
    in.uRungHi = uRungHi;
    in.bDoGas = DoGas();
    pstKick(pst, &in, sizeof(in), NULL, 0);
    TimerStop(TIMER_KICKC);
}

int cmpTime(const void *v1, const void *v2) {
    double *d1 = (double *)v1;
    double *d2 = (double *)v2;

    if (*d1 < *d2) return (-1);
    else if (*d1 == *d2) return (0);
    else return (1);
}

int MSR::ReadOuts(double dTime) {
    char achFile[PST_FILENAME_SIZE];
    FILE *fp;
    int ret;
    double z, a, t, n, newt;
    char achIn[80];

    /*
    ** Add Data Subpath for local and non-local names.
    */
    MakePath(parameters.get_achDataSubPath(), parameters.get_achOutTimes(), achFile);

    dOutTimes.clear();
    dOutTimes.push_back(INFINITY); // Sentinal node

    // I do not like how this is done, the file should be provided by a parameter
    fp = fopen(achFile, "r");
    if (!fp) {
        printf("The output times file: %s, is not present!\n", achFile);
        abort();
    }
    while (1) {
        if (!fgets(achIn, 80, fp)) goto NoMoreOuts;
        switch (achIn[0]) {
        case 'z':
            ret = sscanf(&achIn[1], "%lf", &z);
            if (ret != 1) goto NoMoreOuts;
            a = 1.0/(z + 1.0);
            newt = csmExp2Time(csm, a);
            break;
        case 'a':
            ret = sscanf(&achIn[1], "%lf", &a);
            if (ret != 1) goto NoMoreOuts;
            newt = csmExp2Time(csm, a);
            break;
        case 't':
            ret = sscanf(&achIn[1], "%lf", &t);
            if (ret != 1) goto NoMoreOuts;
            newt = t;
            break;
        default:
            ret = sscanf(achIn, "%lf", &z);
            if (ret != 1) goto NoMoreOuts;
            a = 1.0/(z + 1.0);
            newt = csmExp2Time(csm, a);
        }
        dOutTimes.push_back(newt);
    }
NoMoreOuts:
    fclose(fp);
    dOutTimes.push_back(dTime);
    std::sort(dOutTimes.begin(), dOutTimes.end(), std::less < double>());
    assert( dTime < dOutTimes.back() );
    return dOutTimes.size()-2;
}

void MSR::InitCosmology(CSM csm) {
    ServiceInitLightcone::input in;
    if (parameters.has_h()) csm->val.h = parameters.get_h();
    mdl->RunService(PST_INITCOSMOLOGY, sizeof(csm->val), &csm->val);
    if (parameters.get_bLightCone()) {
        auto sqdegLCP = parameters.get_sqdegLCP();
        if (sqdegLCP <= 0 || sqdegLCP >= 4 * M_1_PI * 180.0 * 180.0 ) {
            in.alphaLCP = -1; // indicates we want an all sky lightcone, a bit weird but this is the flag.
        }
        else {
            in.alphaLCP = sqrt(sqdegLCP * M_1_PI)*(M_PI / 180.0);
        }
        in.bBowtie = parameters.get_bBowtie();
        in.bLightConeParticles = parameters.get_bLightConeParticles();
        in.dBoxSize = parameters.get_dBoxSize();
        in.dRedshiftLCP = parameters.get_dRedshiftLCP();
        in.hLCP = parameters.get_hLCP();
        mdl->RunService(PST_INITLIGHTCONE, sizeof(in), &in);
    }
}

void MSR::ZeroNewRung(uint8_t uRungLo, uint8_t uRungHi, int uRung) {
    ServiceZeroNewRung::input in(uRung, uRungLo, uRungHi);
    mdl->RunService(PST_ZERONEWRUNG, sizeof(in), &in);
}

/*
 * bGreater = 1 => activate all particles at this rung and greater.
 */
void MSR::ActiveRung(int iRung, int bGreater) {
    ServiceActiveRung::input in(iRung, bGreater);
    mdl->RunService(PST_ACTIVERUNG, sizeof(in), &in);

    if (iRung == 0 && bGreater)
        nActive = N;
    else {
        int i;

        nActive = 0;
        for ( i = iRung; i<= (bGreater?parameters.get_iMaxRung():iRung); i++ )
            nActive += nRung[i];
    }
}

void MSR::ActiveOrder() {
    pstActiveOrder(pst, NULL, 0, &(nActive), sizeof(nActive));
}

int MSR::CountRungs(uint64_t *nRungs) {
    ServiceCountRungs::output out;
    int i, iMaxRung = 0;
    mdl->RunService(PST_COUNTRUNGS, &out);
    for (i = 0; i <= MAX_RUNG; ++i) {
        nRung[i] = out[i];
        if (nRung[i]) iMaxRung = i;
        if (nRungs) nRungs[i] = nRung[i];
    }
    iCurrMaxRung = iMaxRung;

    const uint64_t nDT = (N * parameters.get_dFracDualTree());
    const uint64_t nDD = (N * parameters.get_dFracNoDomainDecomp());
    uint64_t nActive = 0;
    iRungDD = 0;
    iRungDT = 0;
    for (i = iCurrMaxRung; i >= 0; --i) {
        nActive += nRung[i];
        if (nActive > nDT && !iRungDT) iRungDT = i;
        if (nActive > nDD && !iRungDD) iRungDD = i;
    }

    return iMaxRung;
}

void MSR::AccelStep(uint8_t uRungLo, uint8_t uRungHi, double dTime, double dDelta) {
    struct inAccelStep in;
    double a;

    in.dEta = Eta();
    a = csmTime2Exp(csm, dTime);
    if (csm->val.bComove) {
        in.dVelFac = 1.0/(a * a);
    }
    else {
        in.dVelFac = 1.0;
    }
    in.dAccFac = 1.0/(a * a * a);
    in.bDoGravity = DoGravity();
    in.bEpsAcc = parameters.get_bEpsAccStep();
    in.dDelta = dDelta;
    in.iMaxRung = parameters.get_iMaxRung();
    in.uRungLo = uRungLo;
    in.uRungHi = uRungHi;
    pstAccelStep(pst, &in, sizeof(in), NULL, 0);
}

uint8_t MSR::GetMinDt() {
    struct outGetMinDt out;

    pstGetMinDt(pst, NULL, 0, &out, sizeof(struct outGetMinDt));
    return out.uMinDt;
}

void MSR::SetGlobalDt(uint8_t minDt) {
    struct outGetMinDt in;
    in.uMinDt = minDt;

    pstSetGlobalDt(pst, &in, sizeof(in), NULL, 0);

}

void MSR::DensityStep(uint8_t uRungLo, uint8_t uRungHi, double dTime, double dDelta) {
    struct inDensityStep in;
    double expand;
    int bSymmetric;

    msrprintf("Calculating Rung Densities...\n");
    bSymmetric = 0;
    Smooth(dTime, dDelta, SMX_DENSITY, bSymmetric, parameters.get_nSmooth());
    in.dDelta = dDelta;
    in.iMaxRung = parameters.get_iMaxRung();
    in.dEta = Eta();
    expand = csmTime2Exp(csm, dTime);
    in.dRhoFac = 1.0/(expand * expand * expand);
    in.uRungLo = uRungLo;
    in.uRungHi = uRungHi;
    pstDensityStep(pst, &in, sizeof(in), NULL, 0);
}

/*
 ** Returns the Very Active rung based on the number of very active particles desired,
 ** or the fixed rung that was specified in the parameters.
 */
void MSR::UpdateRung(uint8_t uRung) {
    struct inUpdateRung in;
    struct outUpdateRung out;
    int iTempRung, iOutMaxRung;

    /* If we are called, it is a mistake -- this happens in analysis mode */
    if (parameters.get_bMemUnordered()&&parameters.get_bNewKDK()) return;

    in.uRungLo = uRung;
    in.uRungHi = MaxRung();
    in.uMinRung = uRung;
    in.uMaxRung = MaxRung();

    pstUpdateRung(pst, &in, sizeof(in), &out, sizeof(out));

    iTempRung =MaxRung()-1;
    while (out.nRungCount[iTempRung] == 0 && iTempRung > 0) --iTempRung;
    iOutMaxRung = iTempRung;

    const auto nTruncateRung = parameters.get_nTruncateRung();
    while (out.nRungCount[iOutMaxRung] <= nTruncateRung && iOutMaxRung > uRung) {
        msrprintf("n_CurrMaxRung = %" PRIu64 "  (iCurrMaxRung = %d):  Promoting particles to iCurrMaxrung = %d\n",
                  out.nRungCount[iOutMaxRung], iOutMaxRung, iOutMaxRung - 1);

        in.uMaxRung = iOutMaxRung; /* Note this is the forbidden rung so no -1 here */
        pstUpdateRung(pst, &in, sizeof(in), &out, sizeof(out));

        iTempRung =MaxRung()-1;
        while (out.nRungCount[iTempRung] == 0 && iTempRung > 0) --iTempRung;
        iOutMaxRung = iTempRung;
    }

    /*
    ** Now copy the rung distribution to the msr structure!
    */
    for (iTempRung = 0; iTempRung < MaxRung(); ++iTempRung) nRung[iTempRung] = out.nRungCount[iTempRung];

    iCurrMaxRung = iOutMaxRung;

    if (parameters.get_bVRungStat()) {
        printf("Rung distribution:\n");
        printf("\n");
        for (iTempRung = 0; iTempRung <= iCurrMaxRung; ++iTempRung) {
            if (out.nRungCount[iTempRung] == 0) continue;
            printf("   rung:%d %" PRIu64 "\n", iTempRung, out.nRungCount[iTempRung]);
        }
        printf("\n");
    }
}

/*
 ** Open the healpix output file, and also the particles files if requested.
 */
void MSR::LightConeOpen(int iStep) {
    if (parameters.get_bLightCone()) {
        struct inLightConeOpen lc;
        if (parameters.get_bLightConeParticles() ) {
            auto filename = BuildName(iStep);
            strcpy(lc.achOutFile, filename.c_str());
        }
        else lc.achOutFile[0] = 0;
        lc.nSideHealpix = parameters.get_nSideHealpix();
        pstLightConeOpen(pst, &lc, sizeof(lc), NULL, 0);
    }
}

/*
 ** Close the files for this step.
 */
void MSR::LightConeClose(int iStep) {
    if (parameters.get_bLightCone()) {
        struct inLightConeClose lc;
        auto filename = BuildName(iStep);
        strcpy(lc.achOutFile, filename.c_str());
        pstLightConeClose(pst, &lc, sizeof(lc), NULL, 0);
    }
}

/*
 ** Correct velocities from a^2 x_dot to a x_dot (physical peculiar velocities) using the
 ** position dependent scale factor within the light cone. This could be expensive.
 */
void MSR::LightConeVel() {
    double dsec;
    struct inLightConeVel in;

    in.dBoxSize = parameters.get_dBoxSize();
    TimerStart(TIMER_NONE);
    pstLightConeVel(pst, &in, sizeof(in), NULL, 0);
    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    printf("Converted lightcone velocities to physical, Wallclock: %f secs.\n", dsec);
}

/* True if we should omit the opening kick */
int MSR::CheckForOutput(int iStep, int nSteps, double dTime, int *pbDoCheckpoint, int *pbDoOutput) {
    int iStop, iCheck;
    long lSec = time(0) - lPrior;

    /*
    ** Check for user interrupt.
    */
    iStop = CheckForStop(STOPFILE);
    iCheck = CheckForStop(CHECKFILE);

    /*
    ** Check to see if the runtime has been exceeded.
    */
    const auto iWallRunTime = parameters.get_iWallRunTime();
    if (!iStop && iWallRunTime > 0) {
        if (iWallRunTime * 60 - (time(0)-lStart) < ((int) (lSec * 1.5)) ) {
            printf("RunTime limit exceeded.  Writing checkpoint and exiting.\n");
            printf("    iWallRunTime(sec): %lld   Time running: %ld   Last step: %ld\n",
                   iWallRunTime * 60, time(0)-lStart, lSec);
            iStop = 1;
        }
    }

    /* Check to see if there should be an output */
    const auto iSignalSeconds = parameters.get_iSignalSeconds();
    if (!iStop && timeGlobalSignalTime > 0) { /* USR1 received */
        if ( (time(0)+(lSec * 1.5)) > timeGlobalSignalTime + iSignalSeconds) {
            printf("RunTime limit exceeded.  Writing checkpoint and exiting.\n");
            printf("    iSignalSeconds: %lld   Time running: %ld   Last step: %ld\n",
                   iSignalSeconds, time(0)-lStart, lSec);
            iStop = 1;
        }
    }

    /*
    ** Output if 1) we've hit an output time
    **           2) We are stopping
    **           3) we're at an output interval
    */
    if (iCheck || (CheckInterval()>0 &&
                   (bGlobalOutput
                    || iStop
                    || (iStep % CheckInterval() == 0) )) ) {
        bGlobalOutput = 0;
        *pbDoCheckpoint = 1 | (iStop<<1);
    }

    if ((OutInterval() > 0 &&
            (bGlobalOutput
             || iStop
             || iStep == nSteps
             || (iStep % OutInterval() == 0))) ) {
        bGlobalOutput = 0;
        *pbDoOutput = 1  | (iStop<<1);
    }

    return (iStep == parameters.get_nStepsSync()) || *pbDoOutput || *pbDoCheckpoint;
}

int MSR::NewTopStepKDK(
    double &dTime,  /* MODIFIED: Current simulation time */
    double dDelta,
    double dTheta,
    int nSteps,
    int bDualTree,      /* Should be zero at rung 0! */
    uint8_t uRung,  /* Rung level */
    double *pdStep, /* Current step */
    uint8_t *puRungMax,
    int *pbDoCheckpoint, int *pbDoOutput, int *pbNeedKickOpen) {
    const auto bEwald = parameters.get_bEwald();
    const auto bGravStep = parameters.get_bGravStep();
    const auto nPartRhoLoc = parameters.get_nPartRhoLoc();
    const auto iTimeStepCrit = parameters.get_iTimeStepCrit();
    double dDeltaRung, dTimeFixed;
    uint32_t uRoot2 = 0;
    int bKickOpen = 1;
    /*
    ** The iStep variable serves only to give a number to the lightcone and group output files.
    ** We define this to be the output number of the final radius of the lightcone surface.
    */
    int iStep = (int)(*pdStep) + 1;

    /* IA: If the next rung that is the first one with actives particles, we
     *   are sure that all particles are synchronized, thus we can output some statistics with
     *   finer time resolution whilst being accurate.
     *
     *   We assume that there is no 'sandwiched' rung, i.e., the differences in dt are smooth
     */
    if ( (nRung[0]!=0 && uRung == 0) || ( (nRung[uRung] == 0) && (nRung[uRung + 1] > 0) ))
        OutputFineStatistics(*pdStep, dTime);

    if (uRung == iRungDT + 1) {
        if ( parameters.get_bDualTree() && uRung < *puRungMax) {
            /* HACK: FIXME: Don't use the dual tree before z = 2; the overlap region is too large */
            /* better would be to construct the tree matching remote processor shape as well as local */
            double a = csmTime2Exp(csm, dTime);
            if (a < (1.0 / 3.0)) bDualTree = 0;
            else {
                bDualTree = 1;
                ServiceDumpTrees::input dump(iRungDT);
                mdl->RunService(PST_DUMPTREES, sizeof(dump), &dump);
                msrprintf("Half Drift, uRung: %d\n", iRungDT);
                dDeltaRung = dDelta/(uintmax_t(1) << iRungDT); // Main tree step
                Drift(dTime, 0.5 * dDeltaRung, FIXROOT);
                dTimeFixed = dTime + 0.5 * dDeltaRung;
                BuildTreeFixed(bEwald, iRungDT);
            }
        }
        else bDualTree = 0;
    }
    if (uRung < *puRungMax) {
        bDualTree = NewTopStepKDK(dTime, dDelta, dTheta, nSteps, bDualTree, uRung + 1, pdStep, puRungMax, pbDoCheckpoint, pbDoOutput, pbNeedKickOpen);
    }

    dDeltaRung = dDelta/(uintmax_t(1) << *puRungMax);
    ActiveRung(uRung, 1);
    if (DoGas() && MeshlessHydro()) {
        MeshlessFluxes(dTime, dDelta);
    }
    ZeroNewRung(uRung, MAX_RUNG, uRung);

#ifdef BLACKHOLES
    if (parameters.get_bBHPlaceSeed()) {
        PlaceBHSeed(dTime, *puRungMax);
    }
#endif
    /* Drift the "ROOT" (active) tree or all particle */
    if (bDualTree) {
        msrprintf("Drift very actives, uRung: %d\n", *puRungMax);
        Drift(dTime, dDeltaRung, ROOT);
    }
    else {
        msrprintf("Drift, uRung: %d\n", *puRungMax);
        Drift(dTime, dDeltaRung, -1);
    }
    dTime += dDeltaRung;
    *pdStep += 1.0/(uintmax_t(1) << *puRungMax);
#ifdef COOLING
    if (csm->val.bComove) {
        const float a = csmTime2Exp(csm, dTime);
        const float z = 1./a - 1.;

        CoolingUpdate(z);
    }
    else {
        CoolingUpdate(0.);
    }
#endif
#ifdef STAR_FORMATION
    StarForm(dTime, dDelta, uRung);
#endif

    ActiveRung(uRung, 1);
    UpdateSoft(dTime);
    if (bDualTree && uRung > iRungDT) {
        uRoot2 = FIXROOT;
        BuildTreeActive(bEwald, iRungDT);
    }
    else {
        DomainDecomp(uRung);
        uRoot2 = 0;

        if (NewSPH()) {
            SelAll(-1, 1);
        }

#ifdef BLACKHOLES
        auto bBHMerger = parameters.get_bBHMerger();
        if (bBHMerger) {
            SelActives();
            BHMerger(dTime);
        }
        if (parameters.get_bBHAccretion() && !bBHMerger) {
            struct outGetNParts Nout;

            Nout.n = 0;
            Nout.nDark = 0;
            Nout.nGas = 0;
            Nout.nStar = 0;
            Nout.nBH = 0;
            pstMoveDeletedParticles(pst, NULL, 0, &Nout, sizeof(struct outGetNParts));
        }
#endif

        BuildTree(bEwald);
    }

    if (!uRung) {
        bKickOpen = !CheckForOutput(iStep, nSteps, dTime, pbDoCheckpoint, pbDoOutput);
    }
    else bKickOpen = 1;
    *pbNeedKickOpen = !bKickOpen;

    /*
    ** At this point the particles are in sync. As soon as we call the next gravity it will
    ** advance the particles as part of the opening kick. For this reason we do the requested
    ** analysis at this poing while everything is properly synchronized. This includes
    ** writing the healpix and lightcone particles, as well as measuring P(k) for example.
    */
    if (uRung == 0) {
        runAnalysis(iStep, dTime); // Run any registered Python analysis tasks

        const auto iPkInterval = parameters.get_iPkInterval();
        if (iPkInterval && iStep % iPkInterval == 0) OutputPk(iStep, dTime);

        /*
        ** We need to write all light cone files (healpix and LCP) at this point before the last
        ** gravity is called since it will advance the particles in the light cone as part of the
        ** opening kick! We also need to open
        */
        LightConeClose(iStep);
        if (bKickOpen) LightConeOpen(iStep + 1);

        /* Compute the grids of linear species at main timesteps, before gravity is called */
        auto nGridLin = parameters.get_nGridLin();
        if (csm->val.classData.bClass && parameters.get_achLinSpecies().length() && nGridLin) {
            GridCreateFFT(nGridLin);
            SetLinGrid(dTime, dDelta, nGridLin, 1, bKickOpen);
            if (parameters.get_bDoLinPkOutput())
                OutputLinPk( *pdStep, dTime);
            LinearKick(dTime, dDelta, 1, bKickOpen);
            GridDeleteFFT();
        }

        if (parameters.get_bFindGroups()) NewFof(parameters.get_dTau(), parameters.get_nMinMembers());
    }

    // We need to make sure we descend all the way to the bucket with the
    // active tree, or we can get HUGE group cells, and hence too much P - P / P - C
    if (NewSPH()) {
        SelAll(-1, 1);
        SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
        uint64_t nParticlesOnRung = 0;
        for (int i = MaxRung(); i >= uRung; i--) {
            nParticlesOnRung += nRung[i];
        }
        if (nParticlesOnRung/((float) N) < SPHoptions.FastGasFraction) {
            // Select Neighbors
            SPHoptions.doGravity = 0;
            SPHoptions.doDensity = 0;
            SPHoptions.nPredictRung = uRung;
            SPHoptions.doSPHForces = 0;
            SPHoptions.doSetDensityFlags = 1;
            *puRungMax = Gravity(uRung, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta,
                                 1, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
            // Select Neighbors of Neighbors
            if (SPHoptions.doInterfaceCorrection) {
                SPHoptions.dofBallFactor = 1;
                TreeUpdateFlagBounds(bEwald, ROOT, 0, SPHoptions);
                SPHoptions.doSetDensityFlags = 0;
                SPHoptions.doSetNNflags = 1;
                SPHoptions.useDensityFlags = 1;
                *puRungMax = Gravity(uRung, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta,
                                     1, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
                SPHoptions.doSetNNflags = 0;
                SPHoptions.useDensityFlags = 0;
            }
        }
        // Calculate Density
        SPHoptions.doSetDensityFlags = 0;
        SPHoptions.doGravity = 0;
        SPHoptions.doDensity = 1;
        SPHoptions.nPredictRung = uRung;
        SPHoptions.doSPHForces = 0;
        SPHoptions.useDensityFlags = 0;
        if (nParticlesOnRung/((float) N) < SPHoptions.FastGasFraction) {
            SPHoptions.useDensityFlags = 1;
            SPHoptions.dofBallFactor = 1;
            if (SPHoptions.doInterfaceCorrection) {
                SPHoptions.useNNflags = 1;
            }
            TreeUpdateFlagBounds(bEwald, ROOT, 0, SPHoptions);
            *puRungMax = Gravity(uRung, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta,
                                 1, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
            if (SPHoptions.doInterfaceCorrection) {
                SPHoptions.doDensity = 0;
                SPHoptions.doDensityCorrection = 1;
                SPHoptions.useDensityFlags = 1;
                SPHoptions.useNNflags = 0;
                SPHoptions.dofBallFactor = 0;
                TreeUpdateFlagBounds(bEwald, ROOT, 0, SPHoptions);
                *puRungMax = Gravity(uRung, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta, 1, bKickOpen,
                                     bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
                UpdateGasValues(uRung, dTime, dDelta, *pdStep, 1, bKickOpen, SPHoptions);
                SPHoptions.doDensityCorrection = 0;
                SPHoptions.useDensityFlags = 0;
            }
        }
        else {
            *puRungMax = Gravity(0, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta,
                                 1, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
            if (SPHoptions.doInterfaceCorrection) {
                SPHoptions.doDensity = 0;
                SPHoptions.doDensityCorrection = 1;
                SPHoptions.dofBallFactor = 0;
                TreeUpdateFlagBounds(bEwald, ROOT, 0, SPHoptions);
                *puRungMax = Gravity(0, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta, 1, bKickOpen,
                                     bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
                UpdateGasValues(0, dTime, dDelta, *pdStep, 1, bKickOpen, SPHoptions);
                SPHoptions.doDensityCorrection = 0;
            }
        }
        // Calculate Forces
        SelAll(-1, 1);
        SPHoptions.doGravity = parameters.get_bDoGravity();
        SPHoptions.doDensity = 0;
        SPHoptions.nPredictRung = uRung;
        SPHoptions.doSPHForces = 1;
        SPHoptions.useDensityFlags = 0;
        SPHoptions.dofBallFactor = 0;
        TreeUpdateFlagBounds(bEwald, ROOT, 0, SPHoptions);
        *puRungMax = Gravity(uRung, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta,
                             1, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
    }
    else { /*if (parameters.get_bDoGravity())*/
        SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
        SPHoptions.doGravity = parameters.get_bDoGravity();
        SPHoptions.nPredictRung = uRung;
        *puRungMax = Gravity(uRung, MaxRung(), ROOT, uRoot2, dTime, dDelta, *pdStep, dTheta,
                             1, bKickOpen, bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
    }

#if defined(FEEDBACK) || defined(STELLAR_EVOLUTION)
    ActiveRung(uRung, 0);
#ifdef FEEDBACK
    if (parameters.get_bCCSNFeedback() || parameters.get_bSNIaFeedback()) {
        Smooth(dTime, dDelta, SMX_SN_FEEDBACK, 1, parameters.get_nSmooth());
    }
#endif
#ifdef STELLAR_EVOLUTION
    if (parameters.get_bChemEnrich()) {
        Smooth(dTime, dDelta, SMX_CHEM_ENRICHMENT, 1, parameters.get_nSmooth());
    }
#endif
#endif

    ActiveRung(uRung, 1);
    if (DoGas() && MeshlessHydro()) {
        EndTimestepIntegration(dTime, dDelta);
        MeshlessGradients(dTime, dDelta);
    }

    if (DoGas() && MeshlessHydro()) {
        HydroStep(dTime, dDelta);
        UpdateRung(uRung);
        uint8_t iTempRung;
        for (iTempRung = 0; iTempRung <= iCurrMaxRung; ++iTempRung) {
            if (nRung[iTempRung] == 0) continue;
            *puRungMax = iTempRung;
        }
    }

    if (!uRung && parameters.get_bFindGroups()) {
        GroupStats();
        HopWrite(BuildName(iStep, ".fofstats").c_str());
    }

    if (uRung && uRung < *puRungMax) bDualTree = NewTopStepKDK(dTime, dDelta, dTheta, nSteps, bDualTree, uRung + 1, pdStep, puRungMax, pbDoCheckpoint, pbDoOutput, pbNeedKickOpen);
    if (bDualTree && uRung == iRungDT + 1) {
        msrprintf("Half Drift, uRung: %d\n", iRungDT);
        dDeltaRung = dDelta/(uintmax_t(1) << iRungDT);
        Drift(dTimeFixed, 0.5 * dDeltaRung, FIXROOT);
    }

    return bDualTree;
}

void MSR::TopStepKDK(
    double dStep,    /* Current step */
    double dTime,    /* Current time */
    double dDeltaRung,   /* Time step */
    double dTheta,
    int iRung,       /* Rung level */
    int iKickRung,   /* Gravity on all rungs from iRung to iKickRung */
    int iAdjust) {   /* Do an adjust? */
    double dDeltaStep = dDeltaRung * (uintmax_t(1) << iRung);
    const auto bEwald = parameters.get_bEwald();
    const auto bGravStep = parameters.get_bGravStep();
    const auto nPartRhoLoc = parameters.get_nPartRhoLoc();
#ifdef BLACKHOLES
    if (!iKickRung && !iRung && parameters.get_bBHPlaceSeed()) {
        PlaceBHSeed(dTime, CurrMaxRung());
#ifdef OPTIM_REORDER_IN_NODES
        ReorderWithinNodes();
#endif
    }
#endif

    if (iAdjust && (iRung < MaxRung()-1)) {
        msrprintf("%*cAdjust, iRung: %d\n", 2 * iRung + 2, ' ', iRung);
        /* JW: Note -- can't trash uRungNew here! Force calcs set values for it! */
        ActiveRung(iRung, 1);
        if (parameters.get_bAccelStep()) AccelStep(iRung, MAX_RUNG, dTime, dDeltaStep);
        if (DoGas() && MeshlessHydro()) {
            HydroStep(dTime, dDeltaStep);
        }
        if (parameters.get_bDensityStep()) {
            DomainDecomp(iRung);
            ActiveRung(iRung, 1);
            BuildTree(0);
            DensityStep(iRung, MAX_RUNG, dTime, dDeltaStep);
        }
#ifdef BLACKHOLES
        BHStep(dTime, dDeltaStep);
#endif
        UpdateRung(iRung);
    }

    msrprintf("%*cmsrKickOpen  at iRung: %d 0.5 * dDelta: %g\n",
              2 * iRung + 2, ' ', iRung, 0.5 * dDeltaRung);
    KickKDKOpen(dTime, 0.5 * dDeltaRung, iRung, iRung);
    /*
     ** Save fine - grained statistics, assuming that there is no 'sandwiched' rung,
     ** i.e., the differences in dt are smooth
     */
    if ( (nRung[0]!=0 && iRung == 0) ||
            ( (nRung[iRung] == 0) && (nRung[iRung + 1] > 0) ))
        OutputFineStatistics(dStep, dTime);

    if (CurrMaxRung() > iRung) {
        /*
        ** Recurse.
        */
        TopStepKDK(dStep, dTime, 0.5 * dDeltaRung, dTheta, iRung + 1, iRung + 1, 0);
        dTime += 0.5 * dDeltaRung;
        dStep += 1.0/(2 << iRung);

        ActiveRung(iRung, 0); /* is this call even needed? */

        TopStepKDK(dStep, dTime, 0.5 * dDeltaRung, dTheta, iRung + 1, iKickRung, 1);
    }
    else if (CurrMaxRung() == iRung) {
        if (DoGas() && MeshlessHydro()) {
            ActiveRung(iKickRung, 1);
            MeshlessFluxes(dTime, dDeltaStep);
        }

        ZeroNewRung(iKickRung, MAX_RUNG, iKickRung); /* brute force */
        /* This Drifts everybody */
        msrprintf("%*cDrift, iRung: %d\n", 2 * iRung + 2, ' ', iRung);
        Drift(dTime, dDeltaRung, ROOT);
        dTime += dDeltaRung;
        dStep += 1.0/(uintmax_t(1) << iRung);

#ifdef COOLING
        if (csm->val.bComove) {
            const float a = csmTime2Exp(csm, dTime);
            const float z = 1./a - 1.;

            CoolingUpdate(z);
        }
        else {
            CoolingUpdate(0.);
        }
#endif

        ActiveRung(iKickRung, 1);
        DomainDecomp(iKickRung);
        if (parameters.get_bAddDelete()) MoveDeletedParticles();
        BuildTree(bEwald);

#ifdef BLACKHOLES
#ifndef DEBUG_BH_ONLY
        auto bBHAccretion = parameters.get_bBHAccretion();
        if (bBHAccretion || parameters.get_bBHFeedback()) {
            BHEvolve(dTime, dDeltaStep);
        }
        if (bBHAccretion) {
            BHAccretion(dTime);
        }
#endif
        if (parameters.get_bBHMerger()) {
            SelActives();
            BHMerger(dTime);
        }
#endif

#ifdef STAR_FORMATION
        StarForm(dTime, dDeltaStep, iKickRung);
#endif

#if defined(OPTIM_REORDER_IN_NODES) && (defined(STAR_FORMATION) || defined(BLACKHOLES))
        ReorderWithinNodes();
#endif

        if (!iKickRung && parameters.get_bFindGroups()) {
            NewFof(parameters.get_dTau(), parameters.get_nMinMembers());
        }

        if (DoGravity() || DoGas()) {
            msrprintf("%*cForces, iRung: %d to %d\n", 2 * iRung + 2, ' ', iKickRung, iRung);
        }

        if (DoGravity()) {
            UpdateSoft(dTime);
            SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
            SPHoptions.doGravity = 1;
            Gravity(iKickRung, MAX_RUNG, ROOT, 0, dTime, dDeltaStep, dStep, dTheta, 0, 0,
                    bEwald, bGravStep, nPartRhoLoc, parameters.get_iTimeStepCrit(),
                    SPHoptions);
        }

#if defined(FEEDBACK) || defined(STELLAR_EVOLUTION)
        ActiveRung(iKickRung, 0);
        double dsec;
#ifdef FEEDBACK
        printf("Computing feedback... ");

        TimerStart(TIMER_FEEDBACK);
        if (parameters.get_bCCSNFeedback() || parameters.get_bSNIaFeedback()) {
            Smooth(dTime, dDeltaStep, SMX_SN_FEEDBACK, 1, parameters.get_nSmooth());
        }
        TimerStop(TIMER_FEEDBACK);
        dsec = TimerGet(TIMER_FEEDBACK);
        printf("took %.5f seconds\n", dsec);
#endif
#ifdef STELLAR_EVOLUTION
        printf("Computing stellar evolution... ");
        TimerStart(TIMER_STEV);
        if (parameters.get_bChemEnrich()) {
            Smooth(dTime, dDeltaStep, SMX_CHEM_ENRICHMENT, 1, parameters.get_nSmooth());
        }
        TimerStop(TIMER_STEV);
        dsec = TimerGet(TIMER_STEV);
        printf("took %.5f seconds\n", dsec);
#endif
        ActiveRung(iKickRung, 1);
#endif

        if (DoGas() && MeshlessHydro()) {
            EndTimestepIntegration(dTime, dDeltaStep);
            MeshlessGradients(dTime, dDeltaStep);
        }

        /*
         * move time back to 1 / 2 step so that KickClose can integrate
         * from 1 / 2 through the timestep to the end.
         */
        dTime -= 0.5 * dDeltaRung;
    }
    else {
        abort();
    }

    msrprintf("%*cKickClose, iRung: %d, 0.5 * dDelta: %g\n",
              2 * iRung + 2, ' ', iRung, 0.5 * dDeltaRung);
    KickKDKClose(dTime, 0.5 * dDeltaRung, iRung, iRung); /* uses dTime - 0.5 * dDelta */

    dTime += 0.5 * dDeltaRung; /* Important to have correct time at step end for SF! */

    if (!iKickRung && !iRung) {
        if (parameters.get_bFindGroups()) GroupStats();
        if (parameters.get_bAddDelete()) MoveDeletedParticles();
        BuildTree(bEwald);
    }

}

void MSR::MoveDeletedParticles() {
    struct outGetNParts Nout;

    Nout.n = 0;
    Nout.nDark = 0;
    Nout.nGas = 0;
    Nout.nStar = 0;
    Nout.nBH = 0;

    pstMoveDeletedParticles(pst, NULL, 0, &Nout, sizeof(struct outGetNParts));

    N = Nout.n;
    nDark = Nout.nDark;
    nGas = Nout.nGas;
    nStar = Nout.nStar;
    nBH = Nout.nBH;
}

void MSR::GetNParts() { /* JW: Not pretty -- may be better way via fio */
    struct outGetNParts outget;

    pstGetNParts(pst, NULL, 0, &outget, sizeof(outget));
    assert(outget.nGas == nGas);
    assert(outget.nDark == nDark);
    assert(outget.nStar == nStar);
    assert(outget.nBH == nBH);
    nMaxOrder = outget.nMaxOrder;
#if 0
    if (outget.iMaxOrderGas > nMaxOrder) {
        nMaxOrder = outget.iMaxOrderGas;
        fprintf(stderr, "WARNING: Largest iOrder of gas > Largest iOrder of star\n");
    }
    if (outget.iMaxOrderDark > nMaxOrder) {
        nMaxOrder = outget.iMaxOrderDark;
        fprintf(stderr, "WARNING: Largest iOrder of dark > Largest iOrder of star\n");
    }
#endif
}

void MSR::AddDelParticles() {
    struct inSetNParts in;
    int i;

    msrprintf("Changing Particle number\n");

    std::unique_ptr<struct outColNParts[]> pColNParts {new struct outColNParts[nThreads]};
    pstColNParts(pst, NULL, 0, pColNParts.get(), nThreads * sizeof(pColNParts[0]));
    /*
     * Assign starting numbers for new particles in each processor.
     */
    std::unique_ptr<uint64_t[]> pNewOrder {new uint64_t[nThreads]};
    for (i = 0; i < nThreads; i++) {
        /*
         * Detect any changes in particle number, and force a tree
         * build.
         */
        if (pColNParts[i].nNew != 0 || pColNParts[i].nDeltaGas != 0 ||
                pColNParts[i].nDeltaDark != 0 || pColNParts[i].nDeltaStar != 0) {
            /*printf("Particle assignments have changed!\n");
              printf("need to rebuild tree, code in msrAddDelParticles()\n");
              printf("needs to be updated. Bailing out for now...\n");
              exit(-1); */
            pNewOrder[i] = nMaxOrder + 1; /* JW: +1 was missing for some reason */
            nMaxOrder += pColNParts[i].nNew;
            nGas += pColNParts[i].nDeltaGas;
            nDark += pColNParts[i].nDeltaDark;
            nStar += pColNParts[i].nDeltaStar;
            nBH += pColNParts[i].nDeltaBH;
        }
    }
    N = nGas + nDark + nStar + nBH;

    /*nMaxOrderDark = nMaxOrder;*/

    pstNewOrder(pst, pNewOrder.get(), (int)sizeof(pNewOrder[0])*nThreads, NULL, 0);

    msrprintf("New numbers of particles: %" PRIu64 " gas %" PRIu64 " dark %" PRIu64 " star\n",
              nGas, nDark, nStar);

    in.nGas = nGas;
    in.nDark = nDark;
    in.nStar = nStar;
    in.nBH = nBH;
    pstSetNParts(pst, &in, sizeof(in), NULL, 0);
}

/* Gas routines */
void MSR::InitSph(double dTime, double dDelta, bool bRestart) {
    if (MeshlessHydro()) {
        if (!bRestart) {
            ActiveRung(0, 1);
            InitBall();
            EndTimestepIntegration(dTime, 0.0);
            MeshlessGradients(dTime, 0.0);
            MeshlessFluxes(dTime, 0.0);
            // We do this twice because we need to have uNewRung for the time
            // limiter of Durier & Dalla Vecchia
            HydroStep(dTime, dDelta);
            HydroStep(dTime, dDelta);
        }
    }
    UpdateRung(0) ;
}

void MSR::CoolSetup(double dTime) {
}

void MSR::Cooling(double dTime, double dStep, int bUpdateState, int bUpdateTable, int bIterateDt) {
}

void MSR::ChemCompInit() {
    struct inChemCompInit in;
    in.dInitialH = parameters.get_dInitialH();
#ifdef HAVE_HELIUM
    in.dInitialHe = parameters.get_dInitialHe();
#endif
#ifdef HAVE_CARBON
    in.dInitialC = parameters.get_dInitialC();
#endif
#ifdef HAVE_NITROGEN
    in.dInitialN = parameters.get_dInitialN();
#endif
#ifdef HAVE_OXYGEN
    in.dInitialO = parameters.get_dInitialO();
#endif
#ifdef HAVE_NEON
    in.dInitialNe = parameters.get_dInitialNe();
#endif
#ifdef HAVE_MAGNESIUM
    in.dInitialMg = parameters.get_dInitialMg();
#endif
#ifdef HAVE_SILICON
    in.dInitialSi = parameters.get_dInitialSi();
#endif
#ifdef HAVE_IRON
    in.dInitialFe = parameters.get_dInitialFe();
#endif
#ifdef HAVE_METALLICITY
    in.dInitialMetallicity = parameters.get_dInitialMetallicity();
#endif

    pstChemCompInit(pst, &in, sizeof(in), NULL, 0);
}

/* END Gas routines */

void MSR::HopWrite(const char *fname) {
    double dsec;

    if (parameters.get_bVStep())
        printf("Writing group statistics to %s\n", fname );
    TimerStart(TIMER_IO);

    /* This is the new parallel binary format */
    struct inOutput out;
    out.eOutputType = OUT_TINY_GROUP;
    out.iPartner = -1;
    out.nPartner = -1;
    out.iProcessor = 0;
    out.nProcessor = parallel_write_count();
    strcpy(out.achOutFile, fname);
    pstOutput(pst, &out, sizeof(out), NULL, 0);
    TimerStop(TIMER_IO);
    dsec = TimerGet(TIMER_IO);
    if (parameters.get_bVStep())
        printf("Written statistics, Wallclock: %f secs\n", dsec);

}

void MSR::Hop(double dTime, double dDelta) {
    struct inSmooth in;
    struct inHopLink h;
    struct outHopJoin j;
    struct inHopFinishUp inFinish;
    struct inGroupStats inGroupStats;
    int i;
    uint64_t nGroups;
    double sec, dsec, ssec;

    ssec = MSR::Time();

    h.nSmooth    = in.nSmooth = 80;
    h.bPeriodic  = in.bPeriodic = parameters.get_bPeriodic();
    h.bSymmetric = in.bSymmetric = 0;
    h.dHopTau    = parameters.get_dHopTau();
    h.smf.a      = in.smf.a = dTime;
    h.smf.dTau2  = in.smf.dTau2 = 0.0;
    h.smf.nMinMembers = in.smf.nMinMembers = parameters.get_nMinMembers();
    SmoothSetSMF(&(in.smf), dTime, dDelta, in.nSmooth);
    SmoothSetSMF(&(h.smf), dTime, dDelta, in.nSmooth);

    if (parameters.get_bVStep()) {
        if (h.dHopTau < 0.0)
            printf("Running Grasshopper with adaptive linking length (%g times softening)\n", -h.dHopTau );
        else
            printf("Running Grasshopper with fixed linking length %g\n", h.dHopTau );
    }

    in.iSmoothType = SMX_DENSITY_M3;
    sec = MSR::Time();
    pstSmooth(pst, &in, sizeof(in), NULL, 0);
    dsec = MSR::Time() - sec;
    if (parameters.get_bVStep())
        printf("Density calculation complete in %f secs, finding chains...\n", dsec);

    h.iSmoothType = SMX_GRADIENT_M3;
    sec = MSR::Time();
    nGroups = 0;
    pstHopLink(pst, &h, sizeof(h), &nGroups, sizeof(nGroups));
    dsec = MSR::Time() - sec;
    if (parameters.get_bVStep())
        printf("Chain search complete in %f secs, building minimal tree...\n", dsec);

    /* Build a new tree with only marked particles */
    sec = MSR::Time();
    BuildTreeMarked();
    dsec = MSR::Time() - sec;
    if (parameters.get_bVStep())
        printf("Tree build complete in %f secs, merging %" PRIu64 " chains...\n", dsec, nGroups);

    h.iSmoothType = SMX_HOP_LINK;
    sec = MSR::Time();
    i = 0;
    do {
        ++i;
        assert(i < 100);
        pstHopJoin(pst, &h, sizeof(h), &j, sizeof(j));
        if (parameters.get_bVStep())
            printf("... %d iteration%s, %" PRIu64 " chains remain\n", i, i == 1?"":"s", j.nGroups);
    } while (!j.bDone);
    nGroups = j.nGroups;
    dsec = MSR::Time() - sec;
    if (parameters.get_bVStep())
        printf("Chain merge complete in %f secs, %" PRIu64 " groups\n", dsec, nGroups);
    inFinish.nMinGroupSize = parameters.get_nMinMembers();
    inFinish.bPeriodic = parameters.get_bPeriodic();
    inFinish.fPeriod = parameters.get_dPeriod();
    pstHopFinishUp(pst, &inFinish, sizeof(inFinish), &nGroups, sizeof(nGroups));
    if (parameters.get_bVStep())
        printf("Removed groups with fewer than %d particles, %" PRIu64 " remain\n",
               inFinish.nMinGroupSize, nGroups);
#if 0
    if (parameters.get_bVStep())
        printf("Unbinding\n");

    struct inHopUnbind inUnbind;
    inUnbind.dTime = dTime;
    inUnbind.bPeriodic = parameters.get_bPeriodic();
    inUnbind.fPeriod = parameters.get_dPeriod();
    inUnbind.nMinGroupSize = parameters.get_nMinMembers();
    inUnbind.iIteration = 0;
    struct inHopGravity inGravity;
    inGravity.dTime = dTime;
    inGravity.bPeriodic = parameters.get_bPeriodic();
    inGravity.nGroup = parameters.get_nGroup();
    inGravity.dEwCut = parameters.get_dEwCut();
    inGravity.dEwhCut = parameters.get_dEwhCut();
    inGravity.uRungLo = 0;
    inGravity.uRungHi = MAX_RUNG;
    inGravity.dTheta = dThetaMin;

    inUnbind.iIteration = 0;
    do {
        sec = MSR::Time();
        struct inHopTreeBuild inTreeBuild;
        inTreeBuild.nBucket = parameters.get_nBucket();
        inTreeBuild.nGroup = parameters.get_nGroup();
        pstHopTreeBuild(pst, &inTreeBuild, sizeof(inTreeBuild), NULL, 0);
        dsec = MSR::Time() - sec;
        if (parameters.get_bVStep())
            printf("... group trees built, Wallclock: %f secs\n", dsec);

        sec = MSR::Time();
        pstHopGravity(pst, &inGravity, sizeof(inGravity), NULL, 0);
        dsec = MSR::Time() - sec;
        if (parameters.get_bVStep())
            printf("... gravity complete, Wallclock: %f secs\n", dsec);

        sec = MSR::Time();
        struct outHopUnbind outUnbind;
        pstHopUnbind(pst, &inUnbind, sizeof(inUnbind), &outUnbind, sizeof(outUnbind));
        nGroups = outUnbind.nGroups;
        dsec = MSR::Time() - sec;
        if (parameters.get_bVStep())
            printf("Unbinding completed in %f secs, %" PRIu64 " particles evaporated, %" PRIu64 " groups remain\n",
                   dsec, outUnbind.nEvaporated, nGroups);
    } while (++inUnbind.iIteration < 100 && outUnbind.nEvaporated);
#endif
    /*
    ** This should be done as a separate msr function.
    */
    inGroupStats.bPeriodic = parameters.get_bPeriodic();
    inGroupStats.dPeriod = parameters.get_dPeriod();
    inGroupStats.rEnvironment[0] = parameters.get_dEnvironment0();
    inGroupStats.rEnvironment[1] = parameters.get_dEnvironment1();
    inGroupStats.iGlobalStart = 1; /* global id 0 means ungrouped particle on all cpus */
    auto dBoxSize = parameters.get_dBoxSize();
    if ( parameters.has_dBoxSize() && dBoxSize > 0.0 ) {
        inGroupStats.rEnvironment[0] /= dBoxSize;
        inGroupStats.rEnvironment[1] /= dBoxSize;
    }
    pstGroupStats(pst, &inGroupStats, sizeof(inGroupStats), NULL, 0);

    dsec = MSR::Time() - ssec;
    if (parameters.get_bVStep())
        printf("Grasshopper complete, Wallclock: %f secs\n\n", dsec);
}

void MSR::NewFof(double dTau, int nMinMembers) {
    struct inNewFof in;
    struct outFofPhases out;
    struct inFofFinishUp inFinish;
    int i;
    uint64_t nGroups;
    double dsec;

    TimerStart(TIMER_FOF);

    in.dTau2 = dTau * dTau;
    in.nMinMembers = nMinMembers;
    in.bPeriodic = parameters.get_bPeriodic();
    in.nReplicas = in.bPeriodic ? parameters.get_nReplicas() : 0;
    in.nBucket = parameters.get_nBucket();

    if (parameters.get_bVStep()) {
        printf("Running FoF with fixed linking length %g\n", dTau );
    }

    TimerStart(TIMER_NONE);
    pstNewFof(pst, &in, sizeof(in), NULL, 0);

    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    if (parameters.get_bVStep())
        printf("Initial FoF calculation complete in %f secs\n", dsec);

    TimerStart(TIMER_NONE);
    i = 0;
    do {
        ++i;
        assert(i < 100);
        pstFofPhases(pst, NULL, 0, &out, sizeof(out));
        if (parameters.get_bVStep())
            printf("... %d iteration%s\n", i, i == 1?"":"s");
    } while (out.bMadeProgress);

    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    if (parameters.get_bVStep())
        printf("Global merge complete in %f secs\n", dsec);

    inFinish.nMinGroupSize = nMinMembers;
    pstFofFinishUp(pst, &inFinish, sizeof(inFinish), &nGroups, sizeof(nGroups));
    if (parameters.get_bVStep())
        printf("Removed groups with fewer than %d particles, %" PRIu64 " remain\n",
               inFinish.nMinGroupSize, nGroups);
    TimerStop(TIMER_FOF);
    dsec = TimerGet(TIMER_FOF);
    if (parameters.get_bVStep())
        printf("FoF complete, Wallclock: %f secs\n", dsec);
}

void MSR::GroupStats() {
    struct inGroupStats inGroupStats;
    double dsec;

    if (parameters.get_bVStep())
        printf("Generating Group statistics\n");
    TimerStart(TIMER_FOF);
    inGroupStats.bPeriodic = parameters.get_bPeriodic();
    inGroupStats.dPeriod = parameters.get_dPeriod();
    inGroupStats.rEnvironment[0] = parameters.get_dEnvironment0();
    inGroupStats.rEnvironment[1] = parameters.get_dEnvironment1();
    inGroupStats.iGlobalStart = 1; /* global id 0 means ungrouped particle on all cpus */
    auto dBoxSize = parameters.get_dBoxSize();
    if ( parameters.has_dBoxSize() && dBoxSize > 0.0 ) {
        inGroupStats.rEnvironment[0] /= dBoxSize;
        inGroupStats.rEnvironment[1] /= dBoxSize;
    }
    pstGroupStats(pst, &inGroupStats, sizeof(inGroupStats), NULL, 0);
    TimerStop(TIMER_FOF);
    dsec = TimerGet(TIMER_FOF);
    if (parameters.get_bVStep())
        printf("Group statistics complete, Wallclock: %f secs\n\n", dsec);
}

#ifdef MDL_FFTW
double MSR::GenerateIC(int nGrid, int iSeed, double z, double L, CSM csm) {
    struct inGenerateIC in;
    struct outGenerateIC out;
    ServiceFftSizes::input inFFTSizes;
    ServiceFftSizes::output outFFTSizes;
    fioSpeciesList nSpecies;
    double sec, dsec;
    double mean, rms;
    uint64_t nTotal;
    int j;

    if (csm) {
        this->csm->val = csm->val;
        if (this->csm->val.classData.bClass)
            csmClassGslInitialize(this->csm);
    }
    else csm = this->csm;

    // We only support periodic initial conditions
    parameters.set_bPeriodic(true);
    parameters.set_bComove(true);

    parameters.set_iSeed(iSeed);
    parameters.set_dRedFrom(z);
    parameters.set_dBoxSize(L);
    parameters.set_nGrid(nGrid);

    in.dBoxSize = L;
    in.iSeed = iSeed;
    in.bFixed = parameters.get_bFixedAmpIC();
    in.fPhase = parameters.get_dFixedAmpPhasePI() * M_PI;
    in.nGrid = nGrid;
    in.b2LPT = parameters.get_b2LPT();
    in.bICgas = parameters.get_bICgas();
    in.nBucket = parameters.get_nBucket();
    in.dInitialT = parameters.get_dInitialT();
    in.dInitialH = parameters.get_dInitialH();
#ifdef HAVE_HELIUM
    in.dInitialHe = parameters.get_dInitialHe();
#endif
#ifdef HAVE_CARBON
    in.dInitialC = parameters.get_dInitialC();
#endif
#ifdef HAVE_NITROGEN
    in.dInitialN = parameters.get_dInitialN();
#endif
#ifdef HAVE_OXYGEN
    in.dInitialO = parameters.get_dInitialO();
#endif
#ifdef HAVE_NEON
    in.dInitialNe = parameters.get_dInitialNe();
#endif
#ifdef HAVE_MAGNESIUM
    in.dInitialMg = parameters.get_dInitialMg();
#endif
#ifdef HAVE_SILICON
    in.dInitialSi = parameters.get_dInitialSi();
#endif
#ifdef HAVE_IRON
    in.dInitialFe = parameters.get_dInitialFe();
#endif
#ifdef HAVE_METALLICITY
    in.dInitialMetallicity = parameters.get_dInitialMetallicity();
#endif

    nTotal  = in.nGrid; /* Careful: 32 bit integer cubed => 64 bit integer */
    nTotal *= in.nGrid;
    nTotal *= in.nGrid;
    in.dBoxMass = csm->val.dOmega0 / nTotal;
    if (in.bICgas) nTotal *= 2;

    for ( j = 0; j <= FIO_SPECIES_LAST; j++) nSpecies[j] = 0;
    if (in.bICgas) {
        nSpecies[FIO_SPECIES_ALL] = nTotal;
        nSpecies[FIO_SPECIES_SPH] = nTotal / 2;
        nSpecies[FIO_SPECIES_DARK]= nTotal / 2;
    }
    else {
        nSpecies[FIO_SPECIES_ALL] = nSpecies[FIO_SPECIES_DARK] = nTotal;
    }
    InitializePStore(nSpecies, getMemoryModel(), parameters.get_nMemEphemeral()); // We now need a bit of cosmology to set the maximum lightcone depth here.
    InitCosmology(csm);

    in.dBaryonFraction = csm->val.dOmegab / csm->val.dOmega0;
    SetDerivedParameters();
    in.dTuFac = dTuFacPrimNeutral;

    assert(z >= 0.0 );
    in.dExpansion = 1.0 / (1.0 + z);

    N = nSpecies[FIO_SPECIES_ALL];
    nGas = nSpecies[FIO_SPECIES_SPH];
    nDark = nSpecies[FIO_SPECIES_DARK];
    nStar = nSpecies[FIO_SPECIES_STAR];
    nBH = nSpecies[FIO_SPECIES_BH];
    nMaxOrder = N - 1; // iOrder goes from 0 to N - 1

    if (parameters.get_bVStart())
        printf("Generating IC...\nN:%" PRIu64 " nDark:%" PRIu64
               " nGas:%" PRIu64 " nStar:%" PRIu64 "\n",
               N, nDark, nGas, nStar);

    /* Read the transfer function */
    in.nTf = 0;
    if (parameters.has_achTfFile()) {
        auto achTfFile = parameters.get_achTfFile();
        FILE *fp = fopen(achTfFile.data(), "r");
        char buffer[256];

        if (parameters.get_bVStart())
            printf("Reading transfer function from %s\n", achTfFile.data());
        if (fp == NULL) {
            perror(achTfFile.data());
            Exit(1);
        }
        while (fgets(buffer, sizeof(buffer), fp)) {
            assert(in.nTf < MAX_TF);
            if (sscanf(buffer, " %lg %lg\n", &in.k[in.nTf], &in.tf[in.nTf])==2) {
                in.k[in.nTf] = log(in.k[in.nTf]);
                in.tf[in.nTf] = log(in.tf[in.nTf]);
                ++in.nTf;
            }
        }
        fclose(fp);
        if (parameters.get_bVStart())
            printf("Transfer function : %d lines kmin %g kmax %g\n",
                   in.nTf, exp(in.k[0]), exp(in.k[in.nTf - 1]));

    }

    sec = MSR::Time();

    /* Figure out the minimum number of particles */
    inFFTSizes.nx = inFFTSizes.ny = inFFTSizes.nz = in.nGrid;
    mdl->RunService(PST_GETFFTMAXSIZES, sizeof(inFFTSizes), &inFFTSizes, &outFFTSizes);
    printf("Grid size %d x %d x %d, per node %d x %d x %d and %d x %d x %d\n",
           inFFTSizes.nx, inFFTSizes.ny, inFFTSizes.nz,
           inFFTSizes.nx, inFFTSizes.ny, outFFTSizes.nMaxZ,
           inFFTSizes.nx, outFFTSizes.nMaxY, inFFTSizes.nz);

    msrprintf("IC Generation @ a=%g with seed %d\n", in.dExpansion, iSeed);
    in.nPerNode = outFFTSizes.nMaxLocal;
    pstGenerateIC(pst, &in, sizeof(in), &out, sizeof(out));
    mean = 2 * out.noiseMean / N;
    rms = sqrt(2 * out.noiseCSQ / N);

    msrprintf("Transferring particles between / within nodes\n");
    pstMoveIC(pst, &in, sizeof(in), NULL, 0);

    SetClasses();
    dsec = MSR::Time() - sec;
    msrprintf("IC Generation Complete @ a=%g, Wallclock: %f secs\n\n", out.dExpansion, dsec);
    msrprintf("Mean of noise same is %g, RMS %g.\n", mean, rms);

    return getTime(out.dExpansion);
}
#endif

double MSR::Read(std::string_view achInFile) {
    double dTime, dExpansion;
    FIO fio;
    int j;
    double dsec;
    fioSpeciesList nSpecies;
    inReadFileFilename achFilename;
    uint64_t mMemoryModel = 0;

    mMemoryModel = getMemoryModel();

    TimerStart(TIMER_NONE);

    auto nBytes = PST_MAX_FILES*(sizeof(fioSpeciesList)+PST_FILENAME_SIZE);
    std::unique_ptr<char[]> buffer {new char[sizeof(inReadFile) + nBytes]};
    auto read = new (buffer.get()) inReadFile;

    /* Add Data Subpath for local and non-local names. */
    MSR::MakePath(parameters.get_achDataSubPath(), achInFile.data(), achFilename);
    fio = fioOpen(achFilename, csm->val.dOmega0, csm->val.dOmegab);
    if (fio == NULL) {
        fprintf(stderr, "ERROR: unable to open input file\n");
        perror(achFilename);
        Exit(1);
    }
    nBytes = fioDump(fio, nBytes, read + 1);

    // If we have the 'Redshift' field, we take that.
    // If not, we assume that the 'Time' field contains the expansion factor
    if (!fioGetAttr(fio, HDF5_HEADER_G, "Redshift", FIO_TYPE_DOUBLE, &dExpansion)) {
        if (!fioGetAttr(fio, HDF5_HEADER_G, "Time", FIO_TYPE_DOUBLE, &dExpansion))
            dExpansion = 0.0;
    }
    else {
        dExpansion = 1.0/(dExpansion + 1.0);
    }
    if (!fioGetAttr(fio, HDF5_HEADER_G, "dEcosmo", FIO_TYPE_DOUBLE, &dEcosmo)) dEcosmo = 0.0;
    if (!fioGetAttr(fio, HDF5_HEADER_G, "dTimeOld", FIO_TYPE_DOUBLE, &dTimeOld)) dTimeOld = 0.0;
    if (!fioGetAttr(fio, HDF5_HEADER_G, "dUOld", FIO_TYPE_DOUBLE, &dUOld)) dUOld = 0.0;

    if (csm->val.bComove) {
        if (!parameters.has_dOmega0())
            fioGetAttr(fio, HDF5_COSMO_G, "Omega_m", FIO_TYPE_DOUBLE, &csm->val.dOmega0);
        if (!parameters.has_dLambda())
            fioGetAttr(fio, HDF5_COSMO_G, "Omega_Lambda", FIO_TYPE_DOUBLE, &csm->val.dLambda);
        if (!parameters.has_dBoxSize()) {
            double dBoxSize;
            fioGetAttr(fio, HDF5_HEADER_G, "BoxSize", FIO_TYPE_DOUBLE, &dBoxSize);
            parameters.set_dBoxSize(dBoxSize);
        }
        if (!parameters.has_h())
            fioGetAttr(fio, HDF5_COSMO_G, "HubbleParam", FIO_TYPE_DOUBLE, &csm->val.h);
    }

    N     = fioGetN(fio, FIO_SPECIES_ALL);
    nGas  = fioGetN(fio, FIO_SPECIES_SPH);
    nDark = fioGetN(fio, FIO_SPECIES_DARK);
    nStar = fioGetN(fio, FIO_SPECIES_STAR);
    nBH   = fioGetN(fio, FIO_SPECIES_BH);
    nMaxOrder = N - 1; // iOrder goes from 0 to N - 1

    read->nProcessors = parallel_read_count();

    if (!fioGetAttr(fio, HDF5_HEADER_G, "NumFilesPerSnapshot", FIO_TYPE_UINT32, &j)) j = 1;
    printf("Reading %" PRIu64 " particles from %d file%s using %d processor%s\n",
           N, j, (j == 1?"":"s"), read->nProcessors, (read->nProcessors == 1?"":"s") );

    dTime = getTime(dExpansion);
    if (parameters.get_bInFileLC()) read->dvFac = 1.0;
    else read->dvFac = getVfactor(dExpansion);

    if (nGas && !DoGas()) parameters.set_hydro_model(HYDRO_MODEL::SPH);
    if (NewSPH()) mMemoryModel |= (PKD_MODEL_NEW_SPH | PKD_MODEL_ACCELERATION | PKD_MODEL_VELOCITY | PKD_MODEL_DENSITY | PKD_MODEL_BALL | PKD_MODEL_NODE_BOB);
    if (nStar) mMemoryModel |= PKD_MODEL_STAR;

    read->nNodeStart = 0;
    read->nNodeEnd = N - 1;

    for ( auto s = FIO_SPECIES(0); s <= FIO_SPECIES_LAST; s = FIO_SPECIES(s + 1)) nSpecies[s] = fioGetN(fio, s);
    InitializePStore(nSpecies, mMemoryModel, parameters.get_nMemEphemeral());

    read->dOmega0 = csm->val.dOmega0;
    read->dOmegab = csm->val.dOmegab;

    SetDerivedParameters();
    read->dTuFac = dTuFac;

    if (read->nProcessors > 1) {
        fioClose(fio);
        pstReadFile(pst, read, sizeof(struct inReadFile)+nBytes, NULL, 0);
    }
    else {
        OneNodeRead(read, fio);
        fioClose(fio);
    }

    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    SetClasses();
    printf("Input file has been successfully read, Wallclock: %f secs.\n", dsec);

    /*
    ** If this is a non-periodic box, then we must precalculate the bounds.
    ** We throw away the result, but PKD will keep track for later.
    */
    auto period = parameters.get_dPeriod();
    if (!parameters.get_bPeriodic() || blitz::any(period >= FLOAT_MAXVAL)) {
        CalcBound();
    }

    InitCosmology(csm);

    if (NewSPH()) {
        const auto bEwald = parameters.get_bEwald();
        /*
        ** Initialize kernel target with either the mean mass or nSmooth
        */
        TimerStart(TIMER_NONE);
        printf("Initializing Kernel target ...\n");
        {
            SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
            if (SPHoptions.useNumDen) {
                parameters.set_fKernelTarget(parameters.get_nSmooth());
            }
            else {
                double Mtot;
                uint64_t Ntot;
                CalcMtot(&Mtot, &Ntot);
                parameters.set_fKernelTarget(Mtot / Ntot * parameters.get_nSmooth());
            }
        }
        TimerStop(TIMER_NONE);
        dsec = TimerGet(TIMER_NONE);
        printf("Initializing Kernel target complete, Wallclock: %f secs.\n", dsec);

        SetSPHoptions();
        InitializeEOS();

        if (parameters.has_dSoft()) SetSoft(Soft());
        /*
        ** Initialize fBall
        */
        TimerStart(TIMER_NONE);
        printf("Initializing fBall ...\n");
        Reorder();
        ActiveRung(0, 1); /* Activate all particles */
        DomainDecomp(-1);
        BuildTree(bEwald);
        Smooth(dTime, 0.0f, SMX_BALL, 0, 2 * parameters.get_nSmooth());
        Reorder();
        TimerStop(TIMER_NONE);
        dsec = TimerGet(TIMER_NONE);
        printf("Initializing fBall complete, Wallclock: %f secs.\n", dsec);

        /*
        ** Convert U
        */
        TimerStart(TIMER_NONE);
        printf("Converting u ...\n");
        ActiveRung(0, 1); /* Activate all particles */
        DomainDecomp(-1);
        BuildTree(bEwald);

        const auto iStartStep = parameters.get_iStartStep();
        auto dTheta = set_dynamic(iStartStep, dTime);

        // Calculate Density
        SPHOptions SPHoptions = initializeSPHOptions(parameters, csm, dTime);
        SPHoptions.doDensity = 1;
        SPHoptions.doUConversion = 1;
        const auto bGravStep = parameters.get_bGravStep();
        const auto nPartRhoLoc = parameters.get_nPartRhoLoc();
        const auto iTimeStepCrit = parameters.get_iTimeStepCrit();
        Gravity(0, MAX_RUNG, ROOT, 0, dTime, 0.0f, iStartStep, dTheta, 0, 1,
                bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
        MemStatus();
        if (SPHoptions.doInterfaceCorrection) {
            SPHoptions.doDensity = 0;
            SPHoptions.doDensityCorrection = 1;
            SPHoptions.dofBallFactor = 0;
            TreeUpdateFlagBounds(bEwald, ROOT, 0, SPHoptions);
            Gravity(0, MAX_RUNG, ROOT, 0, dTime, 0.0f, iStartStep, dTheta, 0, 1,
                    bEwald, bGravStep, nPartRhoLoc, iTimeStepCrit, SPHoptions);
            UpdateGasValues(0, dTime, 0.0f, iStartStep, 0, 1, SPHoptions);
        }
        TimerStop(TIMER_NONE);
        dsec = TimerGet(TIMER_NONE);
        printf("Converting u complete, Wallclock: %f secs.\n", dsec);
        if (parameters.get_bWriteIC() || (parameters.get_nSteps() == 0)) {
            printf("Writing updated IC ...\n");
            TimerStart(TIMER_NONE);
            Write(BuildIoName(0).c_str(), 0.0, 0);
            TimerStop(TIMER_NONE);
            dsec = TimerGet(TIMER_NONE);
            printf("Finished writing updated IC, Wallclock: %f secs.\n", dsec);
        }
        if (parameters.get_nSteps() == 0) exit(0);
    }

    return dTime;
}

// This sets the local pkd->bnd.
void MSR::CalcBound(Bound &bnd) {
    mdl->RunService(PST_CALCBOUND, &bnd);
}
void MSR::CalcBound() {
    Bound bnd;
    CalcBound(bnd);
}

void MSR::OutputGrid(const char *filename, bool k, int iGrid, int nParaWrite) {
    struct inOutput out;
    double dsec, sec = MSR::Time();
    out.eOutputType = k ? OUT_KGRID : OUT_RGRID;
    out.iGrid = iGrid;
    out.iPartner = -1;
    out.nPartner = -1;
    out.iProcessor = 0;
    out.nProcessor = nParaWrite > mdlProcs(mdl) ? mdlProcs(mdl) : nParaWrite;
    strcpy(out.achOutFile, filename);
    printf("Writing grid to %s ...\n", out.achOutFile);
    pstOutput(pst, &out, sizeof(out), NULL, 0);
    dsec = MSR::Time() - sec;
    msrprintf("Grid has been successfully written, Wallclock: %f secs.\n\n", dsec);
}

#ifdef MDL_FFTW
void MSR::OutputPk(int iStep, double dTime) {
    double a, z, vfact, kfact;
    std::string filename;
    int i;

    auto nGridPk = parameters.get_nGridPk();
    if (nGridPk == 0) return;

    auto nBinsPk = parameters.get_nBinsPk();

    if (!csm->val.bComove) a = 1.0;
    else a = csmTime2Exp(csm, dTime);

    auto [nPk, fK, fPk, fPkAll] = MeasurePk(parameters.get_iPkOrder(), parameters.get_bPkInterlace(), nGridPk, a, nBinsPk);

    /* If the Box Size (in mpc / h) was specified, then we can scale the output power spectrum measurement */
    if ( parameters.has_dBoxSize() && parameters.get_dBoxSize() > 0.0 ) kfact = parameters.get_dBoxSize();
    else kfact = 1.0;
    vfact = kfact * kfact * kfact;
    kfact = 1.0 / kfact;

    filename = BuildName(iStep, ".pk");
    std::ofstream fs(filename);
    if (fs.fail()) {
        std::cerr << "Could not create P(k) file: " << filename << std::endl;
        perror(filename.c_str());
        Exit(errno);
    }
    fmt::print(fs, "# k P(k) N(k) P(k)+{linear}\n", "linear"_a = parameters.get_achPkSpecies());
    fmt::print(fs, "# a={a:.8f}  z={z:.8f}\n", "a"_a = a, "z"_a = 1 / a - 1.0 );
    for (i = 0; i < nBinsPk; ++i) {
        if (fPk[i] > 0.0) fmt::print(fs, "{k:.8e} {pk:.8e} {nk} {all:.8e}\n",
                                         "k"_a   = kfact * fK[i] * 2.0 * M_PI,
                                         "pk"_a  = vfact * fPk[i],
                                         "nk"_a  = nPk[i],
                                         "all"_a = vfact * fPkAll[i]);
    }
    fs.close();
    /* Output the k - grid if requested */
    z = 1 / a - 1;
    auto iDeltakInterval = parameters.get_iDeltakInterval();
    auto dDeltakRedshift = parameters.get_dDeltakRedshift();
    if (iDeltakInterval && (iStep % iDeltakInterval == 0) && z < dDeltakRedshift) {
        auto filename = BuildName(iStep, ".deltak");
        OutputGrid(filename.c_str(), true, 0, parallel_write_count());
    }
}

void MSR::OutputLinPk(int iStep, double dTime) {
    std::string filename;
    double a, vfact, kfact;
    int i;

    if (parameters.get_nGridLin() == 0) return;
    if (!csm->val.bComove) return;
    if (!parameters.has_dBoxSize()) return;

    a = csmTime2Exp(csm, dTime);

    auto [nPk, fK, fPk] = MeasureLinPk(parameters.get_nGridLin(), a, parameters.get_dBoxSize());

    if (!csm->val.bComove) a = 1.0;
    else a = csmTime2Exp(csm, dTime);

    if ( parameters.get_dBoxSize() > 0.0 ) kfact = parameters.get_dBoxSize();
    else kfact = 1.0;
    vfact = kfact * kfact * kfact;
    kfact = 1.0 / kfact;

    filename = BuildName(iStep, ".lin_pk");
    std::ofstream fs(filename);
    if (fs.fail()) {
        std::cerr << "Could not create P_lin(k) file: " << filename << std::endl;
        perror(filename.c_str());
        Exit(errno);
    }
    for (i = 0; i < fK.size(); ++i) {
        if (fPk[i] > 0.0) fmt::print(fs, "{k} {pk} {nk}\n",
                                         "k"_a   = kfact * fK[i] * 2.0 * M_PI,
                                         "pk"_a  = vfact * fPk[i],
                                         "nk"_a  = nPk[i]);
    }
    fs.close();
}

#endif

/*
 **  This routine will output all requested files and fields
 */

void MSR::Output(int iStep, double dTime, double dDelta, int bCheckpoint) {
    int bSymmetric;

    // IA: If we allow for adding / deleting particles, we need to recount them to have the
    //  correct number of particles per specie
    if (parameters.get_bAddDelete()) GetNParts();

    printf("Writing output for step %d\n", iStep);

    Write(BuildIoName(iStep).c_str(), dTime, bCheckpoint );

    if (DoGas() && !parameters.get_nSteps()) {  /* Diagnostic Gas */
        Reorder();
        OutArray(BuildName(iStep, ".c").c_str(), OUT_C_ARRAY);
        OutArray(BuildName(iStep, ".hsph").c_str(), OUT_HSPH_ARRAY);
    }

    if (DoDensity() && !NewSPH()) {
        ActiveRung(0, 1); /* Activate all particles */
        DomainDecomp(-1);
        BuildTree(0);
        bSymmetric = 0;  /* should be set in param file! */
        Smooth(dTime, dDelta, SMX_DENSITY, bSymmetric, parameters.get_nSmooth());
    }
    if ( parameters.get_bFindGroups() ) {
        Reorder();
        //sprintf(achFile, "%s.fof", OutName());
        //OutArray(achFile, OUT_GROUP_ARRAY);
        HopWrite(BuildName(iStep, ".fofstats").c_str());
    }
    if ( parameters.get_bFindHopGroups() ) {
        ActiveRung(0, 1); /* Activate all particles */
        DomainDecomp(-1);
        BuildTree(0);
        Hop(dTime, dDelta);
        Reorder();
        //OutArray(BuildName(iStep, ".hopgrp").c_str(), OUT_GROUP_ARRAY);
        HopWrite(BuildName(iStep, ".hopstats").c_str());
    }

    if (parameters.get_bDoAccOutput()) {
        Reorder();
        OutVector(BuildName(iStep, ".acc").c_str(), OUT_ACCEL_VECTOR);
    }
    if (parameters.get_bDoPotOutput()) {
        Reorder();
        OutArray(BuildName(iStep, ".pot").c_str(), OUT_POT_ARRAY);
    }

    if (DoDensity() && !NewSPH()) {
        Reorder();
        OutArray(BuildName(iStep, ".den").c_str(), OUT_DENSITY_ARRAY);
    }
    if (parameters.get_bDoRungOutput()) {
        Reorder();
        OutArray(BuildName(iStep, ".rung").c_str(), OUT_RUNG_ARRAY);
    }
    if (parameters.get_bDoRungDestOutput()) {
        Reorder();
        OutArray(BuildName(iStep, ".rd").c_str(), OUT_RUNGDEST_ARRAY);
    }
    if (parameters.get_bDoSoftOutput()) {
        Reorder();
        OutArray(BuildName(iStep, ".soft").c_str(), OUT_SOFT_ARRAY);
    }
}

uint64_t MSR::CountSelected() {
    uint64_t N;
    mdl->RunService(PST_COUNTSELECTED, &N);
    return N;
}
uint64_t MSR::SelSpecies(uint64_t mSpecies, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelSpecies::input in(mSpecies, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELSPECIES, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelAll(int setIfTrue, int clearIfFalse) {
    return SelSpecies(1<<FIO_SPECIES_ALL, setIfTrue, clearIfFalse);
}
uint64_t MSR::SelGas(int setIfTrue, int clearIfFalse) {
    return SelSpecies(1<<FIO_SPECIES_SPH, setIfTrue, clearIfFalse);
}
uint64_t MSR::SelStar(int setIfTrue, int clearIfFalse) {
    return SelSpecies(1<<FIO_SPECIES_STAR, setIfTrue, clearIfFalse);
}
uint64_t MSR::SelDark(int setIfTrue, int clearIfFalse) {
    return SelSpecies(1<<FIO_SPECIES_DARK, setIfTrue, clearIfFalse);
}
uint64_t MSR::SelDeleted(int setIfTrue, int clearIfFalse) {
    // The "UNKNWON" species marks deleted particles
    return SelSpecies(1<<FIO_SPECIES_UNKNOWN, setIfTrue, clearIfFalse);
}
uint64_t MSR::SelActives(int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelActives::input in(setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELACTIVES, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelBlackholes(int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelBlackholes::input in(setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELBLACKHOLES, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelGroup(int iGroup, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelGroup::input in(iGroup, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELGROUP, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelById(uint64_t idStart, uint64_t idEnd, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelById::input in(idStart, idEnd, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELBYID, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelMass(double dMinMass, double dMaxMass, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelMass::input in(dMinMass, dMaxMass, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELMASS, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelPhaseDensity(double dMinPhaseDensity, double dMaxPhaseDensity, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelPhaseDensity::input in(dMinPhaseDensity, dMaxPhaseDensity, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELPHASEDENSITY, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelBox(blitz::TinyVector<double, 3> center, blitz::TinyVector<double, 3> size, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelBox::input in(center, size, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELBOX, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelSphere(blitz::TinyVector<double, 3> r, double dRadius, int setIfTrue, int clearIfFalse) {
    uint64_t N;
    ServiceSelSphere::input in(r, dRadius, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELSPHERE, sizeof(in), &in, &N);
    return N;
}
uint64_t MSR::SelCylinder(blitz::TinyVector<double, 3> dP1, blitz::TinyVector<double, 3> dP2, double dRadius,
                          int setIfTrue, int clearIfFalse ) {
    uint64_t N;
    ServiceSelCylinder::input in(dP1, dP2, dRadius, setIfTrue, clearIfFalse);
    mdl->RunService(PST_SELCYLINDER, sizeof(in), &in, &N);
    return N;
}

double MSR::TotalMass() {
    struct outTotalMass out;
    pstTotalMass(pst, NULL, 0, &out, sizeof(out));
    return out.dMass;
}

void MSR::CalcDistance(const double *dCenter, double dRadius ) {
    struct inCalcDistance in;
    int j;

    for (j = 0; j < 3; j++) in.dCenter[j] = dCenter[j];
    in.dRadius = dRadius;
    in.bPeriodic = parameters.get_bPeriodic();
    pstCalcDistance(pst, &in, sizeof(in), NULL, 0);
}

void MSR::CalcCOM(const double *dCenter, double dRadius,
                  double *com, double *vcm, double *L, double *M) {
    struct inCalcCOM in;
    struct outCalcCOM out;
    int nOut;
    int j;
    double T[3];

    for (j = 0; j < 3; j++) in.dCenter[j] = dCenter[j];
    in.dRadius = dRadius;
    in.bPeriodic = parameters.get_bPeriodic();
    nOut = pstCalcCOM(pst, &in, sizeof(in), &out, sizeof(out));
    assert( nOut == sizeof(out) );

    *M = out.M;
    if (out.M > 0.0) {
        for (j = 0; j < 3; j++) {
            com[j] = out.com[j] / out.M;
            vcm[j] = out.vcm[j] / out.M;
        }
        cross_product(T, com, vcm);
        vec_add_const_mult(L, out.L, -out.M, T);
        for (j = 0; j < 3; j++) L[j] /= out.M;
    }
}

void MSR::CalcMtot(double *M, uint64_t *N) {
    struct inCalcMtot in;
    struct outCalcMtot out;
    int nOut;

    nOut = pstCalcMtot(pst, &in, sizeof(in), &out, sizeof(out));
    assert( nOut == sizeof(out) );

    *M = out.M;
    *N = out.N;
}

void MSR::SetSPHoptions() {
    struct inSetSPHoptions in;
    in.SPHoptions = initializeSPHOptions(parameters, csm, 1.0);
    pstSetSPHoptions(pst, &in, sizeof(in), NULL, 0);
}

void MSR::ResetCOM() {
    blitz::TinyVector<double, 3> dCenter(0.0), com, vcm, L;
    double M;
    CalcCOM(&dCenter[0], -1.0, &com[0], &vcm[0], &L[0], &M);
    printf("Before reseting: x_com = %.5e, y_com = %.5e, z_com = %.5e, vx_com = %.5e, vy_com = %.5e, vz_com = %.5e\n", com[0], com[1], com[2], vcm[0], vcm[1], vcm[2]);

    struct inResetCOM in;
    in.r_com = com;
    in.v_com = vcm;

    pstResetCOM(pst, &in, sizeof(in), NULL, 0);

    CalcCOM(&dCenter[0], -1.0, &com[0], &vcm[0], &L[0], &M);
    printf("After reseting: x_com = %.5e, y_com = %.5e, z_com = %.5e, vx_com = %.5e, vy_com = %.5e, vz_com = %.5e\n", com[0], com[1], com[2], vcm[0], vcm[1], vcm[2]);
}

void MSR::InitializeEOS() {
    double sec, dsec;
    sec = MSR::Time();
    printf("Initialize EOS ...\n");
    pstInitializeEOS(pst, NULL, 0, NULL, 0);
    dsec = MSR::Time() - sec;
    printf("EOS initialized, Wallclock: %f secs\n\n", dsec);
}

void MSR::CalculateKickParameters(struct pkdKickParameters *kick, uint8_t uRungLo, double dTime, double dDelta, double dStep,
                                  int bKickClose, int bKickOpen, SPHOptions SPHoptions) {
    uint8_t uRungLoTemp;
    int i;
    double dt;
    if (SPHoptions.doDensity || SPHoptions.doDensityCorrection) {
        uRungLoTemp = uRungLo;
        uRungLo = SPHoptions.nPredictRung;
    }

    /*
    ** Now calculate the timestepping factors for kick close and open if the
    ** gravity should kick the particles. If the code uses bKickClose and
    ** bKickOpen it no longer needs to store accelerations per particle.
    */
    kick->bKickClose = bKickClose;
    kick->bKickOpen = bKickOpen;
    if (SPHoptions.doGravity || SPHoptions.doSPHForces || SPHoptions.doDensity || SPHoptions.doDensityCorrection) {
        for (i = 0, dt = 0.5 * dDelta; i <= parameters.get_iMaxRung(); ++i, dt *= 0.5) {
            kick->dtClose[i] = 0.0;
            kick->dtOpen[i] = 0.0;
            if (i >= uRungLo) {
                if (csm->val.bComove) {
                    if (bKickClose) {
                        kick->dtClose[i] = csmComoveKickFac(csm, dTime - dt, dt);
                    }
                    if (bKickOpen) {
                        kick->dtOpen[i] = csmComoveKickFac(csm, dTime, dt);
                    }
                }
                else {
                    if (bKickClose) kick->dtClose[i] = dt;
                    if (bKickOpen) kick->dtOpen[i] = dt;
                }
            }
        }
    }

    /*
    ** Create the deltas for the on - the - fly prediction of velocity and the
    ** thermodynamical variable.
    */
    if (SPHoptions.doSPHForces || SPHoptions.doDensity || SPHoptions.doDensityCorrection) {
        double substepWeAreAt = dStep - floor(dStep); // use fmod instead
        double stepStartTime = dTime - substepWeAreAt * dDelta;
        for (i = 0; i <= parameters.get_iMaxRung(); ++i) {
            if (i < uRungLo) {
                /*
                ** For particles with a step larger than the current rung, the temporal position of
                ** the velocity in relation to the current time is nontrivial, so we calculate it here
                */
                double substepSize = 1.0 / (uintmax_t(1) << i); // 1.0 / (1 << i);
                double substepsDoneAtThisSize = floor(substepWeAreAt / substepSize);
                double TPredDrift = stepStartTime + (substepsDoneAtThisSize + 0.5) * substepSize * dDelta;
                double dtPredDrift = dTime - TPredDrift;
                /* Now that we know how much we have to drift, we can calculate the corresponding
                ** drift factor
                */
                if (csm->val.bComove) {
                    /*
                    ** This gives the correct result, even if dtPredDrift is negative
                    ** but we still may want to use
                    ** -csmComoveKickFac(csm, TPredDrift + dtPredDrift, -dtPredDrift);
                    ** if dtPredDrift is negative, just to be sure
                    */
                    kick->dtPredDrift[i] = csmComoveKickFac(csm, TPredDrift, dtPredDrift);
                }
                else {
                    kick->dtPredDrift[i] = dtPredDrift;
                }
            }
            else {
                /*
                ** In this case, all particles are synchronized, which means that
                ** velocity and the thermodynamical variable are either a half step behind
                ** or ahead, so all information is contained in dtOpen and dtClose and the
                ** bMarked flag.
                */
                kick->dtPredDrift[i] = 0.0;
            }
        }
    }

    /*
    ** Create the deltas for the on - the - fly prediction in case of ISPH
    */
    if (NewSPH() && (SPHoptions.doSPHForces || SPHoptions.doDensity || SPHoptions.doDensityCorrection) && SPHoptions.useIsentropic) {
        double substepWeAreAt = dStep - floor(dStep); // use fmod instead
        double stepStartTime = dTime - substepWeAreAt * dDelta;
        for (i = 0; i <= parameters.get_iMaxRung(); ++i) {
            double substepSize = 1.0 / (uintmax_t(1) << i); // 1.0 / (1 << i);
            double substepsDoneAtThisSize = floor(substepWeAreAt / substepSize);
            double TSubStepStart, TSubStepKicked;
            /* The start of the step is different if the time step is larger than the current */
            if (i < uRungLo) {
                TSubStepStart = stepStartTime + (substepsDoneAtThisSize) * substepSize * dDelta;
                TSubStepKicked = stepStartTime + (substepsDoneAtThisSize + 0.5) * substepSize * dDelta;
            }
            else {
                TSubStepStart = stepStartTime + (substepsDoneAtThisSize - 1.0) * substepSize * dDelta;
                TSubStepKicked = stepStartTime + (substepsDoneAtThisSize - 0.5) * substepSize * dDelta;
            }
            /* At the beginning we have a special case
            ** If we are not doing the closing kick, we are also in a special case.
            ** This only ever happens in simulate at the beginning and after a write
            ** where we do not have to undo a kick
            */
            if ((dTime == 0.0) || (! bKickClose)) {
                TSubStepStart = stepStartTime;
                TSubStepKicked = stepStartTime;
            }
            double dtPredISPHUndoOpen = TSubStepStart - TSubStepKicked;
            double dtPredISPHOpen = (dTime - TSubStepStart) / 2.0;
            double dtPredISPHClose = (dTime - TSubStepStart) / 2.0;
            if (csm->val.bComove) {
                kick->dtPredISPHUndoOpen[i] = csmComoveKickFac(csm, TSubStepKicked, dtPredISPHUndoOpen);
                kick->dtPredISPHOpen[i] = csmComoveKickFac(csm, TSubStepStart, dtPredISPHOpen);
                kick->dtPredISPHClose[i] = csmComoveKickFac(csm, TSubStepStart + dtPredISPHOpen, dtPredISPHClose);
            }
            else {
                kick->dtPredISPHUndoOpen[i] = dtPredISPHUndoOpen;
                kick->dtPredISPHOpen[i] = dtPredISPHOpen;
                kick->dtPredISPHClose[i] = dtPredISPHClose;
            }
        }
    }

    if (SPHoptions.doDensity || SPHoptions.doDensityCorrection) {
        uRungLo = uRungLoTemp;
    }
}

void MSR::UpdateGasValues(uint8_t uRungLo, double dTime, double dDelta, double dStep,
                          int bKickClose, int bKickOpen, SPHOptions SPHoptions) {
    struct inUpdateGasValues in;
    in.SPHoptions = SPHoptions;
    double sec, dsec;
    sec = MSR::Time();
    printf("Update Gas Values ...\n");

    CalculateKickParameters(&in.kick, uRungLo, dTime, dDelta, dStep, bKickClose, bKickOpen, SPHoptions);

    pstUpdateGasValues(pst, &in, sizeof(in), NULL, 0);
    dsec = MSR::Time() - sec;
    printf("Gas Values updated, Wallclock: %f secs\n\n", dsec);
}

void MSR::TreeUpdateFlagBounds(int bNeedEwald, uint32_t uRoot, uint32_t utRoot, SPHOptions SPHoptions) {
    struct inTreeUpdateFlagBounds in;
    const double ddHonHLimit = parameters.get_ddHonHLimit();
    PST pst0;
    LCL *plcl;
    PKD pkd;
    double sec, dsec;

    printf("Update local trees...\n\n");

    pst0 = pst;
    while (pst0->nLeaves > 1)
        pst0 = pst0->pstLower;
    plcl = pst0->plcl;
    pkd = plcl->pkd;

    auto nTopTree = pkd->NodeSize() * (2 * nThreads - 1);
    auto nMsgSize = sizeof(ServiceDistribTopTree::input) + nTopTree;

    std::unique_ptr<char[]> buffer {new char[nMsgSize]};
    auto pDistribTop = new (buffer.get()) ServiceDistribTopTree::input;
    auto pkdn = pkd->tree[reinterpret_cast<KDN *>(pDistribTop + 1)];
    pDistribTop->uRoot = uRoot;
    pDistribTop->allocateMemory = 0;

    in.nBucket = parameters.get_nBucket();
    in.nGroup = parameters.get_nGroup();
    in.uRoot = uRoot;
    in.utRoot = utRoot;
    in.ddHonHLimit = ddHonHLimit;
    in.SPHoptions = SPHoptions;
    sec = MSR::Time();
    nTopTree = pstTreeUpdateFlagBounds(pst, &in, sizeof(in), pkdn, nTopTree);
    pDistribTop->nTop = nTopTree / pkd->NodeSize();
    assert(pDistribTop->nTop == (2 * nThreads - 1));
    mdl->RunService(PST_DISTRIBTOPTREE, nMsgSize, pDistribTop);
    dsec = MSR::Time() - sec;
    printf("Tree updated, Wallclock: %f secs\n\n", dsec);

}

uint64_t MSR::CountDistance(double dRadius2Inner, double dRadius2Outer) {
    struct inCountDistance in;
    struct outCountDistance out;
    int nOut;
    in.dRadius2Inner = dRadius2Inner;
    in.dRadius2Outer = dRadius2Outer;
    nOut = pstCountDistance(pst, &in, sizeof(in), &out, sizeof(out));
    assert( nOut == sizeof(out) );
    return out.nCount;
}

double MSR::countSphere(double r, void *vctx) {
    auto *ctx = reinterpret_cast<SPHERECTX *>(vctx);
    ctx->nSelected = ctx->msr->CountDistance(0.0, r * r);
    return 1.0 * ctx->nSelected - 1.0 * ctx->nTarget;
}

void MSR::profileRootFind(double *dBins, int lo, int hi, int nAccuracy, SPHERECTX *ctx) {
    int nIter;
    int iBin = (lo + hi) / 2;
    if (lo == iBin) return;

    ctx->nTarget = ((ctx->nTotal - ctx->nInner) * ctx->dFrac * iBin + ctx->nInner);
    dBins[iBin] = illinois(countSphere, ctx, dBins[lo], dBins[hi], 0.0, 1.0 * nAccuracy, &nIter);
    profileRootFind(dBins, lo, iBin, nAccuracy, ctx);
    profileRootFind(dBins, iBin, hi, nAccuracy, ctx);
}

double MSR::countShell(double rInner, void *vctx) {
    auto *ctx = reinterpret_cast<SHELLCTX *>(vctx);
    double rOuter;
    local_t nSelected;

    if (rInner == ctx->rMiddle) nSelected = 0;
    else {
        rOuter = pow(10, 2.0 * log10(ctx->rMiddle)-log10(rInner));
        nSelected = ctx->msr->CountDistance(rInner * rInner, rOuter * rOuter);
    }
    return 1.0 * nSelected - 1.0 * ctx->nTarget;
}

/*
** Calculate a profile.
** Bins are of equal size (same number of particles) between dMinRadius and dLogRadius.
** From dLogRadius to dMaxRadius, the binning is done logarithmicly.
** Setting dLogRadius to dMinRadius results in purely logarithmic binning, while
** setting dLogRadius to dMaxRadius results in purely equal sized binning.
*/
void MSR::Profile( const PROFILEBIN **ppBins, int *pnBins,
                   double *r, double dMinRadius, double dLogRadius, double dMaxRadius,
                   int nPerBin, int nBins, int nAccuracy ) {
    SPHERECTX ctxSphere;
    SHELLCTX ctxShell;
    PROFILEBIN *pBins;
    double sec, dsec;
    double com[3], vcm[3], L[3], M;
    size_t inSize;
    int i, j;
    int nBinsInner;
    total_t N, n;
    LCL *plcl;
    PST pst0;

    assert(dMinRadius <= dLogRadius);
    assert(dLogRadius <= dMaxRadius);
    assert(dLogRadius == dMinRadius || nPerBin > 0);
    assert(dLogRadius == dMaxRadius || nBins > 0);

    if (dLogRadius == dMaxRadius) nBins = 0;

    pst0 = pst;
    while (pst0->nLeaves > 1)
        pst0 = pst0->pstLower;
    plcl = pst0->plcl;

    CalcDistance(r, dMaxRadius);
    CalcCOM(r, dMaxRadius, com, vcm, L, &M);

    if (dLogRadius > dMinRadius) {
        /*
        ** The inner radius is calculated such that the logarithmic mid - point
        ** falls on dMinRadius.  This is done so that the profile is plotted
        ** all the way to the inner radius.  The correct radius must be between
        ** dMinRadius and the logrithmic difference between dMinRadius and
        ** dMaxRadius below dMinRadius.
        */
        ctxShell.rMiddle = dMinRadius;
        ctxShell.nTarget = nPerBin;
        ctxShell.msr = this;
        dMinRadius = illinois( countShell, &ctxShell,
                               pow(10, 2.0 * log10(dMinRadius)-log10(dMaxRadius)), dMinRadius,
                               0.0, 0.0, NULL );
        N = CountDistance(dMinRadius * dMinRadius, dLogRadius * dLogRadius);
        nBinsInner = (N + nPerBin / 2) / nPerBin;
    }
    else {
        double dOuter;

        nBinsInner = 0;

        /*
        ** Calculate the logarithmic mid - point and verify that there are enough particles
        ** in the first bin.  If not, invoke the root finder.
        */
        ctxShell.rMiddle = dMinRadius;
        ctxShell.nTarget = nPerBin;
        ctxShell.msr = this;
        dMinRadius = pow(10, (2.0*(nBins + 1)*log10(dMinRadius)-log10(dMaxRadius))/(2 * nBins));
        dOuter = pow(10, 2.0 * log10(ctxShell.rMiddle)-log10(dMinRadius));
        N = CountDistance(dMinRadius * dMinRadius, dOuter * dOuter);
        if (N < nPerBin - nAccuracy) {
            dMinRadius = illinois( countShell, &ctxShell,
                                   pow(10, 2.0 * log10(dMinRadius)-log10(dMaxRadius)), dMinRadius,
                                   0.0, 0.0, NULL );
        }
        dLogRadius = dMinRadius;
    }

    inSize = sizeof(inProfile)-sizeof(inProfile::dRadii[0])*(sizeof(inProfile::dRadii)/sizeof(inProfile::dRadii[0])-nBins - nBinsInner - 1);
    std::unique_ptr<char[]> buffer {new char[inSize]};
    auto in = new (buffer.get()) inProfile;

    in->dRadii[0] = dMinRadius;

    /*
    ** Inner, fixed size bins
    */
    if (nBinsInner) {
        sec = Time();
        msrprintf("Root finding for %d bins\n", nBinsInner);
        ctxSphere.nTotal = CountDistance(0.0, dLogRadius * dLogRadius);
        ctxSphere.nInner = CountDistance(0.0, dMinRadius * dMinRadius);
        ctxSphere.msr = this;
        ctxSphere.dFrac = 1.0 / nBinsInner;
        in->dRadii[nBinsInner] = dLogRadius;
        profileRootFind(in->dRadii, 0, nBinsInner, nAccuracy, &ctxSphere);
        dsec = Time() - sec;
        msrprintf("Root finding complete, Wallclock: %f secs\n\n", dsec);
    }

    /*
    ** Now logarithmic binning for the outer region.  We still obey nPerBin
    ** as the minimum number of particles to include in each bin.
    */
    if (nBins) {
        double dLogMin;
        double dLogMax = log10(dMaxRadius);
        double dRadius;

        ctxSphere.nTotal = SelSphere(r, dMaxRadius, 1, 1);
        ctxSphere.msr = this;

        N = CountDistance(0.0, dLogRadius * dLogRadius);
        for (i = 1; i < nBins; i++) {
            int nBinsRem = nBins - i + 1;

            dLogMin = log10(in->dRadii[nBinsInner + i - 1]);
            dRadius = pow(10, (dLogMax - dLogMin)/nBinsRem + dLogMin);
            n = CountDistance(0.0, dRadius * dRadius);
            if (n - N < nPerBin - nAccuracy) {
                ctxSphere.nTarget = N + nPerBin;
                dRadius = illinois( countSphere, &ctxSphere, 0.0, dMaxRadius,
                                    0.0, 1.0 * nAccuracy, NULL );
                n = ctxSphere.nSelected;
            }
            in->dRadii[nBinsInner + i] = dRadius;
            N = n;
        }
    }

    nBins = nBins + nBinsInner;

    in->dRadii[nBins] = dMaxRadius;

    sec = Time();
    msrprintf("Profiling\n");
    for (i = 0; i < 3; i++) {
        in->dCenter[i] = r[i];
        in->com[i] = com[i];
        in->vcm[i] = vcm[i];
        in->L[i] = L[i];
    }
    in->nBins = nBins + 1;
    in->uRungLo = 0;
    in->uRungHi = MaxRung()-1;
    pstProfile(pst, in, inSize, NULL, 0);

    /*
    ** Finalize bin values
    */
    pBins = plcl->pkd->profileBins;
    for (i = 0; i < nBins + 1; i++) {
        if (pBins[i].dMassInBin > 0.0) {
            pBins[i].vel_radial /= pBins[i].dMassInBin;
            pBins[i].vel_radial_sigma /= pBins[i].dMassInBin;
            pBins[i].vel_tang_sigma = sqrt(pBins[i].vel_tang_sigma / pBins[i].dMassInBin);
            if (pBins[i].vel_radial_sigma > pBins[i].vel_radial * pBins[i].vel_radial)
                pBins[i].vel_radial_sigma = sqrt(pBins[i].vel_radial_sigma - pBins[i].vel_radial * pBins[i].vel_radial);
            else
                pBins[i].vel_radial_sigma = 0.0;
            for (j = 0; j < 3; j++) {
                pBins[i].L[j] /= pBins[i].dMassInBin;
            }
        }
    }

    dsec = Time() - sec;
    msrprintf("Profiling complete, Wallclock: %f secs\n\n", dsec);

    if (ppBins) *ppBins = plcl->pkd->profileBins;
    if (pnBins) *pnBins = nBins + 1;
}

#ifdef MDL_FFTW
void MSR::GridCreateFFT(int nGrid) {
    struct inGridCreateFFT in;
    in.nGrid = nGrid;
    pstGridCreateFFT(pst, &in, sizeof(in), NULL, 0);
}

void MSR::GridDeleteFFT() {
    pstGridDeleteFFT(pst, NULL, 0, NULL, 0);
}

/* Important: call msrGridCreateFFT() before, and msrGridDeleteFFT() after */
std::tuple<std::vector<uint64_t>, std::vector<float>, std::vector<float>, std::vector<float>> // nPk, fK, fPk, fPkAll
MSR::MeasurePk(int iAssignment, int bInterlace, int nGrid, double a, int nBins) {
    std::vector<uint64_t> nPk;
    std::vector<float> fK, fPk, fPkAll;
    double dsec;

    GridCreateFFT(nGrid);

    if (nGrid / 2 < nBins) nBins = nGrid / 2;
    assert(nBins <= PST_MAX_K_BINS);

    TimerStart(TIMER_NONE);
    printf("Measuring P(k) with grid size %d (%d bins)...\n", nGrid, nBins);

    AssignMass(iAssignment, 0, 0.0);
    DensityContrast(0);
    if (bInterlace) {
        AssignMass(iAssignment, 1, 0.5);
        DensityContrast(1);
        Interlace(0, 1); // We no longer need grid 1
    }
    WindowCorrection(iAssignment, 0);

    std::tie(nPk, fK, fPk) = GridBinK(nBins, 0);
    if (csm->val.classData.bClass && parameters.get_nGridLin()>0 && parameters.get_achPkSpecies().length() > 0) {
        AddLinearSignal(0, parameters.get_iSeed(), parameters.get_dBoxSize(), a,
                        parameters.get_bFixedAmpIC(), parameters.get_dFixedAmpPhasePI() * M_PI);
        std::tie(nPk, fK, fPkAll) = GridBinK(nBins, 0);
    }
    else {
        fPkAll.resize(nBins);
    }

    GridDeleteFFT();

    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    printf("P(k) Calculated, Wallclock: %f secs\n\n", dsec);
    return { nPk, fK, fPk, fPkAll };
}

std::tuple<std::vector<uint64_t>, std::vector<float>, std::vector<float>> // nPk, fK, fPk
MSR::MeasureLinPk(int nGrid, double dA, double dBoxSize) {
    struct inMeasureLinPk in;
    int i;
    double dsec;

    TimerStart(TIMER_NONE);

    in.nGrid = nGrid;
    in.nBins = nGrid / 2;
    in.dBoxSize = dBoxSize;
    in.dA = dA;
    in.iSeed = parameters.get_iSeed();
    in.bFixed = parameters.get_bFixedAmpIC();
    in.fPhase = parameters.get_dFixedAmpPhasePI() * M_PI;

    std::unique_ptr<struct outMeasureLinPk> out {new struct outMeasureLinPk};
    printf("Measuring P_lin(k) with grid size %d (%d bins)...\n", in.nGrid, in.nBins);
    pstMeasureLinPk(pst, &in, sizeof(in), out.get(), sizeof(*out));
    std::vector<uint64_t> nPk(nBins);
    std::vector<float> fK(nBins), fPk(nBins);
    for (i = 0; i < in.nBins; i++) {
        if (out->nPower[i] == 0) fK[i] = fPk[i] = 0;
        else {
            if (nPk.size()) nPk[i] = out->nPower[i];
            fK[i] = out->fK[i]/out->nPower[i];
            fPk[i] = out->fPower[i]/out->nPower[i];
        }
    }
    /* At this point, dPk[] needs to be corrected by the box size */

    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    printf("P_lin(k) Calculated, Wallclock: %f secs\n\n", dsec);
    return { nPk, fK, fPk };
}

void MSR::SetLinGrid(double dTime, double dDelta, int nGrid, int bKickClose, int bKickOpen) {
    printf("Setting force grids of linear species with nGridLin = %d \n", nGrid);
    double dsec;
    TimerStart(TIMER_NONE);

    struct inSetLinGrid in;
    in.nGrid = nGrid;

    int do_DeltaRho_lin_avg = 1;
    in.a0 = in.a1 = in.a = csmTime2Exp(csm, dTime);
    if (do_DeltaRho_lin_avg) {
        if (bKickClose) in.a0 = csmTime2Exp(csm, dTime - 0.5 * dDelta);
        if (bKickOpen)  in.a1 = csmTime2Exp(csm, dTime + 0.5 * dDelta);
    }

    in.dBSize = parameters.get_dBoxSize();
    /* Parameters for the grid realization */
    in.iSeed = parameters.get_iSeed();
    in.bFixed = parameters.get_bFixedAmpIC();
    in.fPhase = parameters.get_dFixedAmpPhasePI()*M_PI;
    pstSetLinGrid(pst, &in, sizeof(in), NULL, 0);

    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    printf("Force from linear species calculated, Wallclock: %f, secs\n\n", dsec);
}

/* First call SetLinGrid() to setup the grid */
void MSR::LinearKick(double dTime, double dDelta, int bKickClose, int bKickOpen) {
    struct inLinearKick in;
    double dt = 0.5 * dDelta;
    double dsec;

    printf("Applying Linear Kick...\n");
    TimerStart(TIMER_NONE);
    in.dtOpen = in.dtClose = 0.0;
    if (csm->val.bComove) {
        if (bKickClose) in.dtClose = csmComoveKickFac(csm, dTime - dt, dt);
        if (bKickOpen) in.dtOpen = csmComoveKickFac(csm, dTime, dt);
    }
    else {
        if (bKickClose) in.dtClose = dt;
        if (bKickOpen) in.dtOpen = dt;
    }
    pstLinearKick(pst, &in, sizeof(in), NULL, 0);
    TimerStop(TIMER_NONE);
    dsec = TimerGet(TIMER_NONE);
    printf("Linear Kick Applied, Wallclock: %f secs\n\n", dsec);
}
#endif

mdl::ServiceBuffer MSR::GetParticles(std::vector<std::int64_t> &particle_ids) {
    mdl::ServiceBufferOut outBuffer {
        mdl::ServiceBuffer::Field<struct outGetParticles>(particle_ids.size())
    };
    mdl->RunService(PST_GET_PARTICLES, sizeof(uint64_t)*particle_ids.size(), particle_ids.data(), outBuffer);
    return std::move(outBuffer);
}

void MSR::OutputOrbits(int iStep, double dTime) {
    int i;

    if (parameters.has_lstOrbits()) {
        double dExp, dvFac;

        if (csm->val.bComove) {
            dExp = csmTime2Exp(csm, dTime);
            dvFac = 1.0/(dExp * dExp);
        }
        else {
            dExp = dTime;
            dvFac = 1.0;
        }

        auto particle_ids = parameters.get_lstOrbits();
        auto out = GetParticles(particle_ids);
        auto particles = static_cast<struct outGetParticles *>(out.data(0));

        auto filename = BuildName(iStep, ".orb");
        std::ofstream fs(filename);
        if (fs.fail()) {
            std::cerr << "Could not create orbit file: " << filename << std::endl;
            perror(filename.c_str());
            Exit(errno);
        }
        fmt::print(fs, "{n} {a}\n", "n"_a = particle_ids.size(), "a"_a = dExp);
        for (i = 0; i < particle_ids.size(); ++i) {
            fmt::print(fs, "{id} {mass:.8e} {x:.16e} {y:.16e} {z:.16e} {vx:.8e} {vy:.8e} {vz:.8e} {phi:.8e}\n",
                       "id"_a = particles[i].id, "mass"_a = particles[i].mass, "phi"_a = particles[i].phi,
                       "x"_a = particles[i].r[0], "y"_a = particles[i].r[1], "z"_a = particles[i].r[2],
                       "vx"_a = particles[i].v[0]*dvFac, "vy"_a = particles[i].v[1]*dvFac, "vz"_a = particles[i].v[2]*dvFac);
        }
        fs.close();
    }
}

#ifdef HAVE_ROCKSTAR
void MSR::RsHaloLoadIds(const std::string &filename_template, bool bAppend) {
    std::vector<uint64_t> counts;

    TimerStart(TIMER_NONE);
    printf("Scanning Rockstar halo binary files...\n");
    ServiceRsHaloCount::input hdr;
    strncpy(hdr.filename, filename_template.c_str(), sizeof(hdr.filename));
    hdr.nSimultaneous = hdr.nTotalActive = parallel_read_count();
    hdr.iReaderWriter = 0;
    hdr.nElementSize = 1;
    auto out = new ServiceFileSizes::output[ServiceFileSizes::max_files];
    auto n = mdl->RunService(PST_RS_HALO_COUNT, sizeof(hdr), &hdr, out);
    n /= sizeof(ServiceFileSizes::output);
    counts.resize(n);
    for (auto i = 0; i < n; ++i) {
        assert(out[i].iFileIndex < n);
        counts[out[i].iFileIndex] = out[i].nFileBytes;
    }
    delete [] out;

    if (counts.size()==0) {
        perror(filename_template.c_str());
        abort();
    }
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);

    printf("... identified %" PRIu64 " halos in %d files, Wallclock: %f secs.\n",
           std::accumulate(counts.begin(), counts.end(), uint64_t(0)),
           int(counts.size()), dsec);
    RsLoadIds(PST_RS_HALO_LOAD_IDS, counts, filename_template, bAppend);
    TimerStop(TIMER_IO);
}
#endif

void MSR::RsLoadIds(int sid, std::vector<uint64_t> &counts, const std::string &filename_template, bool bAppend) {
    using mdl::ServiceBuffer;
    ServiceBuffer msg {
        ServiceBuffer::Field<ServiceInput::input>(),
        ServiceBuffer::Field<ServiceRsLoadIds::io_elements>(counts.size()),
        ServiceBuffer::Field<ServiceRsLoadIds::input>()
    };
    auto hdr = static_cast<ServiceInput::input *>(msg.data(0));
    auto elements = static_cast<ServiceRsLoadIds::io_elements *>(msg.data(1));
    auto in = static_cast<ServiceRsLoadIds::input *>(msg.data(2));
    in->bAppend = bAppend;
    hdr->nFiles = counts.size();
    hdr->nElements = std::accumulate(counts.begin(), counts.end(), uint64_t(0));
    std::copy(counts.begin(), counts.end(), elements);
    hdr->iBeg = 0;
    hdr->iEnd = hdr->nElements;
    strncpy(hdr->io.filename, filename_template.c_str(), sizeof(hdr->io.filename));
    hdr->io.nSimultaneous = parallel_read_count();
    hdr->io.nSegment = hdr->io.iThread = 0; // setup later
    hdr->io.iReaderWriter = 0;
    printf("Loading %" PRIu64 " particle IDs from %d files\n", hdr->nElements, hdr->nFiles);
    TimerStart(TIMER_NONE);
    mdl->RunService(sid, msg);
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);
    printf("... finished reading particles IDs, Wallclock: %f secs.\n", dsec);
}

void MSR::RsLoadIds(const std::string &filename_template, bool bAppend) {
    std::vector<uint64_t> counts;
    TimerStart(TIMER_NONE);
    printf("Scanning Particle ID binary files...\n");
    stat_files(counts, filename_template, sizeof(uint64_t));
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);
    printf("... identified %" PRIu64 " IDs in %d files, Wallclock: %f secs.\n",
           std::accumulate(counts.begin(), counts.end(), uint64_t(0)),
           int(counts.size()), dsec);
    RsLoadIds(PST_RS_LOAD_IDS, counts, filename_template, bAppend);
}

void MSR::RsSaveIds(const std::string &filename_template) {
    ServiceRsSaveIds::input hdr;
    strncpy(hdr.io.filename, filename_template.c_str(), sizeof(hdr.io.filename));
    hdr.io.nSimultaneous = parallel_write_count();
    hdr.io.nSegment = hdr.io.iThread = 0; // setup later
    hdr.io.iReaderWriter = 0;
    printf("Saving particle IDS\n");
    TimerStart(TIMER_NONE);
    mdl->RunService(PST_RS_SAVE_IDS, sizeof(hdr), &hdr);
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);
    printf("... finished writing particles IDs, Wallclock: %f secs.\n", dsec);
}

void MSR::RsReorderIds() {
    using mdl::ServiceBuffer;
    ServiceBuffer msg {
        ServiceBuffer::Field<ServiceRsExtract::input>(mdl->Threads()+1)
    };
    auto pOrds = static_cast<ServiceRsExtract::input *>(msg.data(0));
    pOrds[mdl->Threads()] = N;
    printf("Reordering particle IDS\n");
    TimerStart(TIMER_NONE);
    mdl->RunService(PST_GET_ORD_SPLITS, pOrds);
    mdl->RunService(PST_RS_REORDER_IDS, msg);
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);
    printf("... finished reordering particles IDs, Wallclock: %f secs.\n", dsec);
}

void MSR::RsExtract(const char *filename_template) {
    printf("Extracting matching particles\n");
    using mdl::ServiceBuffer;
    ServiceBuffer msg {
        ServiceBuffer::Field<ServiceRsExtract::header>(),
        ServiceBuffer::Field<ServiceRsExtract::input>(mdl->Threads()+1)
    };
    auto hdr = static_cast<ServiceRsExtract::header *>(msg.data(0));
    strncpy(hdr->io.filename, filename_template, sizeof(hdr->io.filename));
    hdr->io.nSimultaneous = parallel_write_count();
    hdr->io.nSegment = hdr->io.iThread = 0; // setup later
    hdr->io.iReaderWriter = 0;

    auto pOrds = static_cast<ServiceRsExtract::input *>(msg.data(1));
    pOrds[mdl->Threads()] = N;
    TimerStart(TIMER_NONE);
    mdl->RunService(PST_GET_ORD_SPLITS, pOrds);
    mdl->RunService(PST_RS_EXTRACT, msg);
    TimerStop(TIMER_NONE);
    auto dsec = TimerGet(TIMER_NONE);
    printf("... finished extracting particles, Wallclock: %f secs.\n", dsec);
}

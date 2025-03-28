/* config.h.  Generated by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */
#ifndef CONFIG_H
#define CONFIG_H
#define NOMINMAX
#include <windows.h>
#define _USE_MATH_DEFINES
#define NO_XDR
#define NO_COOLING
/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
/* #undef CRAY_STACKSEG_END */

/* This defines the base floating point type for particles. */
#define FLOAT double

/* This defines the maximum value that a FLOAT can represent. */
#define FLOAT_MAXVAL HUGE

/* Use the HDF5 1.6 API instead of later versions */
#define H5_USE_16_API 1

/* Define to 1 if you have `alloca', as a function or macro. */
/*#undef HAVE_ALLOCA */

/* Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).
   */
/* #undef HAVE_ALLOCA_H */

/* Define to 1 if you have the <ansidecl.h> header file. */
/* #undef HAVE_ANSIDECL_H */

/* Define to 1 if you have the `clock_gettime' function. */
/* #undef HAVE_CLOCK_GETTIME */

/* Define to 1 if you have the <c_asm.h> header file. */
/* #undef HAVE_C_ASM_H */

/* Define to 1 if you don't have `vprintf' but do have `_doprnt.' */
/* #undef HAVE_DOPRNT */

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the <fenv.h> header file. */
#define HAVE_FENV_H 1

/* Define to 1 if you have the `floor' function. */
#define HAVE_FLOOR 1

/* Define to 1 if fseeko (and presumably ftello) exists and is declared. */
/* #undef HAVE_FSEEKO */

/* Define to 1 if you have the `gethostname' function. */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if you have the `gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Define to 1 if you have the `getpagesize' function. */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the `glob' function. */
/* #undef HAVE_GLOB */

/* Define to 1 if you have the `globfree' function. */
/* #undef HAVE_GLOBFREE */

/* Define to 1 if hrtime_t is defined in <sys/time.h> */
/* #undef HAVE_HRTIME_T */

/* Define to 1 if you have the <intrinsics.h> header file. */
/* #undef HAVE_INTRINSICS_H */

/* Define to 1 if you have the <inttypes.h> header file. */
/* #undef HAVE_INTTYPES_H */

/* Define to 1 if you have the `bz2' library (-lbz2). */
/* #undef HAVE_LIBBZ2 */

/* Define to 1 if you have the `gd' library (-lgd). */
/* #undef HAVE_LIBGD */

/* Define to 1 if you have the `gpfs' library (-lgpfs). */
/* #undef HAVE_LIBGPFS */

/* Define to 1 if you have the `hdf5' library (-lhdf5). */
/* #undef HAVE_LIBHDF5 */

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if you have the `png' library (-lpng). */
/* #undef HAVE_LIBPNG */

/* Define to 1 if you have the `z' library (-lz). */
/* #undef HAVE_LIBZ */

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if you have the `mach_absolute_time' function. */
/* #undef HAVE_MACH_ABSOLUTE_TIME */

/* Define to 1 if you have the <mach/mach_time.h> header file. */
/* #undef HAVE_MACH_MACH_TIME_H */

/* Define to 1 if your system has a GNU libc compatible `malloc' function, and
   to 0 otherwise. */
#define HAVE_MALLOC 1

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `mkdir' function. */
#define HAVE_MKDIR 1

/* Define if posix_memalign should be used instead of malloc for SIMD. */
/*#undef HAVE_POSIX_MEMALIGN */

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* Define to 1 if you have the `read_real_time' function. */
/* #undef HAVE_READ_REAL_TIME */

/* Define to 1 if your system has a GNU libc compatible `realloc' function,
   and to 0 otherwise. */
#define HAVE_REALLOC 1

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if `stat' has the bug that it succeeds when given the
   zero-length file name argument. */
/* #undef HAVE_STAT_EMPTY_STRING_BUG */

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strchr' function. */
#define HAVE_STRCHR 1

/* Define to 1 if you have the `strdup' function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strrchr' function. */
#define HAVE_STRRCHR 1

/* Define to 1 if you have the `strstr' function. */
#define HAVE_STRSTR 1

/* Define to 1 if you have the <sys/param.h> header file. */
/* #undef HAVE_SYS_PARAM_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
/* #undef HAVE_SYS_TIME_H */

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the `time_base_to_time' function. */
/* #undef HAVE_TIME_BASE_TO_TIME */

/* Define to 1 if you have the <unistd.h> header file. */
/*#undef HAVE_UNISTD_H */

/* Define to 1 if you have the `vprintf' function. */
#define HAVE_VPRINTF 1

/* Define to 1 if you have the `wordexp' function. */
/* #undef HAVE_WORDEXP */

/* Define to 1 if you have the `wordfree' function. */
/* #undef HAVE_WORDFREE */

/* Define if you have the UNICOS _rtc() intrinsic. */
/* #undef HAVE__RTC */

/* Define if hermite should be used. */
/* #undef HERMITE */

/* Define if SHRINK H4 kernel should be used. */
/* #undef HSHRINK */

/* Instrument the code. */
#define INSTRUMENT 1

/* Define if local expansions should be used. */
#define LOCAL_EXPANSION 1

/* Define to 1 if `lstat' dereferences a symlink specified with a trailing
   slash. */
#define LSTAT_FOLLOWS_SLASHED_SYMLINK 1

/* Include support for FFTW. */
#define MDL_FFTW 1

/* The maximum number of IO processors */
/* #undef MDL_MAX_IO_PROCS */

/* Name of package */
#define PACKAGE "pkdgrav2"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "bugs@pkdgrav2.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "pkdgrav2"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "pkdgrav2 2.2.11"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "pkdgrav2"

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.2.11"

/* Define if planets should be used. */
/* #undef PLANETS */

/* Define if RELAXATION should be used. */
/* #undef RELAXATION */

/* Define as the return type of signal handlers (`int' or `void'). */
#define RETSIGTYPE void

/* Define if softening should NOT be weighted by mass. */
/* #undef SOFTENING_NOT_MASS_WEIGHTED */

/* Define if SOFTLINEAR should be used. */
/* #undef SOFTLINEAR */

/* Define if SOFTSQUARE should be used. */
/* #undef SOFTSQUARE */

/* If using the C implementation of alloca, define if you know the
   direction of stack growth for your system; otherwise it will be
   automatically deduced at run-time.
	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown */
/* #undef STACK_DIRECTION */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define if symba should be used. */
/* #undef SYMBA */

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
/* #undef TIME_WITH_SYS_TIME */

/* Define if BSC tracing should be used. */
/* #undef USE_BSC */

/* Print a backtrace on crash. */
/* #undef USE_BT */

/* Use CUDA to accelerate calculations */
#define USE_CUDA 1
/* Use CUDA to accelerate calculations */
//#define USE_CL 1

/* Define GRAFICIC IC should be available. */
/* #undef USE_GRAFIC */

/* Define if HDF5 I/O should be available. */
/* #undef USE_HDF5 */

/* Define if lustre support should be included. */
/* #undef USE_LUSTRE */

/* Define if MDL style I/O should be used. */
/* #undef USE_MDL_IO */

/* Define if PNG output is supported. */
/* #undef USE_PNG */

/* Define if phase-space coordinates/functions should be available. */
/* #undef USE_PSD */

/* Defined if python support should be compiled into pkdgrav2 */
/* #undef USE_PYTHON */

/* Define if SIMD optimizations should be used. */
/* Visual Studio definition */
#if defined(__AVX__)
#define USE_SIMD 1
#ifndef __SSE__
#define __SSE__
#endif
#ifndef __SSE2__
#define __SSE2__
#endif
#ifdef __AVX2__
#define __FMA__
#endif
#elif defined(_M_X64)
#define __SSE__
#define __SSE2__
#define __AVX__
#ifdef __AVX2__
#define __FMA__
#endif
#define USE_SIMD 1
#elif _M_IX86_FP >= 1
#define USE_SIMD 1
#define __SSE__
#if _M_IX86_FP >= 2
#define __SSE2__
#endif
#endif

#ifdef USE_SIMD

/* Define if SIMD optimizations should be used for Ewald. */
#define USE_SIMD_EWALD 1

/* Define if SIMD optimizations should be used for FMM interactions. */
#define USE_SIMD_FMM 1

/* Define if SIMD optimizations should be used for MOMRs. */
/* #undef USE_SIMD_MOMR */

/* Define if SIMD optimizations should be used for the opening criteria. */
#define USE_SIMD_OPEN 1

/* Define if SIMD optimizations should be used for PP interactions. */
#define USE_SIMD_PP 1

/* Define if SIMD optimizations should be used for PC interactions. */
#define USE_SIMD_PC 1
#endif

/* Use single precision FFT. */
#define USE_SINGLE 1

/* Version number of package */
#define VERSION "2.2.11"

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* This is needed for posix_memalign to be defined in stdlib.h */
#define _GNU_SOURCE 1

/* Define to 1 to make fseeko visible on some hosts (e.g. glibc 2.2). */
/* #undef _LARGEFILE_SOURCE */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#define inline __inline
#endif
#define restrict

/* Define to rpl_malloc if the replacement function should be used. */
/* #undef malloc */

/* Define to `long' if <sys/types.h> does not define. */
/* #undef off_t */

/* Define to rpl_realloc if the replacement function should be used. */
/* #undef realloc */

/* Define to `unsigned' if <sys/types.h> does not define. */
/* #undef size_t */

/* Define to empty if the keyword `volatile' does not work. Warning: valid
   code using `volatile' can become incorrect without. Disable with care. */
/* #undef volatile */
//static inline float fmin(float a, float b) {
//    return a < b ? a : b;
//}
//static inline float fmax(float a, float b) {
//    return a > b ? a : b;
//}
#endif

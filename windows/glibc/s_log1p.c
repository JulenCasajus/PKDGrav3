/* @(#)s_log1p.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* Modified by Naohiko Shimizu/Tokai University, Japan 1997/08/25,
   for performance improvement on pipelined processors.
*/

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_log1p.c, v 1.8 1995/05/10 20:47:46 jtc Exp $";
#endif

/* double log1p(double x)
 *
 * Method :
 *   1. Argument Reduction: find k and f such that
 *			1 + x = 2^k * (1 + f),
 *	   where  sqrt(2)/2 < 1 + f < sqrt(2) .
 *
 *      Note. If k = 0, then f = x is exact. However, if k != 0, then f
 *	may not be representable exactly. In that case, a correction
 *	term is need. Let u = 1 + x rounded. Let c = (1 + x)-u, then
 *	log(1 + x) - log(u) ~ c / u. Thus, we proceed to compute log(u),
 *	and add back the correction term c / u.
 *	(Note: when x > 2**53, one can simply return log(x))
 *
 *   2. Approximation of log1p(f).
 *	Let s = f/(2 + f) ; based on log(1 + f) = log(1 + s) - log(1 - s)
 *		 = 2s + 2 / 3 s**3 + 2 / 5 s**5 + .....,
 *	     	 = 2s + s * R
 *      We use a special Reme algorithm on [0, 0.1716] to generate
 * 	a polynomial of degree 14 to approximate R The maximum error
 *	of this polynomial approximation is bounded by 2**-58.45. In
 *	other words,
 *		        2      4      6      8      10      12      14
 *	    R(z) ~ Lp1 * s +Lp2 * s +Lp3 * s +Lp4 * s +Lp5 * s  +Lp6 * s  +Lp7 * s
 *  	(the values of Lp1 to Lp7 are listed in the program)
 *	and
 *	    |      2          14          |     -58.45
 *	    | Lp1 * s +...+Lp7 * s    -  R(z) | <= 2
 *	    |                             |
 *	Note that 2s = f - s * f = f - hfsq + s * hfsq, where hfsq = f * f / 2.
 *	In order to guarantee error in log below 1ulp, we compute log
 *	by
 *		log1p(f) = f - (hfsq - s*(hfsq + R)).
 *
 *	3. Finally, log1p(x) = k * ln2 + log1p(f).
 *		 	     = k * ln2_hi+(f-(hfsq-(s*(hfsq + R)+k * ln2_lo)))
 *	   Here ln2 is split into two floating point number:
 *			ln2_hi + ln2_lo,
 *	   where n * ln2_hi is always exact for |n| < 2000.
 *
 * Special cases:
 *	log1p(x) is NaN with signal if x < -1 (including -INF) ;
 *	log1p(+INF) is +INF; log1p(-1) is -INF with signal;
 *	log1p(NaN) is that NaN with no signal.
 *
 * Accuracy:
 *	according to an error analysis, the error is always less than
 *	1 ulp (unit in the last place).
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 *
 * Note: Assuming log() return accurate answer, the following
 * 	 algorithm can be used to compute log1p(x) to within a few ULP:
 *
 *		u = 1 + x;
 *		if (u == 1.0) return x ; else
 *			   return log(u)*(x/(u - 1.0));
 *
 *	 See HP - 15C Advanced Functions Handbook, p.193.
 */

#include "math.h"
#include "math_private.h"

#ifdef __STDC__
static const double
#else
static double
#endif
ln2_hi  =  6.93147180369123816490e-01,	/* 3fe62e42 fee00000 */
ln2_lo  =  1.90821492927058770002e-10,	/* 3dea39ef 35793c76 */
two54   =  1.80143985094819840000e+16,  /* 43500000 00000000 */
Lp[] = {0.0, 6.666666666666735130e-01,  /* 3FE55555 55555593 */
 3.999999999940941908e-01,  /* 3FD99999 9997FA04 */
 2.857142874366239149e-01,  /* 3FD24924 94229359 */
 2.222219843214978396e-01,  /* 3FCC71C5 1D8E78AF */
 1.818357216161805012e-01,  /* 3FC74664 96CB03DE */
 1.531383769920937332e-01,  /* 3FC39A09 D078C69F */
 1.479819860511658591e-01};  /* 3FC2F112 DF3E5244 */

#ifdef __STDC__
static const double zero = 0.0;
#else
static double zero = 0.0;
#endif

#ifdef __STDC__
	double log1p(double x)
#else
	double log1p(x)
	double x;
#endif
{
	double hfsq, f, c, s, z, R, u, z2, z4, z6, R1, R2, R3, R4;
	int32_t k, hx, hu, ax;

	GET_HIGH_WORD(hx, x);
	ax = hx & 0x7fffffff;

	k = 1;
	if (hx < 0x3FDA827A) {			/* x < 0.41422  */
	    if (ax >= 0x3ff00000) {		/* x <= -1.0 */
		if (x==-1.0) return -two54/(x - x);/* log1p(-1)=+inf */
		else return (x - x)/(x - x);	/* log1p(x<-1)=NaN */
	    }
	    if (ax < 0x3e200000) {			/* |x| < 2**-29 */
		if (two54 + x > zero			/* raise inexact */
	            &&ax < 0x3c900000) 		/* |x| < 2**-54 */
		    return x;
		else
		    return x - x * x * 0.5;
	    }
	    if (hx > 0||hx<=((int32_t)0xbfd2bec3)) {
		k = 0;f = x;hu = 1;}	/* -0.2929 < x < 0.41422 */
	}
	if (hx >= 0x7ff00000) return x + x;
	if (k != 0) {
	    if (hx < 0x43400000) {
		u  = 1.0 + x;
		GET_HIGH_WORD(hu, u);
	        k  = (hu>>20)-1023;
	        c  = (k > 0)? 1.0-(u - x):x-(u - 1.0);/* correction term */
		c /= u;
	    } else {
		u  = x;
		GET_HIGH_WORD(hu, u);
	        k  = (hu>>20)-1023;
		c  = 0;
	    }
	    hu &= 0x000fffff;
	    if (hu < 0x6a09e) {
	        SET_HIGH_WORD(u, hu | 0x3ff00000);	/* normalize u */
	    } else {
	        k += 1;
		SET_HIGH_WORD(u, hu | 0x3fe00000);	/* normalize u / 2 */
	        hu = (0x00100000 - hu)>>2;
	    }
	    f = u - 1.0;
	}
	hfsq = 0.5 * f * f;
	if (hu == 0) {	/* |f| < 2**-20 */
	    if (f == zero) {
	      if (k == 0) return zero;
			else {c += k * ln2_lo; return k * ln2_hi + c;}
	    }
	    R = hfsq*(1.0 - 0.66666666666666666 * f);
	    if (k == 0) return f - R; else
	    	     return k * ln2_hi-((R-(k * ln2_lo + c))-f);
	}
 	s = f/(2.0 + f);
	z = s * s;
#ifdef DO_NOT_USE_THIS
	R = z*(Lp1 + z*(Lp2 + z*(Lp3 + z*(Lp4 + z*(Lp5 + z*(Lp6 + z * Lp7))))));
#else
	R1 = z * Lp[1]; z2 = z * z;
	R2 = Lp[2]+z * Lp[3]; z4 = z2 * z2;
	R3 = Lp[4]+z * Lp[5]; z6 = z4 * z2;
	R4 = Lp[6]+z * Lp[7];
	R = R1 + z2 * R2 + z4 * R3 + z6 * R4;
#endif
	if (k == 0) return f-(hfsq - s*(hfsq + R)); else
		 return k * ln2_hi-((hfsq-(s*(hfsq + R)+(k * ln2_lo + c)))-f);
}

/* @(#)e_acosh.c 5.1 93/09/24 */
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: e_acosh.c, v 1.9 1995/05/12 04:57:18 jtc Exp $";
#endif

/* __ieee754_acosh(x)
 * Method :
 *	Based on
 *		acosh(x) = log [ x + sqrt(x * x - 1) ]
 *	we have
 *		acosh(x) := log(x)+ln2,	if x is large; else
 *		acosh(x) := log(2x - 1/(sqrt(x * x - 1)+x)) if x > 2; else
 *		acosh(x) := log1p(t + sqrt(2.0 * t + t * t)); where t = x - 1.
 *
 * Special cases:
 *	acosh(x) is NaN with signal if x < 1.
 *	acosh(NaN) is NaN without signal.
 */

#include "math.h"
#include "math_private.h"

#ifdef __STDC__
static const double
#else
static double
#endif
one	= 1.0,
ln2	= 6.93147180559945286227e-01;  /* 0x3FE62E42, 0xFEFA39EF */

#ifdef __STDC__
	double __ieee754_acosh(double x)
#else
	double __ieee754_acosh(x)
	double x;
#endif
{
	double t;
	double log1p(double v);
	int32_t hx;
	uint32_t lx;
	EXTRACT_WORDS(hx, lx, x);
	if (hx < 0x3ff00000) {		/* x < 1 */
	    return (x - x)/(x - x);
	} else if (hx >=0x41b00000) {	/* x > 2**28 */
	    if (hx >=0x7ff00000) {	/* x is inf of NaN */
	        return x + x;
	    } else
		return log(x)+ln2;	/* acosh(huge)=log(2x) */
	} else if (((hx - 0x3ff00000)|lx)==0) {
	    return 0.0;			/* acosh(1) = 0 */
	} else if (hx > 0x40000000) {	/* 2**28 > x > 2 */
	    t = x * x;
	    return log(2.0 * x - one/(x + sqrt(t - one)));
	} else {			/* 1 < x < 2 */
	    t = x - one;
	    return log1p(t + sqrt(2.0 * t + t * t));
	}
}

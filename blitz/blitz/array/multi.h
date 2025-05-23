// -*- C++ -*-
/***************************************************************************
 * blitz/array/multi.h  Support for multicomponent arrays
 *
 * $Id$
 *
 * Copyright (C) 1997-2011 Todd Veldhuizen <tveldhui@acm.org>
 *
 * This file is a part of Blitz.
 *
 * Blitz is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * Blitz is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Blitz.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Suggestions:          blitz-devel@lists.sourceforge.net
 * Bugs:                 blitz-support@lists.sourceforge.net
 *
 * For more information, please see the Blitz++ Home Page:
 *    https://sourceforge.net/projects/blitz/
 *
 ****************************************************************************/
#ifndef BZ_ARRAYMULTI_H
#define BZ_ARRAYMULTI_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/multi.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * The multicomponent_traits class provides a mapping from multicomponent
 * tuples to the element type they contain.  For example:
 *
 * multicomponent_traits<complex<float> >::T_numtype is float,
 * multicomponent_traits<TinyVector<int, 3> >::T_numtype is int.
 *
 * This is used to support Array<T, N>::operator[], which extracts components
 * from a multicomponent array.
 */

// By default, produce a harmless component type, and zero components.
template<typename T_component>
struct multicomponent_traits {
    typedef T_component T_element;
    static const int numComponents = 0;
};

// TinyVector
template<typename T_numtype, int N_rank>
struct multicomponent_traits<TinyVector<T_numtype, N_rank> > {
    typedef T_numtype T_element;
    static const int numComponents = N_rank;
};

// TinyMatrix
template<typename T_numtype, int N_rows, int N_cols>
struct multicomponent_traits<TinyMatrix<T_numtype, N_rows, N_cols> > {
    typedef T_numtype T_element;
    static const int numComponents = N_rows * N_cols;
};

#ifdef BZ_HAVE_COMPLEX
// complex<T>
template<typename T>
struct multicomponent_traits<complex<T> > {
    typedef T T_element;
    static const int numComponents = 2;
};
#endif

// This macro is provided so that users can register their own
// multicomponent types.

#define BZ_DECLARE_MULTICOMPONENT_TYPE(T_tuple, T, N)          \
  BZ_NAMESPACE(blitz)                                        \
  template<>                                                 \
  struct multicomponent_traits<T_tuple > {                   \
    typedef T T_element;                                     \
    static const int numComponents = N;                      \
  };                                                         \
  BZ_NAMESPACE_END

BZ_NAMESPACE_END

#endif // BZ_ARRAYMULTI_H

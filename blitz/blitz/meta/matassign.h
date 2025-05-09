// -*- C++ -*-
/***************************************************************************
 * blitz/meta / matassign.h   TinyMatrix assignment metaprogram
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
 **************************************************************************/


#ifndef BZ_META_MATASSIGN_H
#define BZ_META_MATASSIGN_H

BZ_NAMESPACE(blitz)

template<int N_rows, int N_columns, int I, int J>
class _bz_meta_matAssign2 {
public:
    static const int go = (J < N_columns - 1) ? 1 : 0;

    template<typename T_matrix, typename T_expr, typename T_updater>
    static inline void f(T_matrix& mat, T_expr expr, T_updater u)
    {
        u.update(mat(I, J), expr(I, J));
        _bz_meta_matAssign2<N_rows * go, N_columns * go, I * go, (J + 1) * go>
            ::f(mat, expr, u);
    }
};

template<>
class _bz_meta_matAssign2<0, 0, 0, 0> {
public:
    template<typename T_matrix, typename T_expr, typename T_updater>
    static inline void f(T_matrix&, T_expr, T_updater)
    { }
};

template<int N_rows, int N_columns, int I>
class _bz_meta_matAssign {
public:
    static const int go = (I < N_rows - 1) ? 1 : 0;

    template<typename T_matrix, typename T_expr, typename T_updater>
    static inline void f(T_matrix& mat, T_expr expr, T_updater u)
    {
        _bz_meta_matAssign2<N_rows, N_columns, I, 0>::f(mat, expr, u);
        _bz_meta_matAssign<N_rows * go, N_columns * go, (I + 1) * go>
            ::f(mat, expr, u);
    }
};

template<>
class _bz_meta_matAssign<0, 0, 0> {
public:
    template<typename T_matrix, typename T_expr, typename T_updater>
    static inline void f(T_matrix&, T_expr, T_updater)
    { }
};


BZ_NAMESPACE_END

#endif // BZ_META_ASSIGN_H

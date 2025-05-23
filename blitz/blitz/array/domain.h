// -*- C++ -*-
/***************************************************************************
 * blitz/array/domain.h  Declaration of the RectDomain class
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
#ifndef BZ_DOMAIN_H
#define BZ_DOMAIN_H

#include <blitz/blitz.h>
#include <blitz/et/-forward.h>
#include <blitz/range.h>

/*
 * Portions of this class were inspired by the "RectDomain" class
 * provided by the Titanium language (UC Berkeley).
 */

BZ_NAMESPACE(blitz)

template<int N_rank>
class RectDomain {

    typedef TinyVector<int, N_rank> Bounds;

public:

    RectDomain() { }
    RectDomain(const Bounds& lbound, const Bounds& ubound): lbound_(lbound), ubound_(ubound) { }
    RectDomain(const TinyVector<Range, N_rank>& bndrange): lbound_(), ubound_() {
        for (int i = 0;i < N_rank;++i) {
            lbound_(i) = bndrange(i).first();
            ubound_(i) = bndrange(i).last();
        }
    }

    // NEEDS_WORK: better constructors
    // RectDomain(Range, Range, ...)
    // RectDomain with any combination of Range and int

          Bounds& lbound()       { return lbound_; }
          Bounds& ubound()       { return ubound_; }
    const Bounds& lbound() const { return lbound_; }
    const Bounds& ubound() const { return ubound_; }

    int& lbound(const int i)       { return lbound_(i); }
    int& ubound(const int i)       { return ubound_(i); }
    int  lbound(const int i) const { return lbound_(i); }
    int  ubound(const int i) const { return ubound_(i); }

    Range operator[](const int rank) const { return Range(lbound_(rank), ubound_(rank)); }

    void shrink(const int amount) {
        lbound_ += amount;
        ubound_ -= amount;
    }

    void shrink(const int dim, const int amount) {
        lbound_(dim) += amount;
        ubound_(dim) -= amount;
    }

    void expand(const int amount) {
        lbound_ -= amount;
        ubound_ += amount;
    }

    void expand(const int dim, const int amount) {
        lbound_(dim) -= amount;
        ubound_(dim) += amount;
    }

private:

    Bounds lbound_;
    Bounds ubound_;
};

/*
 * StridedDomain added by Julian Cummings
 */

template<int N_rank>
class StridedDomain {

    typedef TinyVector<int, N_rank> Bounds;
    typedef TinyVector<diffType, N_rank> Strides;

public:

    StridedDomain(const Bounds& lbound, const Bounds& ubound, const Strides& stride):
        lbound_(lbound), ubound_(ubound), stride_(stride) { }

    // NEEDS_WORK: better constructors
    // StridedDomain(Range, Range, ...)
    // StridedDomain with any combination of Range and int

    const Bounds&  lbound() const { return lbound_; }
    const Bounds&  ubound() const { return ubound_; }
    const Strides& stride() const { return stride_; }

    int lbound(const int i) const { return lbound_(i); }
    int ubound(const int i) const { return ubound_(i); }
  diffType stride(const int i) const { return stride_(i); }

    Range operator[](const int rank) const { return Range(lbound_(rank), ubound_(rank), stride_(rank)); }

    void shrink(const int amount) {
        lbound_ += amount * stride_;
        ubound_ -= amount * stride_;
    }

    void shrink(const int dim, const int amount) {
        lbound_(dim) += amount * stride_(dim);
        ubound_(dim) -= amount * stride_(dim);
    }

    void expand(const int amount) {
        lbound_ -= amount * stride_;
        ubound_ += amount * stride_;
    }

    void expand(const int dim, const int amount) {
        lbound_(dim) -= amount * stride_(dim);
        ubound_(dim) += amount * stride_(dim);
    }

private:

    Bounds  lbound_;
    Bounds  ubound_;
    Strides stride_;
};


template<int N_rank>
inline RectDomain<N_rank>
strip(const TinyVector<int, N_rank>& startPosition, const int stripDimension, const int ubound) {
    BZPRECONDITION((stripDimension >= 0) && (stripDimension < N_rank));
    BZPRECONDITION(ubound >= startPosition(stripDimension));

    TinyVector<int, N_rank> endPosition = startPosition;
    endPosition(stripDimension) = ubound;
    return RectDomain<N_rank>(startPosition, endPosition);
}

BZ_NAMESPACE_END

#endif // BZ_DOMAIN_H

// -*- C++ -*-
/***************************************************************************
 * blitz/array/ops.h  Array operators
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
#ifndef BZ_ARRAY_OPS_H
#define BZ_ARRAY_OPS_H

#include <blitz/ops.h>
#include <blitz/funcs.h>
#include <blitz/array/newet-macros.h>

BZ_NAMESPACE(blitz)

// unary operators

BZ_DECLARE_ARRAY_ET_UNARY(operator~, BitwiseNot)
BZ_DECLARE_ARRAY_ET_UNARY(operator!, LogicalNot)
BZ_DECLARE_ARRAY_ET_UNARY(operator+, UnaryPlus)
BZ_DECLARE_ARRAY_ET_UNARY(operator-, UnaryMinus)

// binary operators

BZ_DECLARE_ARRAY_ET_BINARY(operator+,  Add)
BZ_DECLARE_ARRAY_ET_BINARY(operator-,  Subtract)
BZ_DECLARE_ARRAY_ET_BINARY(operator*,  Multiply)
BZ_DECLARE_ARRAY_ET_BINARY(operator/,  Divide)
BZ_DECLARE_ARRAY_ET_BINARY(operator%,  Modulo)
BZ_DECLARE_ARRAY_ET_BINARY(operator^,  BitwiseXor)
BZ_DECLARE_ARRAY_ET_BINARY(operator&,  BitwiseAnd)
BZ_DECLARE_ARRAY_ET_BINARY(operator|,  BitwiseOr)
BZ_DECLARE_ARRAY_ET_BINARY(operator>>, ShiftRight)
BZ_DECLARE_ARRAY_ET_BINARY(operator<<, ShiftLeft)
BZ_DECLARE_ARRAY_ET_BINARY(operator>,  Greater)
BZ_DECLARE_ARRAY_ET_BINARY(operator<,  Less)
BZ_DECLARE_ARRAY_ET_BINARY(operator>=, GreaterOrEqual)
BZ_DECLARE_ARRAY_ET_BINARY(operator<=, LessOrEqual)
BZ_DECLARE_ARRAY_ET_BINARY(operator==, Equal)
BZ_DECLARE_ARRAY_ET_BINARY(operator!=, NotEqual)
BZ_DECLARE_ARRAY_ET_BINARY(operator&&, LogicalAnd)
BZ_DECLARE_ARRAY_ET_BINARY(operator||, LogicalOr)

// \todo are these for Arrays of TinyVectors? How do we distinguish these from element - wise operations? they must now be done using the scalar() function
/*
// Declare binary ops between Array and "scalar-like" TinyVector
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator+,  Add)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator-,  Subtract)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator*,  Multiply)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator/,  Divide)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator%,  Modulo)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator^,  BitwiseXor)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator&,  BitwiseAnd)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator|,  BitwiseOr)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator>,  Greater)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator<,  Less)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator>=, GreaterOrEqual)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator<=, LessOrEqual)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator==, Equal)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator!=, NotEqual)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator&&, LogicalAnd)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(operator||, LogicalOr)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC((min), Min)
BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC((max), Max)
*/

#define BZ_DECLARE_ARRAY_ET_SCALAR_OPS(sca)                            \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator+,  Add, sca)                \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator-,  Subtract, sca)           \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator*,  Multiply, sca)           \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator/,  Divide, sca)             \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator%,  Modulo, sca)             \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator^,  BitwiseXor, sca)         \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator&,  BitwiseAnd, sca)         \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator|,  BitwiseOr, sca)          \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator>>, ShiftRight, sca)	       \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator<<, ShiftLeft, sca)	       \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator>,  Greater, sca)            \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator<,  Less, sca)               \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator>=, GreaterOrEqual, sca)     \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator<=, LessOrEqual, sca)        \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator==, Equal, sca)              \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator!=, NotEqual, sca)           \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator&&, LogicalAnd, sca)         \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(operator||, LogicalOr, sca)          \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR((min), Min, sca)		       \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR((max), Max, sca)

BZ_DECLARE_ARRAY_ET_SCALAR_OPS(char)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(unsigned char)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(short)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(unsigned short)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(int)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(unsigned int)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(long)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(unsigned long)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(float)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(double)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(long double)
#ifdef BZ_HAVE_COMPLEX
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(complex<float>)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(complex<double>)
BZ_DECLARE_ARRAY_ET_SCALAR_OPS(complex<long double>)
#endif


BZ_NAMESPACE_END

#endif // BZ_ARRAY_OPS_H

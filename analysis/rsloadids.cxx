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

#include "rsloadids.h"
#include "io/iochunk.h"

void ServiceRsLoadIds::Read(PST pst, uint64_t iElement, const std::string &filename, uint64_t iBeg, uint64_t iEnd) {
    pst->plcl->pkd->RsIdRead(iElement, filename, iBeg, iEnd);
}

void ServiceRsLoadIds::start(PST pst, uint64_t nElements, void *vin, int nIn) {
    auto in = static_cast <input *> (vin);
    pst->plcl->pkd->RsIdStart(nElements, in->bAppend);
}

void ServiceRsLoadIds::finish(PST pst, uint64_t nElements, void *vin, int nIn) {
    pst->plcl->pkd->RsIdFinish(nElements);
}

void pkdContext::RsIdStart(uint64_t nElements, bool bAppend) {
    if (!bAppend) {
        nRsElements = 0;
    }
}

void pkdContext::RsIdFinish(uint64_t nElements) {}

void pkdContext::RsIdRead(uint64_t iElement, const std::string &filename, uint64_t iBeg, uint64_t iEnd) {
    auto ids = static_cast <uint64_t *> (pLite);
    auto nRead = iEnd - iBeg;
    auto nBytes = nRead * sizeof(uint64_t);
    auto iOffset = iBeg * sizeof(uint64_t);
    io_chunk_read(filename.c_str(), ids + nRsElements, nBytes, iOffset);
    nRsElements += nRead;
}
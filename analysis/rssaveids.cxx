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

#include "rssaveids.h"
#include "io/iochunk.h"

void ServiceRsSaveIds::Write(PST pst, void *vin, int nIn, int iGroup, const std::string &filename, int iSegment, int nSegment) {
    pst->plcl->pkd->RsIdSave(iGroup, filename, iSegment, nSegment);
}

void pkdContext::RsIdSave(int iGroup, const std::string &filename, int iSegment, int nSegment) {
    auto ids = static_cast <uint64_t *> (pLite);
    auto nBytes = nRsElements * sizeof(ids[0]);
    io_chunk_write(filename.c_str(), ids, nBytes, (iSegment > 0));
}

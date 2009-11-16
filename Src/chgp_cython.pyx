from numpy import array
from numpy cimport ndarray

cdef extern from "chgp.h":

    double contr_hrr(int lena, double xa, double ya, double za, double *anorms,
         int la, int ma, int na, double *aexps, double *acoefs,
         int lenb, double xb, double yb, double zb, double *bnorms,
         int lb, int mb, int nb, double *bexps, double *bcoefs,
         int lenc, double xc, double yc, double zc, double *cnorms,
         int lc, int mc, int nc, double *cexps, double *ccoefs,
         int lend, double xd, double yd, double zd, double *dnorms,
         int ld, int md, int nd, double *dexps, double *dcoefs)


def contr_coulomb(*args):
    # convert args to numpy arrays
    args = [array(x) for x in args]
    cdef ndarray aexps, acoefs, anorms, xyza, lmna, \
            bexps, bcoefs, bnorms, xyzb, lmnb, \
            cexps, ccoefs, cnorms, xyzc, lmnc, \
            dexps, dcoefs, dnorms, xyzd, lmnd
    aexps, acoefs, anorms, xyza, lmna, \
            bexps, bcoefs, bnorms, xyzb, lmnb, \
            cexps, ccoefs, cnorms, xyzc, lmnc, \
            dexps, dcoefs, dnorms, xyzd, lmnd = args
    # a few sanity checks:
    lena = len(aexps)
    assert lena == len(acoefs)
    assert lena == len(anorms)

    lenb = len(bexps)
    assert lenb == len(bcoefs)
    assert lenb == len(bnorms)

    lenc = len(cexps)
    assert lenc == len(ccoefs)
    assert lenc == len(cnorms)

    lend = len(dexps)
    assert lend == len(dcoefs)
    assert lend == len(dnorms)

    xa, ya, za = xyza
    xb, yb, zb = xyzb
    xc, yc, zc = xyzc
    xd, yd, zd = xyzd

    la, ma, na = lmna
    lb, mb, nb = lmnb
    lc, mc, nc = lmnc
    ld, md, nd = lmnd

    Jij = contr_hrr(
            lena, xa, ya, za, <double *>(anorms.data), la, ma, na,
                <double *>(aexps.data), <double *>(acoefs.data),
            lenb, xb, yb, zb, <double *>(bnorms.data), lb, mb, nb,
                <double *>(bexps.data), <double *>(bcoefs.data),
            lenc, xc, yc, zc, <double *>(cnorms.data), lc, mc, nc,
                <double *>(cexps.data), <double *>(ccoefs.data),
            lend, xd, yd, zd, <double *>(dnorms.data), ld, md, nd,
                <double *>(dexps.data), <double *>(dcoefs.data)
                )
    return Jij

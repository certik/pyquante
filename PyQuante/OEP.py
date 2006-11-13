#!/usr/bin/env python

# Yang/Wu's OEP implementation, in PyQuante.

from math import sqrt
from PyQuante.NumWrap import zeros,matrixmultiply,transpose,dot,identity,\
     array
from PyQuante.Ints import getbasis, getints, getJ
from PyQuante.LA2 import GHeigenvectors,mkdens,TraceProperty
from PyQuante.hartree_fock import get_fock
from PyQuante.CGBF import three_center
from PyQuante.optimize import fmin,fminBFGS,fminNCG,fminBFGS2
from PyQuante.fermi_dirac import get_efermi, get_fermi_occs,mkdens_occs,\
     get_entropy
from PyQuante import logging

gradcall=0

def exx(atoms,orbs,**opts):
    return oep_hf(atoms,orbs,**opts)

def oep_hf(atoms,orbs,**opts):
    """oep_hf - Form the optimized effective potential for HF exchange.
       See notes on options and other args in oep routine.
    """
    return oep(atoms,orbs,get_exx_energy,get_exx_gradient,**opts)

def oep(atoms,orbs,energy_func,grad_func=None,**opts):
    """oep - Form the optimized effective potential for a given energy expression

    oep(atoms,orbs,energy_func,grad_func=None,**opts)

    atoms       A Molecule object containing a list of the atoms
    orbs        A matrix of guess orbitals
    energy_func The function that returns the energy for the given method
    grad_func   The function that returns the force for the given method

    Options
    -------
    verbose       False   Output terse information to stdout (default)
                  True    Print out additional information 
    ETemp         False   Use ETemp value for finite temperature DFT (default)
                  float   Use (float) for the electron temperature
    bfs           None    The basis functions to use. List of CGBF's
    basis_data    None    The basis data to use to construct bfs
    integrals     None    The one- and two-electron integrals to use
                          If not None, S,h,Ints
    opt_method    BFGS    Use BFGS for the OEP optimization
                  NM      Use Nelder-Mead (Simplex, Amoeba) for the OEP opt
                  CG      Use Conjugate-Gradient for the OEP optimization
                  BFGS2   Rick's experimental BFGS optimizer
    """
    verbose = opts.get('verbose',False)
    ETemp = opts.get('ETemp',False)
    opt_method = opts.get('opt_method','BFGS')

    bfs = opts.get('bfs',None)
    if not bfs:
        basis = opts.get('basis',None)
        bfs = getbasis(atoms,basis)

    # The basis set for the potential can be set different from
    #  that used for the wave function
    pbfs = opts.get('pbfs',None) 
    if not pbfs: pbfs = bfs
    npbf = len(pbfs)

    integrals = opts.get('integrals',None)
    if integrals:
        S,h,Ints = integrals
    else:
        S,h,Ints = getints(bfs,atoms)

    nel = atoms.get_nel()
    nocc,nopen = atoms.get_closedopen()

    Enuke = atoms.get_enuke()

    # Form the OEP using Yang/Wu, PRL 89 143002 (2002)
    nbf = len(bfs)
    norb = nbf
    bp = zeros(nbf,'d')

    bvec = opts.get('bvec',None)
    if bvec:
        assert len(bvec) == npbf
        b = array(bvec)
    else:
        b = zeros(npbf,'d')


    # Form and store all of the three-center integrals
    # we're going to need.
    # These are <ibf|gbf|jbf> (where 'bf' indicates basis func,
    #                          as opposed to MO)
    # N^3 storage -- obviously you don't want to do this for
    #  very large systems
    Gij = []
    for g in range(npbf):
        gmat = zeros((nbf,nbf),'d')
        Gij.append(gmat)
        gbf = pbfs[g]
        for i in range(nbf):
            ibf = bfs[i]
            for j in range(i+1):
                jbf = bfs[j]
                gij = three_center(ibf,gbf,jbf)
                gmat[i,j] = gij
                gmat[j,i] = gij

    # Compute the Fermi-Amaldi potential based on the LDA density.
    # We're going to form this matrix from the Coulombic matrix that
    # arises from the input orbitals. D0 and J0 refer to the density
    # matrix and corresponding Coulomb matrix
    
    D0 = mkdens(orbs,0,nocc)
    J0 = getJ(Ints,D0)
    Vfa = (2*(nel-1.)/nel)*J0
    H0 = h + Vfa

    if opt_method == 'BFGS': 
        b = fminBFGS(energy_func,b,grad_func,
                     (nbf,nel,nocc,ETemp,Enuke,S,h,Ints,H0,Gij),
                     logger=logging)
    elif opt_method == 'BFGS2': 
        b = fminBFGS2(energy_func,b,grad_func,
                     (nbf,nel,nocc,ETemp,Enuke,S,h,Ints,H0,Gij),
                     logger=logging)
    elif opt_method == 'NM': 
        b = fmin(energy_func,b,
                 (nbf,nel,nocc,ETemp,Enuke,S,h,Ints,H0,Gij),
                 logger=logging,maxiter=1000*nbf,maxfun=1000*nbf)
    elif opt_method == 'CG':
        print "Warning: OEP/CG optimization is still experimental"
        b = fminNCG(energy_func,b,grad_func,None,None,
                    (nbf,nel,nocc,ETemp,Enuke,S,h,Ints,H0,Gij),
                    logger=logging)
    else:
        raise "Unknown OEP optimization method: %s" % opt_method

    energy,orbe,orbs = energy_func(b,nbf,nel,nocc,ETemp,Enuke,
                                   S,h,Ints,H0,Gij,return_flag=1)
    return energy,orbe,orbs


def get_exx_energy(b,nbf,nel,nocc,ETemp,Enuke,S,h,Ints,H0,Gij,**opts):
    """Computes the energy for the OEP/HF functional

    Options:
    return_flag    0   Just return the energy
                   1   Return energy, orbe, orbs
                   2   Return energy, orbe, orbs, F
    """
    return_flag = opts.get('return_flag',0)
    Hoep = get_Hoep(b,H0,Gij)
    orbe,orbs = GHeigenvectors(Hoep,S)
        
    if ETemp:
        efermi = get_efermi(nel,orbe,ETemp)
        occs = get_fermi_occs(efermi,orbe,ETemp)
        D = mkdens_occs(orbs,occs)
        entropy = get_entropy(occs,ETemp)
    else:
        D = mkdens(orbs,0,nocc)
        
    F = get_fock(D,Ints,h)
    energy = TraceProperty(h+F,D)+Enuke
    if ETemp: energy += entropy
    iref = nel/2
    gap = 627.51*(orbe[iref]-orbe[iref-1])

    logging.debug("EXX Energy, B, Gap: %10.5f %10.5f %10.5f"
                  % (energy,sqrt(dot(b,b)),gap))
    logging.debug("%s" % orbe)
    if return_flag == 1:
        return energy,orbe,orbs
    elif return_flag == 2:
        return energy,orbe,orbs,F
    return energy

def get_exx_gradient(b,nbf,nel,nocc,ETemp,Enuke,S,h,Ints,H0,Gij,**opts):
    """Computes the gradient for the OEP/HF functional.

    return_flag    0   Just return gradient
                   1   Return energy,gradient 
                   2   Return energy,gradient,orbe,orbs 
    """
    # Dump the gradient every 10 steps so we can restart...
    global gradcall
    gradcall += 1
    if gradcall % 5 == 0:
        logging.debug("B vector:\n%s" % b)

    # Form the new potential and the new orbitals
    energy,orbe,orbs,F = get_exx_energy(b,nbf,nel,nocc,ETemp,Enuke,
                                        S,h,Ints,H0,Gij,return_flag=2)

    Fmo = matrixmultiply(orbs,matrixmultiply(F,transpose(orbs)))

    norb = nbf
    bp = zeros(nbf,'d') # dE/db

    for g in range(nbf):
        # Transform Gij[g] to MOs. This is done over the whole
        #  space rather than just the parts we need. I can speed
        #  this up later by only forming the i,a elements required
        Gmo = matrixmultiply(orbs,matrixmultiply(Gij[g],
                                                 transpose(orbs)))

        # Now sum the appropriate terms to get the b gradient
        for i in range(nocc):
            for a in range(nocc,norb):
                bp[g] = bp[g] + Fmo[i,a]*Gmo[i,a]/(orbe[i]-orbe[a])

    logging.debug("EXX  Grad: %10.5f" % (sqrt(dot(bp,bp))))
    return_flag = opts.get('return_flag',0)
    if return_flag == 1:
        return energy,bp
    elif return_flag == 2:
        return energy,bp,orbe,orbs
    return bp

def get_Hoep(b,H0,Gij):
    Hoep = H0
    # Add the contributions from the gaussian potential functions
    # H[ij] += b[g]*<ibf|g|jbf>
    for g in range(len(b)):
        Hoep = Hoep + b[g]*Gij[g]
    return Hoep

def update_bfgs(X,G,HIo,Xo,Go,**opts):
    # Numerical recipes BFGS routine 
    N = len(X)
    stpmax = opts.get('stpmax',0.1)
    stpmaxn = opts.get('stpmaxn',N*stpmax)

    if HIo is None:
        # Inverse Hessian values of .7 correspond to Hessians of
        # 1.4 Hartree/Angstrom**2 - about right                 
        HIo = 0.7*identity(N,'d')

    if Go is None or Xo is None:
        ChgeX = -0.7*G
        HI = HIo
    else:
        DG = G-Go
        DX = X-Xo
        HDG = dot(HIo,DG)
        fac = 1./dot(DG,DX)
        fae = dot(DG,HDG)
        fad = 1./fae
        w = fac*DX - fad*HDG
        HI = zeros((N,N),'d')
        for i in range(N):
            for j in range(N):
                HI[i,j] = HIo[i,j] \
                          + fac*DX[i]*DX[j] \
                          - fad*HDG[i]*HDG[j] \
                          - fae*w[i]*w[j]
        ChgeX = -dot(HI,G)

    stpl = sqrt(dot(ChgeX,ChgeX))
    if stpl > stpmaxn: ChgeX *= stpmaxn/stpl

    lgstst = max(abs(ChgeX))
    if lgstst > stpmax: ChgeX *= stpmax/lgstst

    Xn = X + ChgeX
    return Xn,HI

if __name__ == '__main__': main()
#!/usr/bin/env python
"H2O using Gaussians"

# Jaguar results for 6-31G** basis and 1 bohr distance:
# Energy -76.005 909
#
# GAMESS-UK results for above 
# Energy -76.01175904

import time
from PyQuante.Ints import getbasis,getints
from PyQuante.hartree_fock import rhf
from PyQuante.Molecule import Molecule

energy = -76.011751 # Changed 2003-04-07 to reflect DIIS results
name = "H2O"

def main():
    r = 1./0.52918
    h2o=Molecule('h2o',atomlist = [(8,(0,0,0)),(1,(r,0,0)),(1,(0,r,0))])
    en,orbe,orbs = rhf(h2o)
    return en


def profmain():
    import profile,pstats
    profile.run('main()','prof')
    prof = pstats.Stats('prof')
    prof.strip_dirs().sort_stats('time').print_stats(15)

if __name__ == '__main__':
    t0 = time.time()
    en = main()
    #profmain()
    t1 = time.time()
    print en,energy,t1-t0


    
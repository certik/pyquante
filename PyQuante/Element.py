#!/usr/bin/env python
"""\
 Elements.py: Miscellaneous data about elements

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""

symbol = [
    'X','H','He',
    'Li','Be','B','C','N','O','F','Ne',
    'Na','Mg','Al','Si','P','S','Cl','Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe']


sym2no = {}
for i in range(len(symbol)):
    sym2no[symbol[i]] = i
    sym2no[symbol[i].lower()] = i
    
name = [
    'dummy',
    'hydrogen', 'helium',
    'lithium','beryllium','boron','carbon','nitrogen',
    'oxygen','fluorine','neon','sodium','magnesium',
    'aluminum','silicon','phosphorus','sulfur','chlorine',
    'argon','potassium', 'calcium', 'scandium', 'titanium',
    'vanadium', 'chromium', 'manganese', 'iron',
    'cobalt', 'nickel', 'copper', 'zinc',
    'gallium', 'germanium', 'arsenic', 'selenium', 'bromine',
    'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium',
    'niobium', 'molybdenum', 'technetium', 'ruthenium','rhodium',
    'palladium', 'silver', 'cadmium','indium', 'tin', 'antimony',
    'tellerium', 'iodine', 'xenon']

name2no = {}
for i in range(len(name)):
    name2no[name[i]] = i
    name2no[name[i].upper()] = i

mass = [
    0.00,
    1.0008, 4.0026,
    6.941,9.0122,
    10.811,12.011,14.007,15.999,18,998,20.179,
    22.990,24.305,
    26.982,28.086,30.974,32.066,35.453,39.948,
    39.098, 40.078,
    44.9559, 47.867, 50.9415, 51.9961, 54.938, 55.845,
    58.9332, 58.6934, 63.546,65.39,
    69.723, 72.61, 74.9216, 78.96, 79.904, 83.80,
    85.4678, 87.62,
    88.90686, 91.224, 92.90638, 95.94, 98, 101.07,
    102.90550, 106.42, 107.8682, 112.411,
    114.818, 118.710, 121.760, 127.60, 126.90447, 131.29]


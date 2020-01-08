#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:52:14 2018

@author: monvilre

Parameters
"""

import numpy as np


E = 1e-4 # Nombre d'Ekman
P = 0.3 # Nombre de Prandt
S =  3 # Nombre de Schmidt

L = S/P # Nombre de Lewis
Gamma = 1.76

eta = Gamma/E
#500 80
nbr =200# Nombre de points nbr*2.5
llog =10 # Limite en échelle log = 10^llog
trh = 4
llin = 5e6 # Limite en échelle lin

mmax = 60 # Mode max 
lmax = 60000

ze = [0]
#M= np.logspace(0,np.log10(mmax),int(mmax/5000)) # liste des nombres d'ondes azimutaux
#M =np.concatenate((ze,M))
M = np.linspace(0,mmax,mmax)
#M = [1,2,3,4,5]
#LL = np.logspace(0,np.log10(lmax),int(lmax/8000)) # liste des nombres d'ondes radiaux

LL = [1,2,4]

# Liste des modes
#M= np.logspace(0,np.log10(mmax),50)
log = 1 # 0:Echelle linéaire 1:Echelle logarithmique


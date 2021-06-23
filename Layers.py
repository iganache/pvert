#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:30:15 2021

@author: indujaa
"""

import numpy as np
from scipy.constants import *
import cmath

class Layers:

    def __init__(self, depth = 0, eps = complex(0,0), emrough = 0, corr_length = 0.1):
        self.d = depth
        self.eps = eps
        self.ri = self.eps ** 0.5
        self.upperks = emrough
        self.corrlen = corr_length
        
        self.theta_i = 0.
        self.theta_t = 0.
        self.theta_s = 0.

        # # Fresnel coefficients for layers
        self.rho = {}
        self.gamma = {}
        self.tau = {}
        self.T = {}
        
        # # Surface backscatter 
        self.surf_bsc = {}
        
        
    class inclusions():
        
        def __init__(self, eps = complex(0,0), radius = 0., volfrac = 0., axisratio = 1., alpha=45., beta=45.):
            self.eps = eps
            self.ri = self.eps ** 0.5
            self.a = radius
            self.vol = (4/3) * pi * self.a**3              # # only for spherical scatterers                        
            self.Dmax = .03
            self.nw = 100
            self.alpha = np.deg2rad(alpha)
            self.beta = np.deg2rad(beta)
            self.axratio = axisratio
            
        def set_properties(self):
            self.ri = self.eps ** 0.5
#             self.n0 = self.volfrac / self.vol 
            

    def make_layers():
        atm = Layers(eps = complex(1, 0))
        
        rock = Layers(eps = complex(6, 0.05), 
                      emrough = 0.002, corr_length = 12.6e-2)
        
        ash = Layers(depth = 5, 
                     eps = complex(2.7, 0.003), 
                     emrough = 0.01, 
                     corr_length = 12.6e-2)
        
        ash.inclusions = ash.inclusions(eps = complex(6,0.01), 
                                        radius = 1e-2, 
                                        axisratio = 1.26/0.63, 
                                        alpha=45., beta=45.)
        
        return [atm, ash, rock]
    
    
    def set_properties(self):
        self.ri = self.eps ** 0.5

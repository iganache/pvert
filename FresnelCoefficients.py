#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:39:40 2021

Computes Fresnel reflectiona and Transmission COefficients 
for a scattering model based on

@author: indujaa
"""

import numpy as np

def FresnelH(eps1, eps2, theta_i):
    """ formula from Ulaby and Long, 2014 - page 61 (or) 476 """
    mu1 = np.cos(theta_i)
    n = np.sqrt(eps2/eps1)
        
    mu2 = np.sqrt(1.0 - (1.0 - mu1**2) / n**2).real
        
    refh = (mu1 - n * mu2) / (mu1 + n * mu2)

    transh = 2*mu1 / (mu1 + n*mu2)
#       transh = 1 + refh
    rh = refh.real**2 + refh.imag**2
#       th - 1 - rh
    th = (n*mu2/mu1).real * np.abs(transh) ** 2

    return np.array([refh, transh, rh, th])
       
    
def FresnelV(eps1, eps2, theta_i):
    """ formula from Ulaby and Long, 2014 - page 61 (or) 476 """
    mu1 = np.cos(theta_i)
    n = np.sqrt(eps2/eps1)
        
    mu2 = np.sqrt(1.0 - (1.0 - mu1**2) / n**2).real
        
    refv = (mu2 - n * mu1) / (mu2 + n * mu1)

    transv = 2*mu1 / (mu2 + n*mu1)
#       transv = (1 + refv) * mu2 / mu1
    rv = refv.real**2 + refv.imag**2
#         tv = 1 - rv
    tv = (n*mu2/mu1).real * np.abs(transv) ** 2

    return np.array([refv, transv, rv, tv])


def TransmissionAngle(eps_1, eps_2, theta_i):
        mu1 = np.cos(theta_i)
        n = np.sqrt(eps_2/eps_1)
        
        mu2 = np.sqrt(1.0 - ((1.0 - mu1**2) / n**2)).real
        theta_t = np.arccos(mu2)
#        theta_t = np.arcsin(l1.ri.real * np.sin(l2.theta_i) / l2.ri.real)
        return theta_t


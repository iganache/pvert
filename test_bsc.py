#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for testing rough surface backscattering from an interface

@author: indujaa
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# # VRT, Layer and Rough surface class objects
from Layers import Layers
from VRT import VRT
from RoughSurface import RoughSurface
import FresnelCoefficients as Fresnel

def VRT_values(inc):
    """
    Function for accessing and getting values from the VRT code
    """
    layers = Layers.make_layers()
    vrt = VRT(layers)                                      # the different layers can now be accessed as vrt.atm, vrt.l1, vrt.l2
    
    wv = 12.6e-2
    ruffsurf = RoughSurface(wv, vrt.atm.eps, vrt.l1.eps, vrt.l1.upperks, vrt.l1.corrlen, autocorr_fn = "exponential", theta_i = vrt.l1.theta_i, theta_s = vrt.l1.theta_s)    
    svv, shh = ruffsurf.Copol_BSC()
    
    return svv, shh


def ref_values():
    """
    Function for computing scatteing from a smooth planar surface
    """
    pass


def plot_output(xvalues, vrtvalues, refvalues):
    """
    Function for visualizing the test results
    """
    
    plt.figure()
    plt.plot(xvalues, vrtvalues, xvalues, refvalues)
    plt.show()
    

def write_output(xvalues, vrtvalues, refvalues):
    """
    Function for writing the test results to a CSV file
    """
    pass

def main():
    
    vrt_list = []
    
    inc_angles = np.linspace([0,80,10])
    for i in inc_angle:
        a,b = VRT_values(i)
        vrt_list.add(a)
        
    
    svv, shh = VRT_values()
    svv_ref, shh_ref = ref_values()
    
    ## plot output
    ## write ourput
    

if __name__ == '__main__':
    main()
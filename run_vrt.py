#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:33:21 2021
Range of vlaues for different parameters

'theta_i': 0-8,'d':.05-20, 'eps_1': 3-10, 'eps_2':6-15, 'eps_inc':6-15, 
'n0_inc': 0-5000, 'a_inc':.001-.1, 'axratio_inc': 1-0.1, 'emrough1':0.001-0.036, 'emrough2':0.001-0.036


@author: indujaa
"""
import numpy as np
from VRT_module import VRT
from plotting import plotting
from VRTwrapper import VRTmodel
from itertools import product

def generate_dictlist(**kwargs):
    """ Takes in a dictionary with multiple values per key and returns mutiple 
    dictionaries each contaioning a permutation of key-value pairs """
    
    keys = list(kwargs.keys())
    vals = list(kwargs.values())
    sweep = list(product(*vals))
    dicts = []
    for valset in sweep:
        dicts.append(dict(zip(keys, valset)))

    return dicts
        

def main():
    """ Contains all steps needed to determine backscatter and emission. 
    The steps can also be run on command line."""

    # # # STEP 1
    # # # Create an instance of class VRTmodel with number of cores available as input
    vrtmodel = VRTmodel(nproc = 10)
    
       
    # # # STEP 2 
    # # # Create a dictionary with values for all input model parameters
    # # # multuiple values per key must be entered as a list
    
    
    # # # EXAMPLES
    
    # # scene 1 - - fluffy with rough surface
    # # scattering mechanisms - surface
#     dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 10)),
#                                   d=[0.126], 
#                                   atm_eps = [1+0j], 
#                                   eps2r=[9], eps2i=[0.1],
#                                   eps1r=[5, 7], eps1i=[.01],
#                                   epsincr=[6], epsinci=[.05],
#                                   s1=[0.026, 0.04], s2=[0.008],
#                                   cl1 = [0.5], cl2 = [0.5],
#                                   psdfunc = ["exponential"],
#                                   n0 = [1], volfrac = [0.1], 
#                                   Dmax = [0.03], Lambda = [500], mu = [100],
#                                   a=[None], abyc= [1.0], 
#                                   alpha=[0.], beta=[0.])
    
    # # Sscene 1 - - effect of correlation length
#     dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 10)),
#                                   d=[0.126], 
#                                   atm_eps = [1+0j], 
#                                   eps2r=[9], eps2i=[0.1],
#                                   eps1r=[5], eps1i=[.01],
#                                   epsincr=[6], epsinci=[.05],
#                                   s1=[0.01, 0.04], s2=[0.008],
#                                   cl1 = [0.1, 0.5, 1.0], cl2 = [0.5],
#                                   psdfunc = ["exponential"],
#                                   n0 = [1], volfrac = [0.1], 
#                                   Dmax = [0.03], Lambda = [500], mu = [100],
#                                   a=[None], abyc= [1.0], 
#                                   alpha=[0.], beta=[0.])

        
        
    # # scene 2 - - mantled rough surface
    # scattering mechanisms - surface, subsurface
#     dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 20)),
#                                   d=[0.126, 0.5], 
#                                   atm_eps = [1+0j], 
#                                   eps2r=[8], eps2i=[.01],
#                                   eps1r=[2], eps1i=[.001],
#                                   epsincr=[6], epsinci=[.05],
#                                   s1=[0.01], s2=[0.026, 0.04],
#                                   cl1 = [0.5], cl2 = [0.5],
#                                   psdfunc = ["exponential"],
#                                   n0 = [1], volfrac = [0.1], 
#                                   Dmax = [0.03], Lambda = [500], mu = [100],
#                                   a=[None], abyc= [1.0], 
#                                   alpha=[0.], beta=[0.])
    
    
    # # scene 2 - -  equal contribution from surface roughness
    # scattering mechanisms - surface, subsurface
#     dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 20)),
#                                   d=[0.126], 
#                                   atm_eps = [1+0j], 
#                                   eps2r=[65], eps2i=[.01],
#                                   eps1r=[2], eps1i=[.001],
#                                   epsincr=[6], epsinci=[.05],
#                                   s1=[0.04], s2=[0.04],
#                                   cl1 = [0.5], cl2 = [0.5],
#                                   psdfunc = ["exponential"],
#                                   n0 = [1], volfrac = [0.1], 
#                                   Dmax = [0.03], Lambda = [500], mu = [100],
#                                   a=[None], abyc= [1.0], 
#                                   alpha=[0.], beta=[0.])


    # # scene 3 - volume scattering from rocks in fluffy medium 
    # # not using lossy / ferroeectric subsy=tarte as that could decrease emissivity further
    # # scattering mechanisms - surface, subsurface, volume, volume-subsurface

#     dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 20)),
#                                   d=[0.5, 5], 
#                                   atm_eps = [1+0j], 
#                                   eps1r = [2], eps1i = [0.05],
#                                   eps2r=[8], eps2i=[0.05], 
#                                   epsincr=[8], epsinci=[.05],
#                                   s1=[0.005, 0.01, 0.04], s2=[0.01],
#                                   cl1 = [0.5], cl2 = [0.5],
#                                   psdfunc = ["exponential"],
#                                   n0 = [1e8], volfrac = [.05], 
#                                   Dmax = [0.016], Lambda = [205], mu = [50],
#                                   a=[None], abyc= [1.5], 
#                                   alpha=[0.], beta=[0.])

    
   

   
    # # # STEP 3
    # # # Assign the types of scattering mechanisms to be modeled to the variable scattertypes as a list of lists
    # # # Acceptable values include "surface", "susbsurface", "volume", "volume-subsurface"
   
    scattertypes = [["surface", "subsurface"]]
    
    
    # # # STEP 4
    # # # Run the model using VRT_multiprocess in a multi-core machine 
    # # # Or run the model using VRT_singleprocess in a single-core machine
    # # # Save outout to a CSV file
    
    output_df = vrtmodel.VRT_multiprocess(dict_list, scattertypes)
    outfile = "ResultPlots/Case3/midrough/sce3-equal-new.csv"
    output_df.to_csv(outfile, sep = ',') 


if __name__ == '__main__':
    main()   

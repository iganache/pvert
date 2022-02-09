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
from Layers import Layers
from VRT_module import VRT
from plotting import plotting
from VRTwrapper import VRTmodel
from itertools import product

def generate_dictlist(**kwargs):
    keys = list(kwargs.keys())
    vals = list(kwargs.values())
    sweep = list(product(*vals))
    dicts = []
    for valset in sweep:
        dicts.append(dict(zip(keys, valset)))

    return dicts
        

def main():

    vrtmodel = VRTmodel(nproc = 10)
    
     # for different incidence angles
        
    # # scene 1 - - fluffy with rough surface
    # # scattering mechanisms - surface
    dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 10)),
                                  d=[0.126], 
                                  atm_eps = [1+0j], 
                                  eps2r=[9], eps2i=[0.1],
                                  eps1r=[5, 7], eps1i=[.01],
                                  epsincr=[6], epsinci=[.05],
                                  s1=[0.026, 0.04], s2=[0.008],
                                  cl1 = [0.5], cl2 = [0.5],
                                  psdfunc = ["exponential"],
                                  n0 = [1], volfrac = [0.1], 
                                  Dmax = [0.03], Lambda = [500], mu = [100],
                                  a=[None], abyc= [1.0], 
                                  alpha=[0.], beta=[0.])
    
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

        
        
    # # scene 2  - mantled rough surface
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

    # # scene 3 - volume scattering from rocks in fluffy medium (d = 100 m)
    # # not using lossy / ferroeectric subsy=tarte as that could decrease emissivity further
    # # scattering mechanisms - surface, subsurface, volume
    # # volume - gamma: Dmax = .026, mu = 50, N = 5e4
    # # volume - exponential: Dmax = .02, lambda = 300, N = 1e5
    # # volume - exponential: Dmax = .016, lambda=350, N = 1e8; vf = .01
    # # volume - exponential: Dmax = .016, lambda=140, N = 1e8; vf = .1
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

    
   

# # # inclusion permittivity
#     dict_list = generate_dictlist(thetai=list(np.linspace(0.0, 80.0, 20)),
#                                   d=[0.5], 
#                                   atm_eps = [1+0j], 
#                                   eps1r = [2], eps1i = [0.05],
#                                   eps2r=[8], eps2i=[0.05], 
#                                   epsincr=[8], epsinci=[0.05, 100],
#                                   s1=[0.004], s2=[0.01],
#                                   cl1 = [0.5], cl2 = [0.5],
#                                   psdfunc = ["exponential"],
#                                   n0 = [1e8], volfrac = [0.1], 
#                                   Dmax = [0.012], Lambda = [140], mu = [50],
#                                   a=[None], abyc= [1.5], 
#                                   alpha=[0.], beta=[0.])



    
    # # pass the scattering mechanisms as a list of lists 
    scattertypes = [["surface"]]
    output_df = vrtmodel.VRT_multiprocess(dict_list, scattertypes)
    myplt = plotting()
    
    
    
    # # Plotting - scene1 - fluffy with ruff surface
    outfile = "ResultPlots/Case1/surfaceBSC_Jan26.csv"
    output_df.to_csv(outfile, sep = ',') 
#     myplt.incplot(output_df, 'thetai', ['shh_sur'], ['ks1', 'cl1'], xlabel="Incidence angle", ylabel='$\sigma_{HH}$ (dB)',legend= '', data = "Mag_sigma.csv", xlim = [0, 80], ylim = [-40, 10], outfile = "ResultPlots/Case1/corrlen/BSCvsInc-corrlen.png")
#     myplt.incplot(output_df, 'thetai', ['eh_sur'], ['ks1', 'cl1'], xlabel="Incidence Angle", ylabel='$e_{H}$',legend= '', data = "Mag_emis.csv", ylim = [0.2, 1], outfile = "ResultPlots/Case1/corrlen/BSCvsemis-corrlen.png")

    
#     # # Plotting - scene2 - ruff surface buried by thin layer - no surface - this layer depth smaller than penetration depth
#     outfile = "ResultPlots/Case3/sce3-regeps2r-higheps2i.csv"
#     output_df.to_csv(outfile, sep = ',') 
#     myplt.incplot(output_df, 'thetai', ['shh_total'], ['ks2', 'd'], xlabel="Incidence angle", ylabel='$\sigma_{HH}$ (dB)',legend= '', data = "Mag_sigma.csv", ylim = [-40, 10], outfile = "ResultPlots/Case3/BSCvsInc-new-higheps2r.png")
#     myplt.incplot(output_df, 'thetai', ['eh_sub'], ['ks2', 'd'], xlabel="Incidence Angle", ylabel='$e_{H}$',legend= '', data = "Mag_emis.csv", ylim = [0.2, 1], outfile = "ResultPlots/Case3/EmisvsInc-new-higheps2r.png")

#     outfile = "ResultPlots/Case3/midrough/sce3-reg.csv"
#     output_df.to_csv(outfile, sep = ',') 
#     myplt.incplot(output_df, 'thetai', ['shh_total'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/Case3/midrough/BSCvsInc-reg.png")
#     myplt.incplot(output_df, 'thetai', ['eh_sub'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", ylim = [0.2, 1], legend= '', outfile = "ResultPlots/Case3/midrough/EmisvsInc-reg.png")

#     # # Plotting - scene3  
#     outfile = "ResultPlots/Case4/inc_permittivity/vol_vf1_inc_clast_Jan20.csv"
#     output_df.to_csv(outfile, sep = ',')
#     myplt.incplot(output_df, 'thetai', ['shh_total'], ['epsincr', 'epsinci'],  xlabel="Incidence Angle", ylabel='$\sigma_{HH}$',legend= '', data = "Mag_sigma.csv",  xlim = [5, 80], ylim = [-40, 5], outfile =  "ResultPlots/Case4/inc_permittivity/vol_vf1_inc_clast_BSCvsInc_Jan20.png")
#     myplt.incplot(output_df, 'thetai', ['eh_vol'], ['epsincr', 'epsinci'],  xlabel="Emission Angle", ylabel='$e_{H}$',legend= '', data = "Mag_emis.csv", xlim = [5, 80], ylim = [0.2, 1], outfile = "ResultPlots/Case4/inc_permittivity/vol_vf1_inc_clast_EmisvsInc_Jan20.png")

if __name__ == '__main__':
    main()   

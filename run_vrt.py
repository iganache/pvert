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
# from VRT_Tsang import VRT
from VRTwrapper import VRTmodel, plotCSV
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
    
    # for scene1 - surface scattering
#     dict_list = generate_dictlist(thetai=[45.7],
#                                   d=[0.05], 
#                                   atm_eps = [1+0j], 
#                                   eps1r=list(np.linspace(3, 7, 5)), eps1i=[0.005],
#                                   eps2r=[11], eps2i=[.003],
#                                   epsincr=[6], epsinci=[.05],
#                                   ks2=[0.003], ks1=list(np.linspace(0.001, 0.045,30)),
#                                   n0 = [2], a=[0.0], abyc= [0.6], alpha=[0.], beta=[0.])
    
     # for scene2 - surface scattering
#     dict_list = generate_dictlist(thetai=[45.7],
#                                   d=[0.05], 
#                                   atm_eps = [1+0j], 
#                                   eps1r=[5], eps1i=[0, 1, 10, 100],
#                                   eps2r=[11], eps2i=[.003],
#                                   epsincr=[6], epsinci=[.05],
#                                   ks2=[0.003], ks1=list(np.linspace(0.001, 0.045,30)),
#                                   n0 = [2], a=[0.0], abyc= [0.6], alpha=[0.], beta=[0.])
    
    # for scene3 - mantled rough surface scattering
    dict_list = generate_dictlist(thetai=[45.7],
                                  d=[0.063], 
                                  atm_eps = [1+0j], 
                                  eps2r=[7,9,11,13,15], eps2i=[0.1],
                                  eps1r=[3], eps1i=[.001],
                                  epsincr=[6], epsinci=[.05],
                                  ks1=[0.00], ks2=list(np.linspace(0.001, 0.045, 30)),
                                  n0 = [2], a=[0.0], abyc= [0.6], alpha=[0.], beta=[0.])
    
    
#     # for volume scattering
#     dict_list = generate_dictlist(thetai=[55],d=[1], atm_eps = [1+0j], eps2=[8+.005j], eps1=[4+.005j], 
#                               epsinc=list(eps), n0 = [2], a=list(np.linspace(0.01, 0.04, 4)), abyc= [1.5], 
#                               ks1=[0.001], ks2=[0.001],
#                                  alpha=[0.], beta=[0.])
    
    # # pass the scattering mechanisms as a list of lists 
    scattertypes = ['subsurface']
    output_df = vrtmodel.VRT_multiprocess(dict_list, scattertypes)
    
    # # Plotting - scene1
#     vrtmodel.lineplot(output_df, 'ks1', 'shh_sur', 'eps1r', xlabel="wavelength scale EM roughness", ylabel='$\sigma_{HH}$',legend= '$\epsilon$'+'\'', data = [-10, -14], outfile = "sce1-BSC.png")
#     vrtmodel.lineplot(output_df, 'ks1', 'eh_sur', 'eps1r', xlabel="wavelength scale EM roughness", ylabel='$e_{H}$',legend= '$\epsilon$'+'\'',data = [0.806, 0.876], outfile = "sce1-emis.png")
#     outfile = "sce1.csv"
#     output_df.to_csv(outfile, sep = ',') 

    # # Plotting - scene 2
#     vrtmodel.lineplot(output_df, 'ks1', 'shh_sur', 'eps1i', xlabel="wavelength scale EM roughness", ylabel='$\sigma_{HH}$',legend= '$\epsilon$'+'\"', data = [-10, -14], outfile = "sce2-BSC.png")
#     vrtmodel.lineplot(output_df, 'ks1', 'eh_sur', 'eps1i', xlabel="wavelength scale EM roughness", ylabel='$e_{H}$',legend= '$\epsilon$'+'\"', data = [0.806, 0.876], outfile = "sce2-emis.png")
#     outfile = "sce2.csv"
#     output_df.to_csv(outfile, sep = ',') 
    
    # # Plotting - scene3
    vrtmodel.lineplot(output_df, 'ks2', 'shh_sub', 'eps2r', xlabel="wavelength scale EM roughness", ylabel='$\sigma_{HH}$',legend= '$\epsilon$'+'\'',data = [-10, -14], outfile = "sce3-halfthic-BSC.png")
    vrtmodel.lineplot(output_df, 'ks2', 'eh_sub', 'eps2r', xlabel="wavelength scale EM roughness", ylabel='$e_{H}$',legend= '$\epsilon$'+'\'',data = [0.806, 0.876], outfile = "sce3-halfthic-emis.png")
    outfile = "sce3-halfthic.csv"
    output_df.to_csv(outfile, sep = ',')
    
#     outfile = "rough_fluffy.csv"
#     output_df.to_csv(outfile, sep = ',') 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     vrtmodel.plotsubBS_contour(output_df)
#     vrtmodel.plotsubemis_contour(output_df)
    
#     outfile = "subsurface_noinc_mat.csv"
#     output_df.to_csv(outfile, sep = ',')   
    

    # # For computing backscatter for a range of values
    
    # # possible variable names - 'Incidence angle','Depth', 'Layer permittivity', 'Substrate permittivity', 'Permittivity of scatterers', 
    # # 'Number concentration of scatterers', 'Maximum scatterer size', 'Axis ratio of scatterers', 'Surface roughness', 'Subsurface roughness'
#     var_name = 'Layer permittivity'

#     value = np.linspace(10,80,20)                # incidence angle
#     value = np.linspace(5e-2, 20, 10)           # depth
#     epsreal = np.linspace(4,10,2)              # permittivity variables
#     value = epsreal + 1j*0.003                  # permittivity variables
#     epsreal = 5
#     epsimg = 1j* np.array([.003,.03,.3])
#     value = epsreal + epsimg
    
#     value= np.linspace(1,.1,10)                # axis ratio of scatterers (takes a long time to run)
#     value = np.linspace(.02, 0.06, 5)         # scatterer size
#     value = np.linspace(0.001, 0.1, 10)         # scatterer size ? (upper limit seems large)
#     value = np.arange(0., 0.036, .004)          # interface roughness?
#     value = np.linspace(1, 10, 10)              # axis ratio of scatterers?
#     value = np.linspace(0, 5000, 10)            # number concentration of scatterers?
    
#    value = np.array([complex(2.7, 0.003), complex(4.2, 0.003), complex(5.8, 0.003), complex(7, 0.003)])
#    value = np.linspace(2e-3, 10e-2, 0)
    
    
    
    
    # # single thread
#     vrt.runModelVar(var_name, value)
#     vrt.plotOutput(var_name, value)
#     vrt.writeCSV("vrt_depth.csv", var_name, value)

    
    # # multithread
#     vrtmodel = VRTmodel(vrt, var_name, value)
#     vrtmodel.VRT_multiprocess()
#     vrtmodel.plotOutput()
#     vrtmodel.plotSurfaceBSC()
#     vrtmodel.writeCSV("vrt_depth.csv")
#    plotCSV("vrt_depth.csv")



if __name__ == '__main__':
    main()   
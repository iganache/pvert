#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:33:21 2021

@author: indujaa
"""
import numpy as np
from Layers import Layers
from VRT import VRT
# from VRT_Tsang import VRT
from VRTmodel import VRTmodel, plotCSV



def main():
    
    # # User defined
#    layers = [atmosphere(), layer(), substrate()]
    
    # # use defined parameetrs
    layers= Layers.make_layers()

    # # For computing backscatter for a range of values
    
    # # possible variable names - 'Incidence angle','Depth', 'Layer permittivity', 'Substrate permittivity', 'Permittivity of scatterers', 
    # # 'Number concentration of scatterers', 'Maximum scatterer size', 'Axis ratio of scatterers', 'Surface roughness', 'Subsurface roughness'
    var_name = 'Layer permittivity'

#     value = np.linspace(10,80,2)                # incidence angle
#     value = np.linspace(5e-2, 20, 10)           # depth
    epsreal = np.linspace(4,6,5)              # permittivity variables
    value = epsreal + 1j*0.003                  # permittivity variables
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
    
    vrt = VRT(layers)
    
    
    # # single thread
#     vrt.runModelVar(var_name, value)
#     vrt.plotOutput(var_name, value)
#     vrt.writeCSV("vrt_depth.csv", var_name, value)

    
    # # multithread
    vrtmodel = VRTmodel(vrt, var_name, value)
    vrtmodel.VRT_multiprocess()
    vrtmodel.plotOutput()
#     vrtmodel.plotFresnelCoef()
#     vrtmodel.plotSurfaceBSC()
#     vrtmodel.writeCSV("vrt_depth.csv")
#    plotCSV("vrt_depth.csv")



if __name__ == '__main__':
    main()   
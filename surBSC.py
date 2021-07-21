#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:33:21 2021

@author: indujaa
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Layers import Layers
from VRT import VRT
# from VRT_Tsang import VRT
from VRTmodel import VRTmodel, plotCSV

def get_MagellanBS(file):
    df = pd.read_csv(file, sep =',')
    inc = np.array(df['IncAngle'].tolist(), dtype=np.float32)
    bsc_mean = np.array(df['Mean'].tolist(), dtype=np.float32)
    bsc_std = np.array(df['Stdev'].tolist(), dtype=np.float32)
    return inc, bsc_mean, bsc_std

def plotBS_contour(eps, ks, shh, svv= None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    BSplot = ax.contourf(eps, ks, shh, cmap = plt.cm.bone)
    cbar = fig.colorbar(BSplot)
    cbar.ax.set_ylabel("Surface backscatter (dB)")

    plt.show()

def plotBS(var_name, values, svv, shh, inc, meanbs, stdbs):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(values, svv, label = 'VV', color = 'k')
    ax.plot(values, shh, label = 'HH', color = 'r')
    
    if var_name == 'Incidence angle':
        ax.scatter(inc, meanbs, marker = 'o', color = 'b')
    else:
        ax.axhline(y=np.mean(meanbs), color = 'darkgrey')
    
    ax.set_xlabel(var_name)
    ax.set_ylabel = ('$\sigma_{0}$')
    ax.legend()
    plt.show()

def main():
    
    # # User defined - all changes to be made in layers file
    layers= Layers.make_layers()

    # # For computing surface backscatter for a range of values
    # # Input parameter space would typically consist of the valriable:
    # # 'Incidence angle', 'Layer permittivity', 'Surface roughness'
    # #  currently set up two make filled contour plots as a function of 
    # # permittivity and roughness
    
    
    var_name = 'Layer permittivity'
    epsreal = np.linspace(2,13,10)              # permittivity variables
    eps = epsreal + 1j*0.005 
    emrough = np.linspace(0.001, 0.05, 12)
    

    EPS, KS = np.meshgrid(emrough, eps, indexing='ij')
    outputhh = np.zeros((len(emrough), len(eps)), dtype = np.float32)
    
    # VRT 
    vrt = VRT(layers)
    
    
    # # set up VRTmodel to return results as a whole
    # # results is split in the following loop
    for i,ks in enumerate(emrough):
        vrt.l1.upperks = float(ks)
        vrtmodel = VRTmodel(vrt, var_name, eps)
        results  = np.array(vrtmodel.VRT_multiprocess())
        print(results)
        for j in range(len(eps)):
            outputhh[i,j] = results[j][1]
#         print(outputhh[i,:])
    plotBS_contour(EPS, KS, outputhh)
    
        
#     # # Magellan BS
#     inc, meanbs, stdbs = get_MagellanBS('/home/indujaa/pvert/D1.csv')
    
#     # # Plotting backscatter vs paramater (any one)
#     plotBS(var_name, values, svv, shh, inc, meanbs, stdbs)
        
        
if __name__ == '__main__':
    main()   
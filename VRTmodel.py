#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:39:40 2021

@author: indujaa
"""

import numpy as np
from Layers import Layers
from VRT import VRT
# from VRT_Tsang import VRT
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd

# np.set_printoptions(formatter={'complex_kind': '{:.8f}'.format},suppress = True)

class VRTmodel:
    
    def __init__(self, vrt, var, values):
        # # set up vrt model with user-defined set of variable range
        self.vrt = vrt
        self.variable = var
        self.values = values
        self.n = len(values)
        
        # # placeholder for output
        self.sigmavv = np.zeros((5, self.n), dtype=np.float32)
        self.sigmahh = np.zeros_like(self.sigmavv)
        self.cpr = np.zeros_like(self.sigmavv)
        self.e_v = np.zeros(self.n)
        self.e_h = np.zeros(self.n)
        
        self.refh = np.zeros(self.n)
        self.refv = np.zeros(self.n)
        self.transh = np.zeros(self.n)
        self.transv = np.zeros(self.n)
        
        self.rh = np.zeros(self.n)
        self.rv = np.zeros(self.n)
        self.th = np.zeros(self.n)
        self.tv = np.zeros(self.n)
        
        self.svv1 = np.zeros(self.n)
        self.svv2 = np.zeros(self.n)
        self.shh1 = np.zeros(self.n)
        self.shh2 = np.zeros(self.n)
        self.svvt1 = np.zeros(self.n)
        self.shht1 = np.zeros(self.n)
        self.svvt2 = np.zeros(self.n)
        self.shht2 = np.zeros(self.n)
        
        # # multiprocessing threads
        self.nproc = 10
        

    def VRT_multiprocess(self):
        
        pool = mp.Pool(processes=self.nproc)
        
        
        if self.variable == 'Depth': 
            results = pool.map(self.vrt.VRTsolver_depth, self.values)
        elif self.variable  == 'Layer permittivity': 
            results = pool.map(self.vrt.VRTsolver_eps1, self.values)
        elif self.variable  == 'Substrate permittivity': 
            results = pool.map(self.vrt.VRTsolver_eps2, self.values)
        elif self.variable  == 'Permittivity of scatterers':
            results = pool.map(self.vrt.VRTsolver_epsscatterer, self.values)
        elif self.variable  == 'Axis ratio of scatterers': 
            results = pool.map(self.vrt.VRTsolver_scatterershape, self.values)
        elif self.variable  == 'Number concentration of scatterers': 
            results = pool.map(self.vrt.VRTsolver_scatterernumconc, self.values)
        elif self.variable  == 'Maximum scatterer size': 
            results = pool.map(self.vrt.VRTsolver_scatterersize, self.values)
        elif self.variable  == 'Surface roughness': 
            results = pool.map(self.vrt.VRTsolver_emrough, self.values)
        elif self.variable  == 'Subsurface roughness': 
            results = pool.map(self.vrt.VRTsolver_emroughsub, self.values)
        elif self.variable == 'Incidence angle':
            results = pool.map(self.vrt.VRTsolver_incidence, self.values)

        # # reshape results
        
#         print(results)
        
#         for i in range(self.n):
#             self.sigmavv[:,i] = results[i][0]
#             self.sigmahh[:,i] = results[i][1]
#             self.cpr[:,i]= results[i][2]  
#             self.e_v[i] = results[i][3] 
#             self.e_h[i] = results[i][4] 
            
            # # only if returning Fresnel coefficients
#             self.refh[i] = results[i][3]
#             self.refv[i] = results[i][4]
#             self.transh[i] = results[i][5]
#             self.transv[i] = results[i][6]
            
#             self.rh[i] = results[i][7]
#             self.rv[i] = results[i][8]
#             self.th[i] = results[i][9]
#             self.tv[i] = results[i][10]
            
        # only for backscatter plots
#         for i in range(self.n):
#             self.svv1[i] = results[i][0]
#             self.svv2[i] = results[i][1]
#             self.shh1[i] = results[i][2]
#             self.shh2[i] = results[i][3]
#             self.svvt1[i] = results[i][4]
#             self.shht1[i] = results[i][5]
#             self.svvt2[i] = results[i][6]
#             self.shht2[i] = results[i][7]
            
        return results
            
    
    def VRT_singleprocess(self):
        
           
        for i in range(len(self.values)):
            
            # # Converting from numpy types to simple data types for use with the matlab i2em function
            if self.variable  == 'Depth': self.vrt.l1.d = float(self.values[i])
            elif self.variable  == 'Layer permittivity': self.vrt.l1.eps = complex(self.values[i])
            elif self.variable  == 'Substrate permittivity': self.vrt.l2.eps = complex(self.values[i])
            elif self.variable  == 'Permittivity of scatterers': self.vrt.l1.inclusions.eps = complex(self.values[i])
            elif self.variable  == 'Axis ratio of scatterers': self.vrt.l1.inclusions.axratio = float(self.values[i])
            elif self.variable  == 'Number concentration of scatterers': self.vrt.l1.inclusions.nw = float(self.values[i])
            elif self.variable  == 'Maximum scatterer size': self.vrt.l1.inclusions.Dmax = float(self.values[i])
            elif self.variable  == 'Surface roughness': self.vrt.l1.upperks = float(self.values[i])
            elif self.variable  == 'Suburface roughness': self.vrt.l2.upperks = float(self.values[i])
            elif self.variable  == 'Incidence angle':
                self.vrt.theta_i = np.deg2rad(self.values[i])
                self.vrt.l1.theta_i = np.deg2rad(self.values[i])
            
            [self.sigmavv[:,i], self.sigmahh[:,i], self.cpr[:,i]] = self.vrt.VRTsolver()
            
            
    def plotBSC(self, ax, varx, vary, pol, colors, linestyles, labels, xlabel):
        
        if pol == self.vrt.polH: ylabel = '$\sigma_{HH}$'
        elif pol == self.vrt.polV: ylabel = '$\sigma_{VV}$'
        elif pol == 'CPR': ylabel = 'CPR'
        elif pol == 'Emissivity_H': ylabel = 'Emissivity'
        elif pol == 'Emissivity_V': ylabel = 'Emissivity'
        elif pol == None: ylabel = '$\sigma$'

        if vary.ndim > 1:
            for i in range(len(vary)):
                
                if labels[i] in ['Volume', 'Volume-Subsurface']:
                    ax.plot(varx, vary[i], color = colors[i], linestyle = linestyles[i],  marker='o', label = labels[i])
                else:
                    ax.plot(varx, vary[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
        elif vary.ndim == 1:
            ax.plot(varx, vary, color = colors, linestyle = linestyles, label = labels)
#            if labels[i] in ['Volume', 'Volume-Subsurface']:
#                ax.plot(varx, vary[i], color = colors[i], linestyle = linestyles[i],  marker='o', label = labels[i])
#            else:
#                ax.plot(varx, vary[i], color = colors[i], linestyle = linestyles[i], label = labels[i])
#            
        # ax.hlines(I1_mean, min(x), max(x), color = 'lightgray', linestyle = '--', label = "Magellan $\sigma^0$")
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
#         ax.set_ylim(-100, 2)
        ax.legend(loc = "best")
    
    def plotFresnelCoef(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.values, self.refh, label = 'Reflection coeff H')
        ax.plot(self.values, self.refv, label = 'Reflection coeff V')
        ax.plot(self.values, self.transh, label = 'Transmission coeff H')
        ax.plot(self.values, self.transv, label = 'Transmission coeff V')
        ax.plot(self.values, self.rh, label = 'Reflectivity H')
        ax.plot(self.values, self.rv, label = 'Reflectivity V')
        ax.plot(self.values, self.th, label = 'Transmittivity H')
        ax.plot(self.values, self.tv, label = 'Transmittivity V')
        
        ax.set_xlabel(self.variable)
        ax.legend()
        plt.show()
        
        
    def plotOutput(self, sigmaVV = 0., sigmaHH = 0., cpr = 0., e_v = 0., e_h = 0.):
        
        # # set values to calculated backscatter if input not provided
        if sigmaVV == 0.: sigmaVV= self.sigmavv
        if sigmaHH == 0.: sigmaHH= self.sigmahh
        if cpr == 0.: cpr = self.cpr
        if e_v == 0.: e_v = self.e_v
        if e_h == 0.: e_h = self.e_h
        
        xlabel = self.variable
        x = self.values

        print(x)
        
        ### Update font sizes
        params = {'axes.labelsize': 20,
              'axes.titlesize': 30,
              'font.size': 15}
        plt.rcParams.update(params)
        
        fig, ax = plt.subplots(nrows=2, ncols=2)
        
        colors = ["white", "maroon", "olive", "grey", "teal"]
        linestyles = ['-', '-.', '-.', ':', ':']
        labels = ['Total', 'Surface', 'Subsurface', 'Volume', 'Volume-Subsurface']
        
        self.plotBSC(ax[0,0], x, sigmaVV, self.vrt.polV, colors, linestyles, labels, xlabel)
        self.plotBSC(ax[0,1], x, sigmaHH, self.vrt.polH, colors, linestyles, labels, xlabel)
        self.plotBSC(ax[1,0], x, e_h, 'Emissivity_H', "blue", '-', "e_h", xlabel)
        self.plotBSC(ax[1,0], x, e_v, 'Emissivity_V', "black", '-', "e_v", xlabel)
        self.plotBSC(ax[1,1], x, cpr, 'CPR', colors, linestyles, labels, xlabel)
        
        
        figure = plt.gcf()
        figure.set_size_inches(16, 10)
        plt.savefig("vrtnew.png")
        
        plt.show()
        
    def plotSurfaceBSC(self):
        xlabel = self.variable
        x = self.values

#         sur_bsc = 10*np.log10(np.array([self.svv1, self.svv2, self.shh1, self.shh2, self.svvt1, self.shht1, self.svvt2, self.shht2]))
#         sur_bsc = np.array([self.svv1, self.svv2, self.shh1, self.shh2, self.svvt1, self.shht1])
        sur_bsc = 10*np.log10(np.array([self.svv1, self.svv2, self.shh1, self.shh2]))
        ### Update font sizes
        ### Update font sizes
        params = {'axes.labelsize': 20,
              'axes.titlesize': 30,
              'font.size': 15}
        plt.rcParams.update(params)
        
#         colors = ["#1b9e77","#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
        colors = ["red","red", "blue", "blue", "green", "green", "pink", "pink"]
        linestyles = ['-.', ':', '-.', ':', '-.', ':', '-.', ':']
        labels = ['VV_Ulaby', 'VV', 'HH_Ulaby', 'HH']
#         labels = ['VV_Ulaby', 'VV', 'HH_Ulaby', 'HH', 'VV_trans', 'HH_trans']
#         labels = ['VV_nc', 'VV_c', 'HH_nc', 'HH_c', 'VV_t_nc', 'VV_t_c', 'HH_t_nc', 'HH_t_c']
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.plotBSC(ax, x, sur_bsc, None, colors, linestyles, labels, xlabel)
        
        figure = plt.gcf()
        figure.set_size_inches(16, 12)
        plt.savefig("IEM_BSC.png")
        
        plt.show()
        
        
    def writeCSV(self, outfile):
        
        xlabel = self.variable
        x = self.values
        data = np.concatenate((x.reshape(len(x),1), np.transpose(self.sigmavv), np.transpose(self.sigmahh), np.transpose(self.cpr)), axis=1)

        
        df = pd.DataFrame(data, columns=[xlabel, "Total_VV", "Surface_VV", "Subsurface_VV", "Volume_VV", "Volume_sub_VV",
                                       "Total_HH", "Surface_HH", "Subsurface_HH", "Volume_HH", "Volume_sub_HH", 
                                       "Total_CPR", "Surface_CPR", "Subsurface_CPR", "Volume_CPR", "Volume_sub_CPR"])
                                     
        df.to_csv(outfile, sep = ',')
        
        
        
def plotCSV(infile):
    
    # # read csv as pandas df
    data_df = pd.read_csv(infile, sep=',', header=0)
    xlabel =  data_df.columns[0]
    
    # # read csv as array
    data_arr = np.loadtxt(infile, dtype=np.float32, delimiter=',', skiprows = 1)
    
    x = np.transpose(data_arr[:,0])
    sigmavv = np.transpose(data_arr[:,1:5])
    sigmahh = np.transpose(data_arr[:,6:10])
    cpr = np.transpose(data_arr[:,11:15])
    
    
    self.plotOutput(x, xlabel, sigmavv, sigmahh, cpr)
    

  
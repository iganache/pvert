#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:39:40 2021

@author: indujaa
"""

import numpy as np
from Layers import Layers
from VRT_module import VRT
# from VRT_Tsang import VRT
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd


class VRTmodel:
    
    def __init__(self, nproc = 1):
        # # multiprocessing threads
        self.nproc = nproc

    def VRT_multiprocess(self, input_dict, scattertype):
        
        pool = mp.Pool(processes=self.nproc)
        
        # # results is returned as a pd series
        vrt = VRT()
        results = pool.starmap(vrt.VRTsolver, list(product(input_dict, scattertype)))
        
        results_df = self.make_df(results, input_dict)
            
        return results_df
    
    def make_df(self, results, input_dict):
        
        output_cols = ["shh_sur", "svv_sur", "cpr_sur", "shh_sub", "svv_sub", "cpr_sub", "shh_vol", "svv_vol", "cpr_vol", "shh_volsub", "svv_volsub", "cpr_volsub", "ev_sur", "eh_sur", "ev_sub", "eh_sub"]
        input_cols = list(input_dict[0].keys())
        cols = input_cols + output_cols

        df = pd.DataFrame(columns=cols).astype(np.float32)
#         df = df.astype({"eps1": complex, "eps2": complex, "epsinc": complex})
   
        for row in results:
            df_length = len(df)
            for j in range(len(cols)):
                df.loc[df_length,cols[j]] = row[j]

    
        return df
            
    def plotsurBS_contour(self, df, MagBSC=[-10, -14]):

        eps = df["eps1r"]
        ks = df["ks1"]
        shh = df["shh_sur"]
        
        bsc_df = df[["eps1r", "ks1", "shh_sur"]]
        
        Z = bsc_df.pivot_table(index="ks1", columns="eps1r", values="shh_sur").T.values

        X_unique = np.sort(bsc_df.ks1.unique())
        Y_unique = np.sort(bsc_df.eps1r.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        BSplot = ax.contourf(X, Y, Z, levels = 20, cmap = "Greys_r")
        
        # # shading off area corresponding to magellan Fresnel reflectivity
        # # only for Irnini Mons using Triana's measurements
        rho = np.array([0.088, 0.154])

        eps = np.square((1+rho**0.5) / (1-rho**0.5))
        ax.axhspan(eps[0], eps[1], alpha=0.5, color='white')
        
        # # plotting contours corresponding to the BSC measured from magellan  
        # # for 45.5 deg < thetai < 50 deg
        BSplot2 = plt.contour(BSplot, levels=MagBSC,
                  colors='k', linestyles='--')
        cbar = fig.colorbar(BSplot)
        cbar.ax.set_ylabel("Surface backscatter (dB)")
        
        ax.set_xlabel("EM surface roughness (m)")
        ax.set_ylabel("Real dielectric permittivity")

        plt.show()
        
    def plotvolBS_contour(self, df, MagBSC=[-10, -14]):

        eps = df["epsincr"]
        abyc = df["a"]
        shh = df["shh_vol"]

        bsc_df = df[["epsincr", "a", "shh_vol", "svv_vol"]]
#         pd.set_option('display.max_rows', bsc_df.shape[0]+1)
#         print(bsc_df)

        Z = bsc_df.pivot_table(index="a", columns="epsincr", values="shh_vol").T.values

        X_unique = np.sort(bsc_df.a.unique())
        Y_unique = np.sort(bsc_df.epsincr.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        BSplot = ax.contourf(X, Y, Z, cmap = plt.cm.bone)
        
        # # plotting contours corresponding to the BSC measured from magellan  
        # # for 45.5 deg < thetai < 50 deg
        BSplot2 = plt.contour(BSplot, levels=MagBSC,
                  colors='k')
        cbar = fig.colorbar(BSplot)
        cbar.ax.set_ylabel("Volume backscatter (dB)")

        plt.show()
        
    def plotsubBS_contour(self, df, MagBSC=[-10, -14]):

        eps = df["eps2r"]
        ks = df["ks2"]
        shh = df["shh_sub"]

        bsc_df = df[["eps2r", "ks2", "shh_sub"]]
   
        Z = bsc_df.pivot_table(index="ks2", columns="eps2r", values="shh_sub").T.values

        X_unique = np.sort(bsc_df.ks2.unique())
        Y_unique = np.sort(bsc_df.eps2r.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        BSplot = ax.contourf(X, Y, Z, cmap = plt.cm.bone)
        
        # # plotting contours corresponding to the BSC measured from magellan  
        # # for 45.5 deg < thetai < 50 deg
        BSplot2 = plt.contour(BSplot, levels=MagBSC,
                  colors='k', linestyles='--')
        cbar = fig.colorbar(BSplot)
        cbar.ax.set_ylabel("Substrate backscatter (dB)")

        plt.show()
        
    def plotsuremis_contour(self, df, Magemis=[0.806, 0.876]):

        eps = df["eps1r"]
        ks = df["ks1"]
        shh = df["eh_sur"]
        
        bsc_df = df[["eps1r", "ks1", "eh_sur"]]
        
        Z = bsc_df.pivot_table(index="ks1", columns="eps1r", values="eh_sur").T.values

        X_unique = np.sort(bsc_df.ks1.unique())
        Y_unique = np.sort(bsc_df.eps1r.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        BSplot = ax.contourf(X, Y, Z, cmap = "OrRd")
        
        # # shading off area corresponding to magellan Fresnel reflectivity
        # # only for Irnini Mons using Triana's measurements
        rho = np.array([0.088, 0.154])

        eps = np.square((1+rho**0.5) / (1-rho**0.5))
        ax.axhspan(eps[0], eps[1], alpha=0.5, color='white')
        
        # # plotting contours corresponding to the BSC measured from magellan  
        # # for 45.5 deg < thetai < 50 deg
        BSplot2 = plt.contour(BSplot, levels=Magemis,
                  colors='k', linestyles='--')
        cbar = fig.colorbar(BSplot)
        cbar.ax.set_ylabel("Surface emissivity (dB)")
        
        ax.set_xlabel("EM surface roughness (m)")
        ax.set_ylabel("Real dielectric permittivity")

        plt.show()
        
    def plotsubemis_contour(self, df, Magemis=[0.806, 0.876]):

        eps = df["eps2r"]
        ks = df["ks2"]
        shh = df["eh_sub"]

        bsc_df = df[["eps2r", "ks2", "eh_sub"]]
   
        Z = bsc_df.pivot_table(index="ks2", columns="eps2r", values="eh_sub").T.values

        X_unique = np.sort(bsc_df.ks2.unique())
        Y_unique = np.sort(bsc_df.eps2r.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        BSplot = ax.contourf(X, Y, Z, cmap = "OrRd")
        
        # # plotting contours corresponding to the BSC measured from magellan  
        # # for 45.5 deg < thetai < 50 deg
        BSplot2 = plt.contour(BSplot, levels=Magemis,
                  colors='k', linestyles='--')
        cbar = fig.colorbar(BSplot)
        cbar.ax.set_ylabel("Substrate backscatter (dB)")

        plt.show()
        
    def lineplot(self, df, xcol, ycol, groupcol, xlabel=None, ylabel=None, legend = "", data = None, outfile="ex.png"):
        
        setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        colors = ['#374043', '#4C8D8B', '#D1E2BC','#BFBB68', '#D87439', '#984ea3','#999999', '#e41a1c', '#dede00']
        i=0
        
        for key, grp in df.groupby([groupcol]):
            ax = grp.plot(ax=ax, kind='line', x=xcol, y=ycol, c=colors[i], label=legend+" = "+str(key))
            i+=1
            
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
            
        if data !=None:
            if len(data) == 2:
                ax.axhspan(data[0], data[1], alpha=0.5, color='gray')
            else:
                ax.axhline(data, color='gray')
            
        figure = plt.gcf()
        figure.set_size_inches(16, 12)
        plt.savefig(outfile)   
            
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
    

def setPlotStyle():
        ###### Set matplolib font sizs ###############
        # plt.style.use('dark_background')
        font = {'family' : 'sans-serif',
            'sans-serif':'Arial',
            'size'   : 35}
        plt.rc('font', **font)
        plt.rc('axes', titlesize=40)     # fontsize of the axes title
        plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=35)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=35)    # fontsize of the tick labels
        plt.rc('legend', fontsize=40)    # legend fontsize
        plt.rc('legend', title_fontsize=40)    # legend fontsize
        plt.rc('figure', titlesize=30)  # fontsize of the figure title
    

  
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, cycle
import pandas as pd

class plotting:
    
    def __init__(self):
        
        self.colors = [ "#1D91C0", "#67001F", "#CB181D", "#78C679", "#F46D43", "#A6CEE3", "#FD8D3C", "#A6D854", "#D4B9DA", "#6A51A3", "#7F0000", "#D9D9D9", "#FFF7BC", "#000000", "#F0F0F0", "#C7EAE5", "#003C30", "#F16913", "#FFF7FB", "#8C6BB1", "#C7E9B4", "#762A83", "#FC9272", "#AE017E", "#F7F7F7", "#DF65B0", "#EF3B2C", "#74C476"]
#         self.colors = ["#920000","#924900","#db6d00","#24ff24","#ffff6d", "#000000","#004949","#009292","#ff6db6","#ffb6db", "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff"]

        self.linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), (0, (5, 10))]
    
    
    def lineplot(self, df, xcol, ycol, groupcol, xlabel=None, ylabel=None, legend = "", data = None, ylim = None, outfile=None):
        
        pd.set_option('display.max_rows', df.shape[0]+1)
        print(df)
        
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        i=0
        
        for key, grp in df.groupby([groupcol]):
            ax = grp.plot(ax=ax, kind='line', x=xcol, y=ycol, c=self.colors[i], label=legend+" = "+str(round(key,1)))
            i+=1
            
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
            
        if ylim != None:
            ax.set_ylim(ylim[0], ylim[1])
            
        if data !=None:
            if len(data) == 2:
                ax.axhspan(data[0], data[1], alpha=0.5, color='gray')
            else:
                ax.axhline(data, color='gray')
           
        fig.set_size_inches(16, 12)
        
        if outfile != None:
            plt.savefig(outfile)   
            
        plt.show()          
        
        
    def incplot(self, df, xcol, ycol, groupbycols, xlabel=None, ylabel=None, legend = "", data = None, outfile=None):
        
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        n = df[groupbycols[0]].nunique()
        try:
            m = df[groupbycols[1]].nunique()  
        except:
            m = 0
    
#         for n in range(len(groupbycols)):
        i = j = 0
        for key, grp in df.groupby(groupbycols):
            print("before", i,j)
            leg = []
            for k in key:
                leg.append(round(k,1))
            ax = grp.plot(ax=ax, kind='line', x=xcol, y=ycol, c=self.colors[i], linestyle = self.linestyles[j], label=legend+" = "+str(leg))
            if (j+1)%m == 0:
                i+=1
            j+=1
     
            if i >= n: i=0
            if j >= m: j=0
            print("after", i,j)
            print(list(key))
            
                
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
            
        fig.set_size_inches(16, 12)
        if outfile != None:
            plt.savefig(outfile)   
        plt.show()
        
    
    def plotCSV(self, infile, xcol, ycol, groupcol, xlabel=None, ylabel=None, legend = "", data = None, ylim = None, outfile=None):
        
         # # read csv as pandas df
        df = pd.read_csv(infile, sep=',', header=0)

         # # call lineplot from here
        self.lineplot(df, infile, xcol, ycol, groupcol, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, ylim = ylim, outfile=outfile)
        
    
    def plotCSVinc(self, infile, xcol, ycol, groupbycols, xlabel=None, ylabel=None, legend = "", data = None, ylim = None, outfile=None):
        # # read csv as pandas df
        df = pd.read_csv(infile, sep=',', header=0)
        
        self.incplot(df, xcol, ycol, groupbycols, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, outfile=outfile)
        
    def contourplot(self, df, xcol, ycol, zcol, xlabel=None, ylabel=None, cbarlegend = "", data = None, outfile=None):
        
        bsc_df = df[[xcol, ycol, zcol]]
        
        Z = bsc_df.pivot_table(index=xcol, columns=ycol, values=zcol).T.values

        X_unique = np.sort(bsc_df[xcol].unique())
        Y_unique = np.sort(bsc_df[ycol].unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        BSplot = ax.contourf(X, Y, Z, levels = 20, cmap = "Greys_r")
        
        # # plotting contours corresponding to the BSC measured from magellan  
        # # for 45.5 deg < thetai < 50 deg
        BSplot2 = plt.contour(BSplot, levels=data,
                  colors='blue', linestyles='--')
        cbar = fig.colorbar(BSplot)
        cbar.ax.set_ylabel(cbarlegend)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.set_size_inches(16, 12)
        
        if outfile != None:
            plt.savefig(outfile)   
        plt.show()
        
        
    def setPlotStyle(self):
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
    
    
    
    
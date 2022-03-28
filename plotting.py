import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, StrMethodFormatter
from itertools import product, cycle
import pandas as pd

class plotting:
    
    def __init__(self):
        
#         self.colors = ["#252525",  "#636363", "#969696",  "#cccccc", "#f7f7f7"]
        self.colors = ["#56767D", "#C5785C", "#C1C1A2", "#BD8585", "#9270A9", "#000000"]
#         self.colors = ["#036179", "#CA7313", "#710742", "#029E77", "#37035F", "#000000"]
#         self.colors = [ "#1D91C0", "#67001F", "#78C679", "#CB181D", "#F46D43", "#A6CEE3", "#FD8D3C", "#A6D854", "#D4B9DA", "#6A51A3", "#7F0000", "#D9D9D9", "#FFF7BC", "#000000", "#F0F0F0", "#C7EAE5", "#003C30", "#F16913", "#FFF7FB", "#8C6BB1", "#C7E9B4", "#762A83", "#FC9272", "#AE017E", "#F7F7F7", "#DF65B0", "#EF3B2C", "#74C476"]
#         self.colors = ["#920000","#924900","#db6d00","#24ff24","#ffff6d", "#000000","#004949","#009292","#ff6db6","#ffb6db", "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff"]

        # # brown colors
#         self.colors = ["#A67B5B", "#4B3621", "#ddd1b1"]

        self.linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), (0, (5, 10))]
    
    
    def lineplot(self, df, xcol, ycol, groupbycols=None, xlabel=None, ylabel=None, legend = "", data = None, ylim = None, outfile=None):
        
        pd.set_option('display.max_rows', df.shape[0]+1)
        print(df)
        
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        n = df[groupbycols[0]].nunique()
        try:
            m = df[groupbycols[1]].nunique()  
        except:
            m = 0

        try:
            n_ycols = len(ycol)
        except:
            n_ycols = 1
        
        for num in range(n_ycols):
            i = j = 0
            for key, grp in df.groupby(groupbycols):
                if type(key) != tuple:
                    key = (key,)             
                leg = []
                for k in key:
                    leg.append(round(k,2))
                ax = grp.plot(ax=ax, kind='line', x=xcol, y=ycol[num], c=self.colors[num], label=legend+" = "+str(leg), linewidth = 2.5, )
                if m!=0:
                    if (j+1)%m == 0:
                        i+=1
                elif m==0:
                    i+=1
                j+=1

                if i >= n: i=0
                if j >= m: j=0
            
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
        
        
    def incplot(self, df, xcol, ycol, groupbycols, xlabel=None, ylabel=None, legend = "", data = None, xlim = None, ylim = None, outfile=None):
        
#         df = df[(df["eps2i"] == 100)]
        
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        n = df[groupbycols[0]].nunique()
        try:
            m = df[groupbycols[1]].nunique()  
        except:
            m = 0
    
        
        try:
            n_ycols = len(ycol)
        except:
            n_ycols = 1
#         for n in range(len(groupbycols)):
        for num in range(n_ycols):
            i = j = 0
            for key, grp in df.groupby(groupbycols):

                if type(key) != tuple:
                    key = (key,)             
                leg = []
                for k in key:
                    leg.append(round(k,2))
#                 ax = grp.plot(ax=ax, kind='line', x=xcol, y=ycol[num], c=self.colors[i], linestyle = self.linestyles[j], linewidth = 3, label=legend+" = "+str(leg))
                ax = grp.plot(ax=ax, kind='line', x=xcol, y=ycol[num], c=self.colors[i], linestyle = self.linestyles[j], linewidth = 3, legend=False)
                
                if m!=0:
                    if (j+1)%m == 0:
                        i+=1
                elif m==0:
                    i+=1
                j+=1

                if i >= n: i=0
                if j >= m: j=0
            
                
#         ax.legend(loc = 3)
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
        # # setting degree symbol
        ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
#         ax.yaxis.set_major_formatter(EngFormatter(unit=u"dB"))
           
        if xlim != None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim != None:
            ax.set_ylim(ylim[0], ylim[1])
            
        if data != None:
#             thi, mean, std = self.Magfromcsv(data)
            df_grp = self.Magfromcsv(data)
#             ecolors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#882255', '#44AA99', '#999933', '#AA4499']
#             ecolors = ["#252525",  "#636363", "#969696",  "#cccccc", "#f7f7f7", "#f7f7f7"]
            inc_ecolors = "#2e2249"
            inc_fcolors = "#43385b"
            ecolors = ["#767676", "#636363", "#4f4f4f", "#3b3b3b", "#282828", "#141414"]
            emarks = ['o','^', 'v', 'd', 's',  'h', '*'] 
        
            m = 0
            for name, grp in df_grp:
                if name in ["Irnini", "Anala", "Didilia", "Pavlova"]:
                    ax.errorbar(grp['thetai'], grp['mean'], grp['rms'], label = name, fmt = emarks[m], alpha=0.9, mec = inc_ecolors, mfc=inc_fcolors, markersize = 18, fillstyle=None, ecolor = inc_fcolors, elinewidth = 2.5, capsize = 0)
                else:
                    ax.errorbar(grp['thetai'], grp['mean'], grp['rms'], label = name, fmt = emarks[m], alpha=0.7, mec = 'k', mfc=ecolors[m], markersize = 18, ecolor = "darkgray", elinewidth = 2.5, capsize = 0)
                m+=1
                
                
#             ax.errorbar(thi, mean, yerr=std, fmt='sk', markerfacecolor='none', markersize = 6, ecolor = 'k', elinewidth = 2, capsize = 3)
            
    
        # # # axis styling # # # 
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['top'].set_visible(False)
        
        axcolor = "#453f3d"
        ax.spines['right'].set_color(axcolor)
        ax.spines['left'].set_color(axcolor)
        ax.spines['bottom'].set_color(axcolor)
        ax.spines['top'].set_color(axcolor)
        
        ax.xaxis.label.set_color(axcolor)        #setting up X-axis label color to yellow
        ax.yaxis.label.set_color(axcolor)          #setting up Y-axis label color to blue

        ax.tick_params(axis='x', colors=axcolor)  #setting up X-axis tick color to red
        ax.tick_params(axis='y', colors=axcolor)  #setting up Y-axis tick color to black
        #################################
        
        fig.set_size_inches(16, 12)
        if outfile != None:
            plt.savefig(outfile)   
        plt.show()
        
    def multiscatterplot(self, df, xcol, ycol, xlabel=None, ylabel=None, legend = "", data = None, xlim = None, ylim = None, outfile=None):
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)

    
        line_dict = {"shh_total": "solid", "shh_sur": (0, (5, 10)), "shh_sub": "dashed", "shh_vol": "dotted", "shh_volsub":"dashdot"}
        color_dict = {"shh_total": "#000000", "shh_sur": "#696969", "shh_sub": "#808080", "shh_vol": "#A9A9A9", "shh_volsub":"#D3D3D3"}
        label_dict = {"shh_total": "total", "shh_sur": "surface", "shh_sub": "subsurface", "shh_vol": "volume", "shh_volsub":"vol-sub"}
        
        try:
            n_ycols = len(ycol)
        except:
            n_ycols = 1

#         ax = df.plot(x=xcol, y=ycol, kind = 'line', linewidth = 3)
        for num in range(n_ycols):
            df.plot(x=xcol, y=ycol[num], kind='line', c=color_dict[ycol[num]], linestyle = line_dict[ycol[num]], linewidth = 3, ax=ax, label=label_dict[ycol[num]])

        ax.legend(loc = 3)
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
        # # setting degree symbol
        ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
#         ax.yaxis.set_major_formatter(EngFormatter(unit=u"dB"))
           
        if xlim != None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim != None:
            ax.set_ylim(ylim[0], ylim[1])
            
        if data != None:
#             thi, mean, std = self.Magfromcsv(data)
            df_grp = self.Magfromcsv(data)
#             ecolors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#882255', '#44AA99', '#999933', '#AA4499']
#             ecolors = ["#252525",  "#636363", "#969696",  "#cccccc", "#f7f7f7", "#f7f7f7"]
            inc_ecolors = "#2e2249"
            inc_fcolors = "#43385b"
            ecolors = ["#767676", "#636363", "#4f4f4f", "#3b3b3b", "#282828", "#141414"]
            emarks = ['o','^', 'v', 'd', 's',  'h', '*'] 
        
            m = 0
            for name, grp in df_grp:
                if name in ["Irnini", "Anala", "Didilia", "Pavlova"]:
                    ax.errorbar(grp['thetai'], grp['mean'], grp['rms'], label = name, fmt = emarks[m], alpha=0.9, mec = inc_ecolors, mfc=inc_fcolors, markersize = 18, fillstyle=None, ecolor = inc_fcolors, elinewidth = 2.5, capsize = 0)
                else:
                    ax.errorbar(grp['thetai'], grp['mean'], grp['rms'], label = name, fmt = emarks[m], alpha=0.7, mec = 'k', mfc=ecolors[m], markersize = 18, ecolor = "darkgray", elinewidth = 2.5, capsize = 0)
                m+=1
                
                
#             ax.errorbar(thi, mean, yerr=std, fmt='sk', markerfacecolor='none', markersize = 6, ecolor = 'k', elinewidth = 2, capsize = 3)
            
        fig.set_size_inches(16, 12)
        if outfile != None:
            plt.savefig(outfile)   
        plt.show()
        
    def multicprplot(self, df, xcol, ycol, xlabel=None, ylabel=None, legend = "", data = None, xlim = None, ylim = None, outfile=None):
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
    
        line_dict = {"cpr_total": "solid", "cpr_sur": (0, (5, 10)), "cpr_sub": "dashed", "cpr_vol": "dotted", "cpr_volsub":"dashdot"}
        color_dict = {"cpr_total": "#000000", "cpr_sur": "#696969", "cpr_sub": "#808080", "cpr_vol": "#A9A9A9", "cpr_volsub":"#D3D3D3"}
        label_dict = {"cpr_total": "total", "cpr_sur": "surface", "cpr_sub": "subsurface", "cpr_vol": "volume", "cpr_volsub":"vol-sub"}
        
        try:
            n_ycols = len(ycol)
        except:
            n_ycols = 1

#         ax = df.plot(x=xcol, y=ycol, kind = 'line', linewidth = 3)
        for num in range(n_ycols):
            df.plot(x=xcol, y=ycol[num], kind='line', c=color_dict[ycol[num]], linestyle = line_dict[ycol[num]], linewidth = 3, ax=ax)

        ax.legend(loc = 3)
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
        # # setting degree symbol
        ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
#         ax.yaxis.set_major_formatter(EngFormatter(unit=u"dB"))
           
        if xlim != None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim != None:
            ax.set_ylim(ylim[0], ylim[1])
            
        if data != None:
#             thi, mean, std = self.Magfromcsv(data)
            df_grp = self.Magfromcsv(data)
            ecolors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#882255', '#44AA99', '#999933', '#AA4499']
            emarks = ['o', 'v', 's', 'd', '^', 'h', '*'] 
            m = 0
            for name, grp in df_grp:
                ax.errorbar(grp['thetai'], grp['mean'], grp['rms'], label = name, fmt = emarks[m], alpha=0.7, mec = 'k', mfc=ecolors[m], markersize = 8, ecolor = "darkgray", elinewidth = 2.5, capsize = 0)
                m+=1
                
                
#             ax.errorbar(thi, mean, yerr=std, fmt='sk', markerfacecolor='none', markersize = 6, ecolor = 'k', elinewidth = 2, capsize = 3)
            
        fig.set_size_inches(16, 12)
        if outfile != None:
            plt.savefig(outfile)   
        plt.show()
        
    def multidlpplot(self, df, xcol, ycol, xlabel=None, ylabel=None, legend = "", data = None, xlim = None, ylim = None, outfile=None):
        self.setPlotStyle()
        fig, ax = plt.subplots(nrows=1, ncols=1)
    
        line_dict = {"dlp_total": "solid", "dlp_sur": (0, (5, 10)), "dlp_sub": "dashed", "dlp_vol": "dotted", "dlp_volsub":"dashdot"}
        color_dict = {"dlp_total": "#000000", "dlp_sur": "#696969", "dlp_sub": "#808080", "dlp_vol": "#A9A9A9", "dlp_volsub":"#D3D3D3"}
        label_dict = {"dlp_total": "total", "dlp_sur": "surface", "dlp_sub": "subsurface", "dlp_vol": "volume", "dlp_volsub":"vol-sub"}
        
        try:
            n_ycols = len(ycol)
        except:
            n_ycols = 1

#         ax = df.plot(x=xcol, y=ycol, kind = 'line', linewidth = 3)
        for num in range(n_ycols):
            df.plot(x=xcol, y=ycol[num], kind='line', c=color_dict[ycol[num]], linestyle = line_dict[ycol[num]], linewidth = 3, ax=ax)

        ax.legend(loc = 3)
        if xlabel != None: ax.set_xlabel(xlabel)
        if ylabel != None: ax.set_ylabel(ylabel)
        # # setting degree symbol
        ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
#         ax.yaxis.set_major_formatter(EngFormatter(unit=u"dB"))
           
        if xlim != None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim != None:
            ax.set_ylim(ylim[0], ylim[1])
            
        if data != None:
#             thi, mean, std = self.Magfromcsv(data)
            df_grp = self.Magfromcsv(data)
            ecolors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#882255', '#44AA99', '#999933', '#AA4499']
            emarks = ['o', 'v', 's', 'd', '^', 'h', '*'] 
            m = 0
            for name, grp in df_grp:
                ax.errorbar(grp['thetai'], grp['mean'], grp['rms'], label = name, fmt = emarks[m], alpha=0.7, mec = 'k', mfc=ecolors[m], markersize = 8, ecolor = "darkgray", elinewidth = 2.5, capsize = 0)
                m+=1
                
                
#             ax.errorbar(thi, mean, yerr=std, fmt='sk', markerfacecolor='none', markersize = 6, ecolor = 'k', elinewidth = 2, capsize = 3)
            
        fig.set_size_inches(16, 12)
        if outfile != None:
            plt.savefig(outfile)   
        plt.show()
        
    def plotCSV(self, infile, xcol, ycol, groupcol, xlabel=None, ylabel=None, legend = "", data = None, ylim = None, outfile=None):
        
         # # read csv as pandas df
        df = pd.read_csv(infile, sep=',', header=0)

         # # call lineplot from here
        self.lineplot(df, infile, xcol, ycol, groupcol, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, ylim = ylim, outfile=outfile)
        
    
    def plotCSVinc(self, infile, xcol, ycol, groupbycols, xlabel=None, ylabel=None, legend = "", data = None, xlim = None, ylim = None, outfile=None):
        # # read csv as pandas df
        df = pd.read_csv(infile, sep=',', header=0)
#         new_df = df[df["eps2i"] == .01]
#         new_df = df[(df["eps1r"] == 10) & (df["s1"] == 0.04)]
        
        self.incplot(df, xcol, ycol, groupbycols, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, xlim = xlim, ylim = ylim, outfile=outfile)
                                                       
    def plotCSVmulti(self, infile, plottype, xcol, ycol, groupbycols = None, xlabel=None, ylabel=None, legend = "", data = None, xlim = None, ylim = None, outfile=None):
        # # read csv as pandas df
        df = pd.read_csv(infile, sep=',', header=0)
        
        # # make conditional columns
        new_df = df[(df["d"] == 0.126) & (df["s2"] == 0.04)]
#         new_df = df[df["d"] == .126]
#         new_df = new_df[new_df["s2"] == .026]
        print(new_df["s2"])
        
        if plottype == "BSC":
            self.multiscatterplot(new_df, xcol, ycol, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, xlim = xlim, ylim = ylim, outfile=outfile)
#         if plottype == "CPR":
#             self.multicprplot(new_df, xcol, ycol, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, xlim = xlim, ylim = ylim, outfile=outfile)
#         if plottype == "DLP":
#             self.multidlpplot(new_df, xcol, ycol, xlabel=xlabel, ylabel=ylabel, legend = legend, data = data, xlim = xlim, ylim = ylim, outfile=outfile)
        
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
        
    def penetrtationdeth(self, wavelength = 0.126, epsr = 3, epsi = np.linspace(0, 0.01, 50)):
        # # depth of penetration = (1/ke) = (1/2alpha)
        
#         ke_Ulaby = 2 * (2*np.pi/wavelength) * np.sqrt(epsi)
        dp_Ulaby = np.sqrt(epsr) / (epsi * 2 * np.pi/wavelength)          # # page 124 = when eps" << eps'
        alpha_Bruce = (2*np.pi/wavelength) * np.sqrt(0.5* epsr * (np.sqrt(1 + (epsi/epsr)**2) - 1))
        dp_Bruce = 1 / (2*alpha_Bruce)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(epsi, dp_Ulaby, label="Ulaby", color = 'k')
        ax.plot(epsi, dp_Bruce, label= "Bruce", color = 'orange', linestyle="-.")
        ax.set_xlabel("Imaginary part of permittivity")
        ax.set_ylabel("Depth of penetration in meters")
        plt.text(.002, 0.5, "dielectric_real = "+str(epsr))
        plt.legend()

        plt.show()
    
    def Magfromcsv(self, csvfile):
         # # read csv as pandas df
        df = pd.read_csv(csvfile, sep=',', header=0)
        df_grp = df.groupby("site") 
        return df_grp
        
#         df = pd.read_csv(csvfile, sep=',', header=None)
#         data_arr = df.to_numpy(dtype=np.float32)
#         theta_i = data_arr[:,0]
#         mean = data_arr[:,1]
#         rms = data_arr[:,2]
#         return theta_i, mean, rms

        
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
        plt.rc('legend', fontsize=35)    # legend fontsize
        plt.rc('legend', title_fontsize=35)    # legend fontsize
        plt.rc('figure', titlesize=35)  # fontsize of the figure title

    
    
    
    
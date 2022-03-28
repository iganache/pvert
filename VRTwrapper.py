#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:39:40 2021

@author: indujaa
"""

import numpy as np
from VRT_module import VRT
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product, cycle
import pandas as pd



class VRTmodel:
    """
    A wrapper class for running the VRT / I2EM model on multiple threads.

        Attributes
        ----------
        nproc : int
            number of cores available 
        
        Methods
        -------
        VRT_multiprocess(input_dict, scattertype):
            Takes in several user input dictionaries and 
            passes them on to the main VRTmodel class.
            
        make_df(results, input_dict):
            Creates an output pandas datafram comprising
            all mode input and output values.
    """
   
    def __init__(self, nproc = 1):
        # # multiprocessing threads
        self.nproc = nproc

    def VRT_multiprocess(self, input_dict, scattertype):
        
        """
        Primary function from which the main VRT model is run in mutiple threads

            Parameters:
                input_dict (dict): A list of dictionaries containing model input as key-value pairs
                scatter_type (list): A list of different scattering mechanisms to be modeled. Accepable
                                    values are: "surface", :subsrface", "volume", "volume-subsurface"

            Returns:
                results_df (pandas dataframe): A dataframe whose rows correspond to outputs from each 
                model run thread.
        """
        
        pool = mp.Pool(processes=self.nproc)
        
        # # results is returned as a pd series
        vrt = VRT()
        results = pool.starmap(vrt.VRTsolver, list(product(input_dict, scattertype)))
        
        results_df = self.make_df(results, input_dict)
            
        return results_df
    
    def make_df(self, results, input_dict):
        
        """
        Creates a pandas dataframe from a list of dicionaries

            Parameters:
                results (dict): A list of dictionaries containing model output as key-value pairs
                input_dict (dict): A list of dictionaries containing model input as key-value pairs

            Returns:
                df (pandas dataframe): A dataframe whose rows correspond to outputs from each 
                    model run thread.
        """
        
        output_cols = ["ks1", "ks2", "shh_sur", "svv_sur", "cpr_sur", "dlp_sur", "shh_sub", "svv_sub", "cpr_sub", "dlp_sub", "shh_vol", "svv_vol", "cpr_vol", "dlp_vol", "shh_volsub", "svv_volsub", "cpr_volsub", "dlp_volsub", "shh_total", "svv_total", "cpr_total", "dlp_total", "ev_sur", "eh_sur", "ev_sub", "eh_sub", "ev_vol", "eh_vol", "ev_total", "eh_total", "ssa_v", "ssa_h"]
        input_cols = list(input_dict[0].keys())
        cols = input_cols + output_cols

        df = pd.DataFrame(columns=cols).astype(np.float32)
#         df = df.astype({"eps1": complex, "eps2": complex, "epsinc": complex})
   
        for row in results:
            df_length = len(df)
            for j in range(len(cols)):
                df.loc[df_length,cols[j]] = row[j]

    
        return df
  
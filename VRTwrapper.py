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
  
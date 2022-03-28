import numpy as np
import matplotlib.pyplot as plt
from itertools import product, cycle
import pandas as pd
from plotting import plotting

myplt = plotting()

# # Case 1


# myplt.plotCSVinc("ResultPlots/Case1/surfaceBSC_Jan28.csv", 'thetai', ['shh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", xlim = [5, 80],  ylim = [-30, 5], legend= '', outfile = "ResultPlots/Case1/BSCvsInc_sur_Mar3.png")
# myplt.plotCSVinc("ResultPlots/Case1/surfaceBSC_Jan28.csv", 'thetai', ['eh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", xlim = [5, 80],  ylim = [0.3, 1], legend= '', outfile = "ResultPlots/Case1/EmisvsInc_sur_Mar3.png")


# # Case 2 -- (old case 3)

# myplt.plotCSVinc("ResultPlots/Case3/sce3-regeps2r-higheps2i.csv", 'thetai', ['shh_total'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", xlim = [5, 80],  ylim = [-30, 5], legend= '', outfile = "ResultPlots/Case3/BSCvsInc-new-regeps2r-higheps2i_sub_Mar3png")
# myplt.plotCSVinc("ResultPlots/Case3/sce3-regeps2r-higheps2i.csv", 'thetai', ['eh_sub'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", xlim = [5, 80],  ylim = [0.3, 1], legend= '', outfile = "ResultPlots/Case3/EmisvsInc-new-regeps2r-higheps2i_sub_Mar3.png")

# myplt.plotCSVinc("ResultPlots/Case3/midrough/sce3-ferro.csv", 'thetai', ['shh_total'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", xlim = [5, 80],  ylim = [-30, 5], legend= '', outfile = "ResultPlots/Case3/midrough/BSCvsInc-ferro_Mar3.png")
# myplt.plotCSVinc("ResultPlots/Case3/midrough/sce3-ferro.csv", 'thetai', ['eh_sub'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", xlim = [5, 80],  ylim = [0.3, 1],legend= '', outfile = "ResultPlots/Case3/midrough/EmisvsInc-ferro_Mar3.png")

# # correlation length 
# myplt.plotCSVinc("ResultPlots/Case1/corrlen/surfaceBSC_corrlen.csv", 'thetai', ['shh_sur'], ['ks1', 'cl1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/Case1/corrlen/BSCvsInc-corrlen_Jan28.png")
# myplt.plotCSVinc("ResultPlots/Case1/corrlen/surfaceBSC_corrlen.csv", 'thetai', ['eh_sur'], ['ks1', 'cl1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", legend= '', outfile = "ResultPlots/Case1/corrlen/EmisvsInc-corrlen_Jan28.png")


# #  Talks

# myplt.plotCSVinc("ResultPlots/Case1/sce1-eps1-27-roughks2.csv", 'thetai', ['shh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", xlim = [1, 80], ylim = [-40, 10], legend= '', outfile = "empty_BSC")
# myplt.plotCSVinc("ResultPlots/Case1/sce1-eps1-27-roughks2.csv", 'thetai', ['eh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", xlim = [1, 80], ylim = [0.2, 1], legend= '', outfile = "empty_emis")


# # mutliple contribution plots
myplt.plotCSVmulti("ResultPlots/Case3/midrough/sce3-equal-new.csv", "BSC", 'thetai', ['shh_total', 'shh_sur', 'shh_sub'], groupbycols = ['d', 'ks2'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', legend = "", data = "Mag_sigma.csv", xlim = [1, 80], ylim = [-60, 10], outfile="ResultPlots/Case3/midrough/sce3-equal-ferro.png")

# myplt.penetrtationdeth(wavelength = 0.126, epsr = 2, epsi = np.array([0.005, .05]))
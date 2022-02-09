import numpy as np
import matplotlib.pyplot as plt
from itertools import product, cycle
import pandas as pd
from plotting import plotting

myplt = plotting()

# # Case 1

# myplt.plotCSVinc("ResultPlots/Case1/sce1-eps1-27-roughks2.csv", 'thetai', ['shh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/Case1/BSCvsInc_sur_Jan17.png")
# myplt.plotCSVinc("ResultPlots/Case1/sce1-eps1-27-roughks2.csv", 'thetai', ['eh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", ylim = [0.2, 1], legend= '', outfile = "ResultPlots/Case1/EmisvsInc_sur_Jan17.png")

# myplt.plotCSVinc("ResultPlots/Case1/surfaceBSC_Jan28.csv", 'thetai', ['shh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/eps-2_BSC")
# myplt.plotCSVinc("ResultPlots/Case1/surfaceBSC_Jan28.csv", 'thetai', ['eh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", ylim = [0.2, 1], legend= '', outfile = "ResultPlots/eps-2_emis")


# # Case 2 -- (old case 3)

# myplt.plotCSVinc("ResultPlots/Case3/sce3-regeps2r-higheps2i.csv", 'thetai', ['shh_total'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/Case3/BSCvsInc-new-regeps2r-higheps2i_sub_Jan28.png")
# myplt.plotCSVinc("ResultPlots/Case3/sce3-regeps2r-higheps2i.csv", 'thetai', ['eh_sub'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", legend= '', outfile = "ResultPlots/Case3/EmisvsInc-new-regeps2r-higheps2i_sub_Jan28.png")

# myplt.plotCSVinc("ResultPlots/Case3/midrough/sce3-semic.csv", 'thetai', ['shh_total'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/Case3/midrough/BSCvsInc-semic_Jan28.png")
# myplt.plotCSVinc("ResultPlots/Case3/midrough/sce3-semic.csv", 'thetai', ['eh_sub'], ['d','ks2'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", ylim = [0.2, 1], legend= '', outfile = "ResultPlots/Case3/midrough/EmisvsInc-semic_Jan28.png")

# # correlation length 
# myplt.plotCSVinc("ResultPlots/Case1/corrlen/surfaceBSC_corrlen.csv", 'thetai', ['shh_sur'], ['ks1', 'cl1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", ylim = [-40, 10], legend= '', outfile = "ResultPlots/Case1/corrlen/BSCvsInc-corrlen_Jan28.png")
# myplt.plotCSVinc("ResultPlots/Case1/corrlen/surfaceBSC_corrlen.csv", 'thetai', ['eh_sur'], ['ks1', 'cl1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", legend= '', outfile = "ResultPlots/Case1/corrlen/EmisvsInc-corrlen_Jan28.png")


# #  Talks

# myplt.plotCSVinc("ResultPlots/Case1/sce1-eps1-27-roughks2.csv", 'thetai', ['shh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$\sigma_{HH}$ (dB)', data = "Mag_sigma.csv", xlim = [1, 80], ylim = [-40, 10], legend= '', outfile = "empty_BSC")
# myplt.plotCSVinc("ResultPlots/Case1/sce1-eps1-27-roughks2.csv", 'thetai', ['eh_sur'], ['eps1r', 'ks1'], xlabel="Incidence Angle", ylabel='$e_{H}$',  data = "Mag_emis.csv", xlim = [1, 80], ylim = [0.2, 1], legend= '', outfile = "empty_emis")



# myplt.penetrtationdeth(wavelength = 0.126, epsr = 2, epsi = np.array([0.005, .05]))
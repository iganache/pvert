import numpy as np
import matplotlib.pyplot as plt


# plt.style.use('dark_background')
font = {'family' : 'sans-serif',
        'sans-serif':'Arial',
        'size'   : 35}
plt.rc('font', **font)
plt.rc('axes', titlesize=35)     # fontsize of the axes title
plt.rc('axes', labelsize=35)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('legend', title_fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title

epsr = 6
epsi = np.linspace(0.01,1,50)
wavelength = [0.1706, 0.1259, 0.0936]
names = ["Pioneer" , "Magellan", "EnVision"]
lines = [(0, (5, 10)), "dashed", "solid", "dotted"]
colors = ["#696969", "#808080", "#A9A9A9"]
# dp_Ulaby = np.sqrt(epsr) / (epsi * 2 * np.pi/wavelength)          # # page 124 = when eps" << eps'
# alpha_Bruce = (2*np.pi/wavelength) * np.sqrt(0.5* epsr * (np.sqrt(1 + (epsi/epsr)**2) - 1))
# dp_Bruce = 1 / (2*alpha_Bruce)

fig, ax = plt.subplots(nrows=1, ncols=1)

for i in range(len(wavelength)):
    dp_Ulaby = np.sqrt(epsr) / (epsi * 2 * np.pi/wavelength[i])
    ax.plot(epsi, dp_Ulaby, color = colors[i], linestyle = lines[i], label =names[i], linewidth=2.5)
#     ax.plot(epsi, dp_Bruce, label= "Bruce", color = 'orange', linestyle="-.")
    
ax.set_xlabel("Imaginary part of permittivity")
ax.set_ylabel("Depth of penetration in meters")
ax.set_yscale('log')
ax.legend()
# plt.text(.002, 0.5, "dielectric_real = "+str(epsr))
plt.legend()

fig.set_size_inches(16, 12)
plt.savefig("depth_pen.png")   
plt.show()

plt.show()
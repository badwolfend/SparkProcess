from __future__ import print_function
import Utils as ut
import numpy as np
from numpy.polynomial import legendre as L
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import PolyFitLiAbel as pfa
# from scipy.interpolate import RBFInterpolator, InterpolatedUnivariateSpline
import os

## First load Air Index#
date = "8_23_2024"
shot = "00126_interferometer"
# shot = "00249_interferometer"

f_name_air_abi = "./Data/Quantaray/"+date+"/"+str(shot)+"_direct_air_n_4.npz"
f_name_plasma_abi = "./Data/Quantaray/"+date+"/"+str(shot)+"_direct_Ne.npz"
# f_name_plasma_abi = "./Data/Quantaray/"+date+"/"+str(shot)+"_direct_air_Ne.npz"

wl=532E-9
ncrit = ut.find_critical_dens(wl * 10 ** 6)
G = 0.0002274 # m^3/kg

k0 = 2*np.pi/wl
multN2 = 1 / k0 # multiplying phase by this gives abel transform in index of refraction (can multiply at the end)
multp = 1/(wl*ncrit/np.pi)
Nair = 2.45E25
nair0 = 1.000293

f=1
l0 = 0.79375  # mm
l0 = l0 / 1000
dx = f * l0 / (1390 - 713)
dxf = f * l0 /(3500-3442)

N2npzfile = np.load(f_name_air_abi)
N2gyA = N2npzfile['arr_0']
nair = multN2 * N2gyA

Plnpzfile = np.load(f_name_plasma_abi)
PlgyA = Plnpzfile['arr_0']
Pln = np.sqrt(1-PlgyA/ncrit)
# PlPlasma = Pln-nair+(nair0-1)
PlPlasma = Pln-nair
NePlasma = ncrit*(1-PlPlasma**2)
ut.plot_one_thing(-nair, 'gyA-Air', to_show=False, vmax=(-0.0015, 0.0015), cmap="twilight_shifted") # delta index
# ut.plot_one_thing(NePlasma, 'Ne-Plasma', vmax=(1e20, 1e25), cmap="twilight") # delta index
ut.plot_one_thing(NePlasma, 'Ne-Plasma', cmap="twilight") # delta index

nz, nx = NePlasma.shape
NePlasmaOnly = np.zeros_like(NePlasma)
NAirOnly = np.zeros_like(NePlasmaOnly)
for z in range(nz):
    zind = ut.find_val_contour(NePlasma[z, :], ymin=100, delt=1, val=1e22)
    NePlasmaOnly[z, :zind]=NePlasma[z, :zind]
    # NeAirOnly[z, zind:]=-(1*multN2 * N2gyA[z, zind:]+nair) / G / 4.65e-23
    NAirOnly[z, zind:]=(multN2 * N2gyA[z, zind:] + nair0)
    NAirOnly[z, zind:]= ((-1 * multN2 * N2gyA[z, zind:] + nair0 - 1)) * Nair / (nair0 - 1)

# ut.plot_one_thing(-1*multN2 * N2gyA, 'nair', vmax=(1-0.005, 1+0.005), cmap="twilight") # delta index

fig =plt.figure()
fig.set_size_inches(4, 6.0)
ax3 = fig.add_subplot(1, 1, 1)
im3 = ax3.imshow(NePlasmaOnly/ncrit, cmap="twilight", vmin=0, vmax=0.003)
plt.tight_layout()
plt.savefig("./Plots/"+date+"_NeNcPlasmaOnly_direct.png", dpi=600)
plt.show()

fig =plt.figure()
fig.set_size_inches(4, 6.0)
ax3 = fig.add_subplot(1, 1, 1)
# im3 = ax3.imshow(NAirOnly/Nair, cmap="bone", vmin=1e20, vmax=1.5e26)
# im3 = ax3.imshow(NAirOnly/Nair, cmap="bone", vmin=0, vmax=5)
im3 = ax3.imshow(NAirOnly/Nair, cmap="bone")

# plt.colorbar(im3)
plt.tight_layout()
plt.savefig("./Plots/"+date+"_NairNair0Only_direct.png", dpi=600)
plt.show()

out = NAirOnly/Nair
print(out[305,243])
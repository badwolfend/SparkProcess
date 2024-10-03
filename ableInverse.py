from __future__ import print_function
import Utils as ut
import numpy as np
from numpy.polynomial import legendre as L
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import PolyFitLiAbel as pfa
import os

#############################################################
## Setup some initial values for the inversion ##############
#############################################################
date = "8_23_2024"
shot = "00118_interferometer"
# shot = "00163_interferometer"
# shot = "00208_interferometer"
shot = "00249_interferometer"
f_name = "./Data/Quantaray/"+date+"/"+str(shot)+"_phasediff.npz"
f_name_abi = "./Data/Quantaray/"+date+"/"+str(shot)+"_direct_n_4.npz"

# Laser Parameters 
wl=532E-9
ncrit = ut.find_critical_dens(wl * 10 ** 6)

# Abel Inversion Parameters
n = 4

## Set order of Abel approximation and polynomial expansion of the phase ##
npzfile = np.load(f_name)
pdiff = npzfile['arr_0'] # Let air be negative phase

ut.plot_one_thing(pdiff, "pdiff")
# pdiffHalf = (wl*ncrit/np.pi)*pdiff[500:, 400:]
pdiffHalf = -pdiff[200-200:, 400+77:]

mean = np.mean(pdiffHalf[:,400:])
print("mean: " +str(mean))

pdiffHalf=pdiffHalf-mean

ut.plot_one_thing(pdiffHalf, "pdiff")
nz, ny = np.shape(pdiffHalf)
np.savez("./Data/Quantaray/"+date+"/"+str(shot)+"_phasediff_half", pdiffHalf)

dx = 0.0085 # mm/pixel
dx = dx/1000 # m/pixel
if not os.path.isfile(f_name_abi):
    gyA = np.zeros((nz, ny))

    for z in range(nz):
        percent = 100*z/nz

        print(f'{percent:.2f}'+" % Done")
        pdiffline = pdiffHalf[z, :]
        pdiffline = ut.find_zero_pad(pdiffline, ymin=100, air=True, to_show=False, whole=True)
        ys = np.linspace(0,1.0, len(pdiffline))
        ny = len(ys)
        dy = ys[1] - ys[0]
        us = pfa.y_2_u(ys)
        alpha = (ny-1)*dx

        Cnn = pfa.Cnn(0.5, n)
        detCnn = np.linalg.det(Cnn)
        for yi in range(ny-10):
            M = pfa.construct_abel_m_notpoly(ys[yi:], pdiffline[yi:], n, dy=dy, delt=0.001, integrator="direct")
            gyA[z, yi] = np.linalg.det(M) / pfa.check_u(pfa.y_2_u(ys[yi]))
        gyA[z, :] = gyA[z, :] * (1 / (detCnn))/alpha

        if True:
            if z %800==0:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(ys, pdiffline, 'xkcd:red orange', linewidth=4, label='I-Fit')
                ax.legend()
                plt.show()
                ut.plot_one_thing(gyA, 'gyA', vmax=(1e19, 1e25))
                # Ne = ut.get_Ne(gyA, f=1, wl=532E-9, dx=dx, to_plot=True)

    np.savez(f_name_abi, gyA)
else:
    npzfile = np.load(f_name_abi)
    gyA = npzfile['arr_0']

# ut.plot_one_thing(gyA, 'gyA', vmax=(0, 5E4))
ut.plot_one_thing(gyA, 'gyA')

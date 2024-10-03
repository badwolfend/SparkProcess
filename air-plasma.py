import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import cumtrapz
import Utils as ut
import PolyFitLiAbel as pfa


def compute_mass(n_density, r, z, particle_mass):
    # Compute dr and dz (spacing in r and z)
    dr = np.gradient(r)  # shape (num_r,)
    dz = np.gradient(z)  # shape (num_z,)
    
    # Mesh the radial and axial grids
    Z, R = np.meshgrid(z, r, indexing='ij')  # Z and R will both be of shape (num_z, num_r)
    
    # Compute the volume element in cylindrical coordinates: dV = 2πr * dr * dz
    dV = 2 * np.pi * R * dr[np.newaxis, :] * dz[:, np.newaxis]  # Ensure correct broadcasting
    
    # Compute the total mass: M = sum(n(r,z) * m * dV)
    total_mass = np.sum(n_density * particle_mass * dV)
    
    return total_mass

#############################################################
## Setup some initial values for the inversion ##############
#############################################################
date = "8_23_2024"

# shot = "00118_interferometer"
# shot = "00163_interferometer"
# shot = "00208_interferometer"
shot = "00249_interferometer"
f_name = "./Data/Quantaray/"+date+"/"+str(shot)+"_phasediff.npz"
f_name_abi = "./Data/Quantaray/"+date+"/"+str(shot)+"_direct_Ne.npz"
f_name_abi = "./Data/Quantaray/"+date+"/"+str(shot)+"_direct_n_4.npz"

# Laser Parameters
wl=532E-9
ncrit = ut.find_critical_dens(wl * 10 ** 6)
k0 = 2*np.pi/wl
multN2 = 1 / k0 # multiplying phase by this gives abel transform in index of refraction (can multiply at the end)
multp = 1/(wl*ncrit/np.pi)
Nair = 2.45E25
nair0 = 1.000293
dx = 0.0085 # mm/pixel
dx = dx/1000

# Mass Scales 
# Define the masses of each particle
m_N = 2.33e-26  # kg (mass of a nitrogen ion)
m_air = 4.81e-26  # kg (average mass of an air molecule)

# Length Scales 

# Load the file 
able_i_file = np.load(f_name_abi)
able_i = able_i_file['arr_0']
nN2 = able_i
ut.plot_one_thing(nN2, 'nN2', cmap="RdBu") # delta index

# Limit the data to the plasma region with positive index of refraction
nz, nx = nN2.shape
nN2Only = np.zeros_like(nN2)
nAirOnly = np.zeros_like(nN2Only)
Pln  = np.zeros_like(nN2Only)
nAirShockOnly = np.zeros_like(nN2Only)  

# Find the Positive Index of Refraction

# Step 1: Create a boolean mask for positive values
positive_mask = nN2 > 0

# Step 2: Apply an operation (e.g., square the positive values) using the mask
nN2Only[positive_mask] = nN2[positive_mask]/multp

# Find negative index of refraction
# nAirOnly[~positive_mask] = (-1*multN2*nN2[~positive_mask] + nair0)
nAirOnly[~positive_mask] = ((-1 * multN2 * nN2[~positive_mask] + nair0 - 1)) * Nair / (nair0 - 1)

mask_shock = nAirOnly > 1.2*Nair
# mask_shock = nAirOnly > 0

nAirShockOnly[mask_shock] = nAirOnly[mask_shock]

# Compute the mass of the plasma

# Define the radial and axial grids
r = np.arange(nx) * dx
z = np.arange(nz) * dx

# Compute the mass of the plasma
vtravel = 12 # km/s 118
vtravel = 9 # km/s 118
vtravel = 6 # km/s 249
vtravel = vtravel * 1000 # m/s

mass = compute_mass(nN2Only, r, z, m_N)
E_kinetic_p = 0.5 * mass * vtravel**2
print(f"Kinetic Energy of the plasma: {E_kinetic_p} J")
print(f"Mass of the plasma: {mass} kg")

# Compute the mass of the air
mass_air = compute_mass(nAirShockOnly, r, z, m_air)
E_kinetic_air = 0.5 * mass_air * vtravel**2
print(f"Kinetic Energy of the air: {E_kinetic_air} J")
print(f"Mass of the air: {mass_air} kg")


# Plot the results
ut.plot_one_thing(nN2Only, 'nN2Only', cmap="twilight_shifted") # delta index
# ut.plot_one_thing(nAirOnly, 'nAir', cmap="reds") # delta index

# Plot the noisy signal, smoothed signal, and detected edges
fig1 = plt.figure(figsize=(13.385, 13.385))
ax0 = fig1.add_subplot(1, 1, 1)

# Plot original noisy signal
im = ax0.imshow(nN2Only, cmap="twilight_shifted", vmin=1e22, vmax=5e24, extent=[0, nx*dx, 0, nz*dx])
plt.colorbar(im)
plt.tight_layout()
plt.savefig("./Plots/Quantaray/"+date+"/"+str(shot)+"_nN2Only.png")

# Plot the noisy signal, smoothed signal, and detected edges
fig2 = plt.figure(figsize=(13.385, 13.385))
ax1 = fig2.add_subplot(1, 1, 1)

# Plot original noisy signal
im = ax1.imshow(nAirShockOnly, cmap="terrain", vmin=1e23, vmax=4e25, extent=[0, nx*dx, 0, nz*dx])
plt.colorbar(im)
plt.tight_layout()
plt.savefig("./Plots/Quantaray/"+date+"/"+str(shot)+"_nAirOnly.png")
plt.show()
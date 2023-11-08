import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math 
from matplotlib.colors import Normalize
from matplotlib import cm


# Extracting data ------------------------------------------------------------
# ============================================================================


mat_phi = scipy.io.loadmat('phi_grid_matrix.mat')  # data is a dictionnary
mat_theta = scipy.io.loadmat('theta_grid_matrix.mat')
mat_TE = scipy.io.loadmat('S12_grid_matrix_TE.mat')
mat_TM = scipy.io.loadmat('S12_grid_matrix_TM.mat')

data_phi = mat_phi.get("phi_grid_matrix")
data_theta = mat_theta.get("Theta_grid_matrix")
data_TE = mat_TE.get("S12_grid_matrix_TE")
data_TM = mat_TM.get("S12_grid_matrix_TM")


# Polarisation ---------------------------------------------------------------
# ============================================================================

E_square_norm = data_TE**2 + data_TM **2
#np.where(P == np.amax(P)))  -> gives 16 and 17
y_ = -data_TE[16][17]
x_ = data_TM[16][17]

E_g = x_ - 1j * y_
E_d = x_ + 1j * y_

if (E_g > E_d) :
    bla = "--> polarisation LHCP"
else :
    bla = "--> polarisation RCHP"
print("E_g =", E_g)
print("E_d =", E_d)
print(bla)


# Directivity ----------------------------------------------------------------
# ============================================================================

integrale_F = ...
D = 4 * np.pi * E_square_norm / integrale_F


# Plot in 2D -----------------------------------------------------------------
# ============================================================================

plt.figure()
plt.imshow(np.abs(data_TE)) #,extent=(0, 2*np.pi, 2*np.pi/2, 0)
plt.title("S12 TE")
plt.figure()
plt.imshow(np.abs(data_TM))
plt.title("S12 TM")

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#color_map = cm.RdYlBu_r
#calarMap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color_map)
#C_colored = scalarMap.to_rgba(np.abs(data_TE))
surf = ax.plot_surface(data_phi, data_theta, np.abs(data_TE),rstride=1, cstride=1)
plt.title("S12 TE")
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(data_phi, data_theta, np.abs(data_TM),rstride=1, cstride=1)
plt.title("12 TM")



#xx=data_TE*np.sin(data_theta)*np.cos(data_phi)
#yy=data_TE*np.sin(data_theta)*np.sin(data_phi)
#zz=data_TE*np.cos(data_theta)

# RHCP !!!

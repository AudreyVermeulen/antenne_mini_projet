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

#E_square_norm = data_TE**2 + data_TM **2
E_square_norm = abs(data_TE)**2 + abs(data_TM)**2
#np.where(P == np.amax(P)))  -> gives 16 and 17
y_ = -data_TE[16][17]   # -> y_ = ê_phi
x_ = data_TM[16][17]    # -> x_ = ê_theta

E_g = x_ - 1j * y_
E_d = x_ + 1j * y_

if (abs(E_g) > abs(E_d)) :  # on choisit la polarisation dominante
    bla = "--> polarisation LHCP"
else :
    bla = "--> polarisation RCHP"
print("E_g =", E_g)
print("E_d =", E_d)
print(bla)
print()

# Directivity ----------------------------------------------------------------
# ============================================================================

# Il faut |F|², mais |E|² et |F|² sont = à une constante près

Jac = np.sin(data_theta)   # = r² sin(theta), mais r = 1
integrale_E = 0
d_phi = 5 * np.pi / 180    # d_phi * d_theta = dS , elem surfacique infinitésimal
d_theta = 4 * np.pi / 180  # step de 4° et 5° donnés dans énoncé

# Approximation integrale avec somme de Riemann
for i in range(len(data_phi)):
    for j in range(len(data_phi[0])):
        integrale_E += E_square_norm[i][j] * Jac[i][j] * d_phi * d_theta
        
print(integrale_E)        
D = 4 * np.pi * E_square_norm / integrale_E


# exemple rieman pour intégrer surface sphère :
def Riemann_surf_sph():
    s = 0
    phi_s = np.arange(0,2*np.pi, 0.001)
    theta_s =  np.arange(0, np.pi, 0.001)
    
    for phi in phi_s:
        for theta in theta_s:
            s += np.sin(theta) * 0.001 * 0.001
    print(s)     
#Riemann_surf_sph()
        


# Plot in 2D -----------------------------------------------------------------
# ============================================================================

#plt.figure()
#plt.imshow(np.abs(data_TE)) #,extent=(0, 2*np.pi, 2*np.pi/2, 0)
#plt.title("S12 TE")
#plt.figure()
#plt.imshow(np.abs(data_TM))
#plt.title("S12 TM")

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#color_map = cm.RdYlBu_r
#calarMap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color_map)
#C_colored = scalarMap.to_rgba(np.abs(data_TE))
surf = ax.plot_surface(data_phi, data_theta, D,rstride=1, cstride=1)
#plt.title("S12 TE")
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#surf = ax.plot_surface(data_phi, data_theta, np.abs(data_TM),rstride=1, cstride=1)
#plt.title("12 TM")

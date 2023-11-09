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

""" Quelle calcul de norme choisir ???
    (1) donne encore des nbrs complexes donc marche pas
    (2) et (3) donne RHCP mais pas le mm graphe de directivité ...
"""
#E_square_norm = data_TE**2 + data_TM **2
E_square_norm = abs(data_TE)**2 + abs(data_TM)**2
#E_square_norm = abs(data_TE**2 + data_TM**2)


idx = np.where(E_square_norm == np.amax(E_square_norm))
idx_l = idx[0][0]
idx_c = idx[1][0] 
print("posisition max :",idx_l,";",idx_c)
print("=> phi =",data_phi[idx_l][idx_c],"; theta =", data_theta[idx_l][idx_c],"[rad]")
print()

x_ = data_TM[idx_l][idx_c]    # -> x_ = ê_theta
y_ = -data_TE[idx_l][idx_c]   # -> y_ = ê_phi

E_g_max = (x_ - 1j * y_) / np.sqrt(2)  
E_d_max = (x_ + 1j * y_) / np.sqrt(2)

if (abs(E_g_max) > abs(E_d_max)) :  # on choisit la polarisation dominante
    bla = "--> polarisation LHCP"
else :
    bla = "--> polarisation RCHP"
    
print("E_g_max =", E_g_max)
print("E_d_max =", E_d_max)
print(bla)
print()


# Directivities --------------------------------------------------------------
# ============================================================================


# Il faut |F|², mais |E|² et |F|² sont = à une constante près.
# On calcule les directivités selon les polarisation circulaires gauches et 
# droites, càd le rapport entre l'énergie du champ LHCP/RHCP sur l'énergie totale.

X = data_TM
Y = -data_TE
E_g = (X - 1j * Y) / np.sqrt(2) 
E_d = (X + 1j * Y) / np.sqrt(2)
E_g_square_norm = abs(E_g)**2
E_d_square_norm = abs(E_d)**2

Jac = np.sin(data_theta)   # = r² sin(theta), mais r = 1
integrale_E = 0            # représente l'énergie totale
d_phi = 5 * np.pi / 180    # d_phi * d_theta = dS , elem surfacique infinitésimal
d_theta = 4 * np.pi / 180  # step de 4° et 5° donnés dans énoncé

# Approximation integrale avec somme de Riemann
for i in range(len(data_phi)):
    for j in range(len(data_phi[0])):
        integrale_E += E_square_norm[i][j] * Jac[i][j] * d_phi * d_theta
        
print("valeur intégrale (énergie totale):",integrale_E)   
     
D_tot  = 4 * np.pi * E_square_norm / integrale_E
D_LHCP = 4 * np.pi * E_g_square_norm / integrale_E
D_RHCP = 4 * np.pi * E_d_square_norm / integrale_E



# exemple riemann pour intégrer surface sphère :
def Riemann_surf_sph():
    s = 0
    phi_s = np.arange(0,2*np.pi, 0.001)
    theta_s =  np.arange(0, np.pi, 0.001)
    
    for phi in phi_s:
        for theta in theta_s:
            s += np.sin(theta) * 0.001 * 0.001
    print(s)     
#Riemann_surf_sph()
print()     


# Plot D in 3D ---------------------------------------------------------------
# ============================================================================

# Selon theta et phi:

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(data_phi, data_theta, D_tot,rstride=1, cstride=1)
plt.title("Directivity versus theta - phi")

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(data_phi, data_theta, D_LHCP,rstride=1, cstride=1)
plt.title("Directivity LHCP versus theta - phi")

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(data_phi, data_theta, D_RHCP,rstride=1, cstride=1)
plt.title("Directivity RHCP versus theta - phi")

# u-v plane (cosine directions)

#u = np.cos(data_phi)
#v = np.cos(data_theta)

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#surf = ax.plot_surface(u, v, D,rstride=1, cstride=1)
#plt.title("Directivity versus u - v ?")


xx=D_tot*np.sin(data_theta)*np.cos(data_phi)
yy=D_tot*np.sin(data_theta)*np.sin(data_phi)
zz=D_tot*np.cos(data_theta)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(xx, yy, zz,rstride=1, cstride=1)
plt.title("Directivity versus xyz")   # C CA QUIL FAUT

xx=D_LHCP*np.sin(data_theta)*np.cos(data_phi)
yy=D_LHCP*np.sin(data_theta)*np.sin(data_phi)
zz=D_LHCP*np.cos(data_theta)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(xx, yy, zz,rstride=1, cstride=1)
plt.title("Directivity LHCP versus xyz")   # C CA QUIL FAUT

xx=D_RHCP*np.sin(data_theta)*np.cos(data_phi)
yy=D_RHCP*np.sin(data_theta)*np.sin(data_phi)
zz=D_RHCP*np.cos(data_theta)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
color_map = cm.RdYlBu_r
scalarMap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color_map)
C_colored = scalarMap.to_rgba(D_RHCP)
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=C_colored)

#surf = ax.plot_surface(xx, yy, zz,rstride=1, cstride=1)
plt.title("Directivity RCHP versus xyz")   # C CA QUIL FAUT


"""
(1) Jsp si le u-v plane est juste
(2) Jsp comment décomposer la directivité en LHCP et RHCP comme c'est pas
    un vecteur
"""

# Link budget ----------------------------------------------------------------
# ============================================================================

f = 24*10**9
c = 3*10**8
lmd = c/f
R = 100
rad_eff = 0.8
Pt = 10**(-3)
Gr_dB = 10

#print(data_phi[15][18] == np.pi/2)
#print(data_theta[15][18] == np.pi/6)

factor = (lmd / (4 * np.pi * R) )**2
Gr = 10*np.log(Gr_dB)
D_u = D_LHCP[15][18]        # D dans la direction d'observation theta=30° et phi=90°
Gt = rad_eff * Pt * D_u

Pav_r = factor * Gr * Gt
Pr = rad_eff * Pav_r
print("LHCP")
print("Puissance disponible au récepteur [mW]:", Pav_r * 10**3)
print("Puissance reçue [mW]:", Pr * 10**3)
print()



factor = (lmd / (4 * np.pi * R) )**2
Gr = 10*np.log(Gr_dB)
D_u = D_RHCP[15][18]        # D dans la direction d'observation theta=30° et phi=90°
Gt = rad_eff * Pt * D_u

Pav_r = factor * Gr * Gt
Pr = rad_eff * Pav_r

print("RHCP")
print("Puissance disponible au récepteur [mW]:", Pav_r * 10**3)
print("Puissance reçue [mW]:", Pr * 10**3)











#plt.figure()
#plt.imshow(np.abs(data_TE)) #,extent=(0, 2*np.pi, 2*np.pi/2, 0)
#plt.title("S12 TE")
#plt.figure()
#plt.imshow(np.abs(data_TM))
#plt.title("S12 TM")

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#color_map = cm.RdYlBu_r
#calarMap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color_map)
#C_colored = scalarMap.to_rgba(np.abs(data_TE))
#surf = ax.plot_surface(data_phi, data_theta, np.abs(data_TE),rstride=1, cstride=1)
#plt.title("S12 TE")
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#surf = ax.plot_surface(data_phi, data_theta, np.abs(data_TM),rstride=1, cstride=1)
#plt.title("S12 TM")

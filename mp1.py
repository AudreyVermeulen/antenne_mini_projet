import scipy.io
import numpy as np
import matplotlib.pyplot as plt

"""
LELEC2910 : Antennes : Mini-Projet 1

- Question 1: Trouver la polarisation de l'antenne autour de son maximum.
- Question 2: Représenter les composantes co-pol et cross-pol de la directivité
  dans le plan u-v.
- Question 3: Link budget : calculer la puissance reçue par l'antenne Yagi.
"""

# Extraction data ------------------------------------------------------------
# ============================================================================


mat_phi = scipy.io.loadmat('phi_grid_matrix.mat')  # data est un dictionnaire
mat_theta = scipy.io.loadmat('theta_grid_matrix.mat')
mat_TE = scipy.io.loadmat('S12_grid_matrix_TE.mat')
mat_TM = scipy.io.loadmat('S12_grid_matrix_TM.mat')

data_phi = mat_phi.get("phi_grid_matrix")
data_theta = mat_theta.get("Theta_grid_matrix")
data_TE = mat_TE.get("S12_grid_matrix_TE")
data_TM = mat_TM.get("S12_grid_matrix_TM")


# Polarisation ---------------------------------------------------------------
# ============================================================================


E_square_norm = abs(data_TE)**2 + abs(data_TM)**2  # norme du champ total

idx = np.where(E_square_norm == np.amax(E_square_norm))
idx_l = idx[0][0]
idx_c = idx[1][0] 

x_ = data_TM[idx_l][idx_c]    # -> x_ = ê_theta
y_ = -data_TE[idx_l][idx_c]   # -> y_ = ê_phi

E_g_max = (x_ - 1j * y_) / np.sqrt(2)    # Décomposition en composantes
E_d_max = (x_ + 1j * y_) / np.sqrt(2)    # circulaires gauches et droites.


# Directivités ---------------------------------------------------------------
# ============================================================================


# Il faut |F|², mais |E|² et |F|² sont = à une constante près.
# On calcule les directivités selon les polarisation circulaires gauches et 
# droites, càd le rapport entre l'énergie du champ LHCP/RHCP sur l'énergie totale.

# Décomposition des champs selon LHCP et RHCP:
    
X = data_TM
Y = -data_TE
E_g = (X - 1j * Y) / np.sqrt(2) 
E_d = (X + 1j * Y) / np.sqrt(2)
E_g_square_norm = abs(E_g)**2
E_d_square_norm = abs(E_d)**2

# Approximation integrale avec somme de Riemann:
    
Jac = np.sin(data_theta)   # = r² sin(theta), mais r = 1
integrale_E = 0            # représente l'énergie totale
d_phi = 5 * np.pi / 180    # d_phi * d_theta = dS , elem surfacique infinitésimal
d_theta = 4 * np.pi / 180  # step de 4° et 5° donnés dans énoncé

for i in range(len(data_phi)):
    for j in range(len(data_phi[0])):
        integrale_E += E_square_norm[i][j] * Jac[i][j] * d_phi * d_theta
        
# Calcul des directivités:
    
D_tot  = 4 * np.pi * E_square_norm   / integrale_E
D_LHCP = 4 * np.pi * E_g_square_norm / integrale_E
D_RHCP = 4 * np.pi * E_d_square_norm / integrale_E


if (abs(E_g_max) > abs(E_d_max)) :  # on choisit la polarisation dominante
    D_dominant = D_LHCP
else :
    D_dominant = D_RHCP


# Plot D en 3D ---------------------------------------------------------------
# ============================================================================


# Directivité totale :
    
u = D_tot * np.sin(data_theta) * np.cos(data_phi)
v = D_tot * np.sin(data_theta) * np.sin(data_phi)
w = D_tot * np.cos(data_theta)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(u, v, w,rstride=1, cstride=1)
plt.title("Directivity versus u-v plane")  
#plt.savefig("graphes/D_tot.svg") 

# Directivité LHCP :
    
u = D_LHCP * np.sin(data_theta) * np.cos(data_phi)
v = D_LHCP * np.sin(data_theta) * np.sin(data_phi)
w = D_LHCP * np.cos(data_theta)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(u, v, w,rstride=1, cstride=1)
plt.title("Co-pol directivity versus u-v plane")   
#plt.savefig("graphes/D_LHCP.svg")

# Directivité RCHP :
    
u = D_RHCP * np.sin(data_theta) * np.cos(data_phi)
v = D_RHCP * np.sin(data_theta) * np.sin(data_phi)
w = D_RHCP * np.cos(data_theta)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(u, v, w,rstride=1, cstride=1)
plt.title("Cross-pol directivity versus u-v plane")   
#plt.savefig("graphes/D_RHCP.svg")


# Link budget ----------------------------------------------------------------
# ============================================================================


# Données:

f = 24*10**9
c = 3*10**8
R = 100
rad_eff = 0.8
Pt = 10**(-3)
Gr_dB = 10

# Calculs :
lmd = c/f
factor = (lmd / (4 * np.pi * R) )**2
Gr = 10*np.log(Gr_dB)
D_u = D_dominant[15][18]      # D dominant dans la direction d'observation 
Gt = rad_eff * Pt * D_u                            # theta=30° et phi=90°.

Pr = factor * Gr * Gt
Pr_loss = rad_eff * Pr


# Affichage dans la console --------------------------------------------------
# ============================================================================


print("posisition max E :",idx_l,";",idx_c)
print("=> phi =",data_phi[idx_l][idx_c],"; theta =", data_theta[idx_l][idx_c],"[rad]")
print()

if (abs(E_g_max) > abs(E_d_max)) :  # on choisit la polarisation dominante
    bla = "--> polarisation LHCP au maximum"
else :
    bla = "--> polarisation RCHP au maximum"
    
print("E_g_max =", E_g_max)
print("E_d_max =", E_d_max)
print(bla)
print()

print("valeur intégrale de la norme du E total):",integrale_E)   
print()

#print("phi = pi/2 aux indices 15-18 :",data_phi[15][18] == np.pi/2)
#print("theta = pi/6 aux indices 15-18 :",data_theta[15][18] == np.pi/6)
#print()

print("Puissance disponible au récepteur [mW]:", Pr * 10**3)
print("Puissance reçue [mW]:", Pr_loss * 10**3)


import numpy as np
import matplotlib.pyplot as plt

"""
LELEC2910 : Antennes : Mini-Projet 2

-> Optimiser la directivité d'une antenne Yagi.

Utilisation:
- D'abord mettre les paramètres souhaités dans le fichier dédié (hors git).
- Lancer le script Matlab (hors git).
- Exécuter ce fichier Python.
"""

# Valeur intitiales des paramètres (NE PAS TOUCHER !):
    
#f  = 5.0000000e-01   # ne pas modifier
#d  = 2.55000000e-01
#N  = 12.000000e+00
#L  = 2.7000000e-01
#dL = 2.0000000e-02
#zg = 0.0000000e+01   # ne pas modifier

chemin_A = "../int_pat2_R2022a/"
chemin_T = 'ton chemin pour accéder aux fichiers renvoyés par le code Matlab'


# Extraction data ------------------------------------------------------------
# ============================================================================

# Setup le bon chemin pour pas avoir de problèmes avec git
while (True):
    chemin = input("Chemin de A ou T ?  ")
    if (chemin == "A" or chemin == "a"):
        chemin = chemin_A
        print()
        print()
        break
    elif (chemin == "T" or chemin == "t"):
        chemin = chemin_T
        print()
        print()
        break
    else :
        print("Chemin inexistant, recommencez.")

# Angles theta et phi :

theta = np.arange(1.5, 180 ,180/60)
theta = theta * np.pi / 180    # [rad]

phi = np.arange(0, 360, 360/80)
phi = phi * np.pi / 180        # [rad]

# Récupération des paramètres :
    
with open(chemin+"parameters.txt") as p :
    data = p.readlines()
    params = {}
    try :
        for i in range(6):
            data[i] = data[i].strip().strip("\n")
           
        params["f"]  = float(data[0])
        params["d"]  = float(data[1])
        params["N"]  = int(float(data[2]))
        params["L"]  = float(data[3])
        params["dL"] = float(data[4])
        params["zg"] = float(data[5])     
    
    except :
        raise Exception("Mauvais nombre de paramètres OU paramètres invalide.")

# Récupération des erreurs :

with open(chemin+"errors.txt") as err :    # Verifies that the power fed to 
    errors = err.readlines()               # the input impedance of the antenna
    errors = errors[0].split("   ")        # corresponds to the radiated power.
    
    count_err = 0
    idx_err = []
    
    for i in range(1,len(errors)):
        if (float(errors[i].strip()) > 0.1): # erreur supérieure à 1%
            count_err += 1
            idx_err.append(i)

# Récupération des F + indices angles :
    
idx_theta = np.zeros((params.get("N"), 4800), dtype=int)
idx_phi   = np.zeros((params.get("N"), 4800), dtype=int)
real_F    = np.zeros((params.get("N"), 4800), dtype=float)
imag_F    = np.zeros((params.get("N"), 4800), dtype=float)

for i in range(1,params.get("N")+1):
    with open(chemin+"pattern"+str(i)+".txt") as file:
        contenu = file.readlines()
        length = len(contenu)
        
        for l in range(length) :
            line = contenu[l].split("  ") # donne 5 elems, dont le premier est vide
            
            idx_theta[i-1][l] = int(float(line[1].strip())) 
            idx_phi[i-1][l]   = int(float(line[2].strip()))
            real_F[i-1][l]    = float(line[3].strip())
            imag_F[i-1][l]    = float(line[4].strip())
      
        print("File",i,"loaded.")
print()    


# Calcul Directivité ---------------------------------------------------------
# ============================================================================


"""
Jsp trop comment calculé la directivité.
- Eske le F d'une antenne correspond à son F quand l'antenne est seule ? Ou 
  bien le F donné prend déjà en compte l'impact des autres antennes ?
- Pour obtenir le F total, il faut faire le produit de tous les F ? Ou bien la 
  somme ?
- Pour calculer F_tot, esk'il faut d'abord sommer/multiplier tous les F et puis
  prendre la abs? Ou bien l'inverse ?
"""

F = real_F + imag_F * 1j
F_square_norm = abs(F)**2

def Riemann(f) :
    """
    Approximation de l'intégrale de f sur la surface d'une sphère 
    de rayon unitaire.
    """
    Jac = np.sin(theta)   
    integrale = 0              
    d_phi = (360/80) * np.pi / 180    
    d_theta = (180/60) * np.pi / 180  

    for i in range(60):       # theta
        for j in range(80):   # phi
            integrale += f[j + i*80] * Jac[i] * d_phi * d_theta
    
    return integrale
  
    
# Directivité pour chaque patterne :
    
mesh_phi, mesh_theta = np.meshgrid(phi, theta)  
  
for i in range(0,params.get("N"),3): # 3 diagrammes par figure
    fig = plt.figure()
    for j in range(3):
    
        D = 4 * np.pi * F_square_norm[i+j] / Riemann(F_square_norm[i+j])
        D = D.reshape(60,80)
        
        u = D * np.sin(mesh_theta) * np.cos(mesh_phi)
        v = D * np.sin(mesh_theta) * np.sin(mesh_phi)
        w = D * np.cos(mesh_theta)
        
        sub = "13"+str(j+1)
        ax = fig.add_subplot(int(sub),projection='3d')
        surf = ax.plot_surface(u, v, w,rstride=1, cstride=1)
    plt.title("Directivity versus u-v plane : pattern "+str(i+1)+" - "+str(i+3)) 


# Affichage console ----------------------------------------------------------
# ============================================================================


print("Paramètres :")
items = params.items()
for tup in items:
    print(tup[0],":",tup[1])
print()


if (count_err > 0):
    print("-------- WARNIG --------")
    print(len(idx_err), "Erreur(s) supérieure(s) à 1%.")
    for i in idx_err :
        print("Antenne",i,":",errors[i])
    print()
    
    
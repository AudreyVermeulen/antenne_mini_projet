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
    errors = errors[0].split("  ")         # corresponds to the radiated power.
    
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


F_square_norm = real_F**2 + imag_F**2

theta = 1.5 + 3 * (idx_theta - 1)
theta = theta * np.pi / 180 #[rad]
phi = 4.5 * (idx_phi - 1)
phi = phi * np.pi / 180     #[rad]

def Riemann(f) :
    """
    Approximation de l'intégrale de f sur la surface d'une sphère 
    de rayon unitaire.
    """
    Jac = np.sin(theta[0]* np.pi / 180)   
    integrale = 0              
    d_phi = (360/80) * np.pi / 180    
    d_theta = (180/60) * np.pi / 180  

    for i in range(60):       # theta
        for j in range(80):   # phi
            integrale += f[j + i*80] * Jac[j + i*80] * d_phi * d_theta
    
    return integrale
  
# Directivité pour chaque patterne :
    
D = np.zeros((params.get("N"), 4800))  

for i in range(params.get("N")):
    D[i] = 4 * np.pi * F_square_norm[i] / Riemann(F_square_norm[i])
    
D_dB = 10 * np.log10(D)
D_max_dB = np.zeros(params.get("N"))
for i in range(params.get("N")):
    D_max_dB[i] = D_dB[i].max()
    
ZE_max = D_max_dB.max()


# Plot u-v plane -------------------------------------------------------------
# ============================================================================



F15 = F_square_norm[14].reshape(60,80)
fig=plt.figure()
mesh_phi = phi[14].reshape(60,80)
mesh_theta = theta[14].reshape(60,80)

ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(mesh_phi, mesh_theta, F15,rstride=1, cstride=1,cmap='viridis')
plt.title("|F|² of last wire versus (phi , theta)") 
ax.set_xlabel("phi [Deg°]")
ax.set_ylabel("theta [Deg°]") 
#plt.savefig("graphes/F15.svg")

def plot_par_3() : 
    """
    Plot la directivité de chaque antenne, 3 graphes par figure.
    """
    for i in range(0,params.get("N"),3): # 3 diagrammes par figure
        fig = plt.figure()
        count = 0
        for j in range(3):
            if (i+j < params.get("N")):
                count +=1
                Di = D[i+j].reshape(60,80)
                
                u = Di * np.sin(mesh_theta) * np.cos(mesh_phi)
                v = Di * np.sin(mesh_theta) * np.sin(mesh_phi)
                w = Di * np.cos(mesh_theta)

                sub = "13"+str(j+1)
                ax = fig.add_subplot(int(sub),projection='3d')
                surf = ax.plot_surface(u, v, w,rstride=1, cstride=1)
        plt.title("Directivity versus u-v plane : patterns "+str(i+1)+" - "+str(i+count)) 
        #plt.savefig("graphes/mp2_D"+str(i)+".png")
plot_par_3()


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
else:
    print(" -- 0 errors")
    print()
    
for i in range(params.get("N")):
    print("D max antenne",i+1,"[dB] : ",D_max_dB[i])
print()
print("--> Maximum directivity [dB] :",ZE_max)
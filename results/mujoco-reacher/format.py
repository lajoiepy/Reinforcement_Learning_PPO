#!/usr/bin/python
import sys

# Ouverture du fichier de donnees
with open("mujoco_results.csv") as fichier:
    lignes_testset = fichier.readlines()    
# Fermeture du fichier de donnees
fichier.close()

# Ouverture du fichier de donnees
with open("mujoco_results_filtered.csv","w") as fichier:
    for i in range(len(lignes_testset)):
        if i%100==0:
            fichier.write(lignes_testset[i])   
# Fermeture du fichier de donnees
fichier.close()


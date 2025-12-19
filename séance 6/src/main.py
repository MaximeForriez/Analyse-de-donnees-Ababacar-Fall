#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

#Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    with open(nom, encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# --- 1. Définition du chemin absolu exact ---
# Nous utilisons le préfixe 'r' pour que Windows interprète bien les barres \
adresse_data = r"C:\test\Analyse d edonnées\séance 6\src\data\island-index.csv"

# --- 2. Fonction ouvrirUnFichier avec gestion d'encodage ---
def ouvrirUnFichier(nom):
    try:
        # On utilise 'utf-8-sig' pour gérer les fichiers créés par Excel
        return pd.read_csv(nom, sep=None, engine='python', encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier est introuvable à l'adresse :\n{nom}")
        return None
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")
        return None

# --- 3. Chargement effectif ---
iles = ouvrirUnFichier(adresse_data)

if iles is not None:
    print("✅ Fichier 'island-index.csv' chargé avec succès !")
    # Nettoyage des noms de colonnes (espaces invisibles)
    iles.columns = iles.columns.str.strip()
    print(iles.head())
#print(iles.head())

#Question 3 : isoler la colonne surface km² et ajouter la liste des continents
surfaces = iles["Surface (km²)"]
surfaces = surfaces.dropna()
print(surfaces.head())

continents = [85545323, 37856841, 7768030, 7605049]
surfaces = surfaces.astype(float)#en décimales
surfaces = pd.concat([surfaces, pd.Series(continents)], ignore_index=True)

print(surfaces.tail())

#Question 4 : ordre decroissant
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste
surfaces_triees = ordreDecroissant(surfaces.tolist())
print(surfaces_triees[:5])#les plus grandes valeurs

#Question 5 : loi rang-taille
rangs= list(range(1, len(surfaces_triees)+1))
plt.figure(figsize=(10,6))
plt.plot(rangs, surfaces_triees, marker='o', color='red')
plt.title("Loi rang-taille des surfaces des continents et des îles")
plt.xlabel("Rang")
plt.ylabel("Surface km²")
plt.savefig("img/loi_rang_taille.png")
plt.close()

#Question 6 : Fonction pour convertir les données en données logarithmiques ????? pourquoi faire
def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

rangs_log = conversionLog(rangs)
surfaces_log = conversionLog(surfaces_triees)
plt.figure(figsize=(10, 6))
plt.plot(rangs_log, surfaces_log, marker='o', linestyle='-', color='blue', linewidth=1)
plt.title("Loi rang–taille (axes converties en Log)")
plt.xlabel("log(rang)")
plt.ylabel("log(surface en km²)")
plt.grid(True, linewidth=0.3)
plt.savefig("img/loi_rang_taille_log.png", dpi=150)
plt.close()

#Question 7 : On peut pas faire des tests sur des rangs...car ce sont des rangs ? donc pas des données statistiques. 
#un rang = position dans un classement à partir des valeurs que j'ai triées c'est tout.




# --- Question 7 : Amélioration de la fonction d'ouverture ---
def ouvrirUnFichier(nom):
    """Ouvre un fichier CSV avec détection automatique du délimiteur et gestion des accents."""
    try:
        # 'utf-8-sig' est idéal pour les fichiers CSV contenant des accents (États, Densité)
        return pd.read_csv(nom, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ Erreur lors de l'ouverture de {nom} : {e}")
        return None

# --- Question 8 & 9 : Chargement effectif du fichier ---
chemin_monde = r"C:\test\Analyse d edonnées\séance 6\src\data\Le-Monde-HS-Etats-du-monde-2007-2025.csv"

# CORRECTION : On appelle la fonction avec son paramètre
monde = ouvrirUnFichier(chemin_monde)

# --- Question 10 : Isolation des colonnes ---
if monde is not None:
    # Nettoyage préventif des noms de colonnes (supprime les espaces invisibles)
    monde.columns = monde.columns.str.strip()
    
    # Liste des colonnes à isoler
    colonnes = ["État", "Pop 2007", "Pop 2025", "Densité 2007", "Densité 2025"]
    
    try:
        monde_selection = monde[colonnes]
        print("\n✅ Sélection des colonnes réussie :")
        print(monde_selection.head())
    except KeyError as e:
        print(f"❌ Erreur : L'une des colonnes est introuvable. Vérifiez l'orthographe : {e}")
        # Optionnel : afficher toutes les colonnes disponibles pour vérifier
        # print(monde.columns.tolist())


# --- Prérequis : Définition des fonctions de tri ---
def ordreDecroissant(liste):
    liste.sort(reverse=True)
    return list(liste)

def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if not np.isnan(pop[element]):
            ordrepop.append([float(pop[element]), etat[element]])
    
    # Tri par valeur numérique
    ordrepop = ordreDecroissant(ordrepop)
    
    # Remplacement de la valeur par le rang (1, 2, 3...)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

# --- Question 11 : Préparation des données ---
# On s'assure que 'monde' est déjà chargé ici
etats = monde["État"].tolist()
pop2007 = monde["Pop 2007"].astype(float).tolist()
pop2025 = monde["Pop 2025"].astype(float).tolist()
dens2007 = monde["Densité 2007"].astype(float).tolist()
dens2025 = monde["Densité 2025"].astype(float).tolist()

pop2007_ordre = ordrePopulation(pop2007, etats)
pop2025_ordre = ordrePopulation(pop2025, etats)
dens2007_ordre = ordrePopulation(dens2007, etats)
dens2025_ordre = ordrePopulation(dens2025, etats)

print("Top 10 Pop 2007 :", pop2007_ordre[:10])




# --- Question 12 : Fonction de comparaison ---
def classementPays(ordre1, ordre2):
    classement = []
    # On compare les deux listes pour trouver les pays communs et leurs rangs respectifs
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2)):
            for element2 in range(0, len(ordre1)):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1)):
            for element2 in range(0, len(ordre2)):
                if ordre2[element2][1] == ordre1[element1][1]:
                    # Correction ici : element1 au lieu de element
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element1][1]])
    return classement

comparaison_2007 = classementPays(pop2007_ordre, dens2007_ordre)
comparaison_2025 = classementPays(pop2025_ordre, dens2025_ordre)

# --- Question 13 : Isolation des rangs ---
rangs_pop_2007 = [ligne[0] for ligne in comparaison_2007]
rangs_dens_2007 = [ligne[1] for ligne in comparaison_2007]

rangs_pop_2025 = [ligne[0] for ligne in comparaison_2025]
rangs_dens_2025 = [ligne[1] for ligne in comparaison_2025]



from scipy.stats import spearmanr, kendalltau

print("\n--- ANALYSE DE CORRÉLATION DES RANGS ---")

# Calcul pour 2007
rho_2007, p_s2007 = spearmanr(rangs_pop_2007, rangs_dens_2007)
tau_2007, p_k2007 = kendalltau(rangs_pop_2007, rangs_dens_2007)

print(f"2007 - Spearman : {rho_2007:.4f} (p={p_s2007:.2e})")
print(f"2007 - Kendall  : {tau_2007:.4f} (p={p_k2007:.2e})")

# Calcul pour 2025
rho_2025, p_s2025 = spearmanr(rangs_pop_2025, rangs_dens_2025)
tau_2025, p_k2025 = kendalltau(rangs_pop_2025, rangs_dens_2025)

print(f"2025 - Spearman : {rho_2025:.4f} (p={p_s2025:.2e})")
print(f"2025 - Kendall  : {tau_2025:.4f} (p={p_k2025:.2e})")





# Chemins Absolus 
path_iles = r"C:\test\Analyse d edonnées\séance 6\src\data\island-index.csv"
path_monde = r"C:\test\Analyse d edonnées\séance 6\src\data\Le-Monde-HS-Etats-du-monde-2007-2025.csv"

# Chargement robuste
iles = ouvrirUnFichier(path_iles)
if iles is not None:
    # Nettoyage des noms de colonnes
    iles.columns = iles.columns.str.replace('"', '').str.strip()
    
    # Correction des noms de colonnes selon l'aperçu précédent
    surfaces = iles["Surface (km²)"].tolist()
    cotes = iles["Trait de côte (km)"].tolist()
    noms = iles["Toponyme"].tolist() # ou 'Identifiant' selon le fichier
    
    print("✅ Données des îles extraites.")



#  Bonus 
#  Analyse des Îles 
# --- Étape 1 : Extraction propre des données des îles ---
if iles is not None:
    # On nettoie les noms de colonnes pour éviter les erreurs
    iles.columns = iles.columns.str.replace('"', '').str.strip()
    
    # Extraction en listes (Vérifiez bien ces noms dans votre console)
    noms_iles = iles["Toponyme"].tolist()
    surfaces_iles = iles["Surface (km²)"].tolist()
    traits_cotes = iles["Trait de côte (km)"].tolist()

    # --- Étape 2 : Algorithme de comparaison des rangs ---
    # On crée les deux classements
    rangs_surface = ordrePopulation(surfaces_iles, noms_iles)
    rangs_cotes = ordrePopulation(traits_cotes, noms_iles)

    # On compare les deux (appariement)
    comparaison_iles = classementPays(rangs_surface, rangs_cotes)

    # Extraction des vecteurs de rangs pour les stats
    r_surf = [ligne[0] for ligne in comparaison_iles]
    r_cote = [ligne[1] for ligne in comparaison_iles]

    # Calcul de la corrélation
    coef_s_iles, _ = spearmanr(r_surf, r_cote)
    print(f"\n✅ Corrélation Îles (Surface vs Côte) : {coef_s_iles:.4f}")



    
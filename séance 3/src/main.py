#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023



# Question 4
chemin = r'./data/resultats-elections-presidentielles-2022-1er-tour.csv'
with open(chemin, mode='r', encoding='utf-8') as fichier:
    # La variable 'fichier' est passée directement à read_csv
    df = pd.read_csv(fichier, sep=',', quotechar='"')

df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
print(df.head())

# Question 5

try:
    with open(chemin, mode='r', encoding='utf-8') as fichier:
        df = pd.read_csv(fichier, sep=',', quotechar='"')
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()

    # 2. Sélection des colonnes quantitatives
    # On identifie les colonnes de base + les colonnes "Voix" répétées
    colonnes_quantitatives = ['Inscrits', 'Abstentions', 'Votants', 'Blancs', 'Nuls', 'Exprimés']
    cols_voix = [c for c in df.columns if 'Voix' in c]
    colonnes_cibles = colonnes_quantitatives + cols_voix

    # Conversion en numérique pour assurer les calculs
    for col in colonnes_cibles:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Calcul des paramètres statistiques sous forme de listes
    # On utilise le DataFrame restreint aux colonnes numériques pour simplifier
    df_quant = df[colonnes_cibles]

    # — Moyennes
    moyennes = df_quant.mean().round(2).tolist()

    # — Médianes
    medianes = df_quant.median().round(2).tolist()

    # — Modes (on prend la première valeur du mode pour chaque colonne)
    modes = df_quant.mode().iloc[0].round(2).tolist()

    # — Écart type (Standard Deviation)
    ecarts_types = df_quant.std().round(2).tolist()

    # — Écart absolu à la moyenne (MAD - Mean Absolute Deviation)
    # Formule : Moyenne de la valeur absolue des écarts (utilisation de np.abs)
    ecarts_absolus = []
    for col in colonnes_cibles:
        mad = np.abs(df[col] - df[col].mean()).mean()
        ecarts_absolus.append(round(mad, 2))

    # — Étendue (Max - Min)
    etendues = (df_quant.max() - df_quant.min()).round(2).tolist()

    # 4. Affichage des résultats
    print("--- RÉSULTATS STATISTIQUES (Colonnes Quantitatives) ---\n")
    print(f"Colonnes analysées : {colonnes_cibles}\n")
    print(f"Moyennes : {moyennes}")
    print(f"Médianes : {medianes}")
    print(f"Modes : {modes}")
    print(f"Écarts types : {ecarts_types}")
    print(f"Écarts absolus moyens : {ecarts_absolus}")
    print(f"Étendues : {etendues}")

except Exception as e:
    print(f"Une erreur est survenue : {e}")



# Question 6
# Affichage structuré des paramètres statistiques
print("\n" + "="*60)
print("SYNTHÈSE DES PARAMÈTRES STATISTIQUES (Séance 3)")
print("="*60)

# On boucle sur les colonnes pour afficher les résultats par variable
for i, col in enumerate(colonnes_cibles):
    print(f"\n📍 VARIABLE : {col}")
    print(f"  - Moyenne :                {moyennes[i]}")
    print(f"  - Médiane :                {medianes[i]}")
    print(f"  - Mode :                   {modes[i]}")
    print(f"  - Écart type :             {ecarts_types[i]}")
    print(f"  - Écart absolu moyen :     {ecarts_absolus[i]}")
    print(f"  - Étendue (Max - Min) :    {etendues[i]}")

print("\n" + "="*60)



# Question 7
# Listes pour stocker les nouveaux paramètres
distances_interquartiles = []
distances_interdeciles = []

# Calcul pour chaque colonne quantitative cible
for col in colonnes_cibles:
    # --- Calcul des quartiles (25% et 75%) ---
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    distances_interquartiles.append(round(iqr, 2))
    
    # --- Calcul des déciles (10% et 90%) ---
    d1 = df[col].quantile(0.1)
    d9 = df[col].quantile(0.9)
    idr = d9 - d1
    distances_interdeciles.append(round(idr, 2))

# --- Affichage sur le terminal ---
print("\n" + "="*60)
print("DISTANCES INTERQUARTILES ET INTERDÉCILES")
print("="*60)

for i, col in enumerate(colonnes_cibles):
    print(f"\n📍 VARIABLE : {col}")
    print(f"  - Distance Interquartile (IQR) : {distances_interquartiles[i]}")
    print(f"  - Distance Interdécile (IDR)    : {distances_interdeciles[i]}")

print("\n" + "="*60)

# Question 8
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# --- 1. Configuration et Chargement ---
chemin = r'./data/resultats-elections-presidentielles-2022-1er-tour.csv'
dossier_sortie = "img"

print("Chargement des données...")
try:
    with open(chemin, mode='r', encoding='utf-8') as fichier:
        df = pd.read_csv(fichier, sep=',', quotechar='"')
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
except FileNotFoundError:
    print(f"ERREUR : Fichier introuvable : {chemin}")
    exit()

# --- 2. Préparation des colonnes quantitatives ---
print("Préparation des colonnes quantitatives...")
# On liste les colonnes de base
cols_base = ['Inscrits', 'Abstentions', 'Votants', 'Blancs', 'Nuls', 'Exprimés']
# On trouve dynamiquement toutes les colonnes "Voix" des candidats
cols_voix = [c for c in df.columns if 'Voix' in c]
# On réunit le tout
colonnes_quantitatives = cols_base + cols_voix

# CONVERSION CRUCIALE EN NUMÉRIQUE
for col in colonnes_quantitatives:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- 3. Création du dossier de sortie ---
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)
    print(f"Dossier '{dossier_sortie}' créé.")

# --- 4. Boucle de génération des Boîtes à moustache ---
print(f"Début de la génération des graphiques dans le dossier '{dossier_sortie}'...")

for col in colonnes_quantitatives:
    fig = None
    try:
        # Données à tracer (on s'assure qu'il n'y a pas de NaN)
        data_to_plot = df[col].dropna()

        if len(data_to_plot) == 0:
            continue

        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # --- Création de la Boîte à moustache ---
        # vert=False : Pour l'avoir à l'horizontale (plus lisible ici)
        # patch_artist=True : Permet de colorier la boîte
        # showfliers=True : Affiche les points aberrants (les "diamants" au-delà des moustaches)
        boxplot = ax.boxplot(data_to_plot, vert=False, patch_artist=True, showfliers=True,
                             boxprops=dict(facecolor='lightblue', color='blue', linewidth=1.5),
                             whiskerprops=dict(color='blue', linewidth=1.5),
                             capprops=dict(color='blue', linewidth=2),
                             medianprops=dict(color='red', linewidth=2.5),
                             flierprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=4, alpha=0.5))

        # --- Habillage du graphique ---
        ax.set_title(f"Distribution (Boîte à moustache) : {col}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Nombre d'individus / Voix", fontsize=12)
        # On enlève l'étiquette "1" sur l'axe Y qui est inutile ici
        ax.set_yticklabels([]) 
        
        # Ajout d'une grille verticale pour faciliter la lecture
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # --- Sauvegarde sécurisée ---
        # Nettoyage du nom de fichier (surtout pour les colonnes "Voix" bizarres)
        nom_propre = re.sub(r'[^a-zA-Z0-9]', '_', col)
        chemin_fichier = os.path.join(dossier_sortie, f"boxplot_{nom_propre}.png")

        plt.savefig(chemin_fichier, bbox_inches='tight')
        # print(f"✅ Généré : {col}") # Décommentez pour voir le défilement

    except Exception as e:
        print(f"❌ Erreur lors de la création du boxplot pour {col} : {e}")

    finally:
        # IMPORTANT : Fermer la figure pour libérer la mémoire dans la boucle
        if fig:
            plt.close(fig)

print(f"\n--- Terminé ! ---")
print(f"Les images sont enregistrées dans : {os.path.abspath(dossier_sortie)}")
# os.startfile(os.path.abspath(dossier_sortie)) # Ouvrir le dossier sous Windows



# Question 9
chemin_island = r'./data/island-index.csv'



# Question 10
import pandas as pd
import os

# Chemin vers le fichier de la séance 3
chemin_island = r'./data/island-index.csv'

# Étape 4 : Instruction 'with' et variable 'fichier'
try:
    with open(chemin_island, mode='r', encoding='utf-8-sig') as fichier: # utf-8-sig gère les caractères cachés Excel
        contenu = pd.read_csv(fichier, sep=None, engine='python') # Détection auto du séparateur

    # Étape 5 : Utilisation de DataFrame(...) sur la variable contenu
    df_islands = pd.DataFrame(contenu)

    # NETTOYAGE CRITIQUE : on renomme les colonnes pour supprimer TOUT caractère spécial
    df_islands.columns = [c.strip().replace('"', '').replace("'", "") for c in df_islands.columns]
    
    # Étape 6 : Affichage du nombre de lignes et colonnes avec len()
    print(f"Nombre de lignes : {len(df_islands)}")
    print(f"Nombre de colonnes : {len(df_islands.columns)}")
    print("Colonnes réelles trouvées :", df_islands.columns.tolist())

    # Identification automatique de la colonne Surface (même si le nom varie légèrement)
    col_surface = [c for c in df_islands.columns if 'Surface' in c][0]
    print(f"Utilisation de la colonne : {col_surface}")

    # --- ALGORITHME DE CATÉGORISATION ---
    # Conversion en numérique et arrondi à deux décimaux
    df_islands[col_surface] = pd.to_numeric(df_islands[col_surface], errors='coerce').round(2)

    # Définition des intervalles (bins)
    bornes = [0, 10, 25, 50, 100, 2500, 5000, 10000, float('inf')]
    noms_categories = ["]0, 10]", "]10, 25]", "]25, 50]", "]50, 100]", 
                       "]100, 2500]", "]2500, 5000]", "]5000, 10000]", "> 10000"]

    # Création de la variable qualitative
    df_islands['Categorie'] = pd.cut(df_islands[col_surface], bins=bornes, labels=noms_categories)

    # Dénombrement
    print("\nRésultat du dénombrement :")
    print(df_islands['Categorie'].value_counts().sort_index())

except Exception as e:
    print(f"Erreur persistante : {e}")


# Bonnus

stats = {
    'Moyenne': round(df_islands[col_surface].mean(), 2),
    'Médiane': round(df_islands[col_surface].median(), 2),
    'Mode': round(df_islands[col_surface].mode()[0], 2),
    'Écart-type': round(df_islands[col_surface].std(), 2),
    'Étendue': round(df_islands[col_surface].max() - df_islands[col_surface].min(), 2),
    'Distance Interquartile': round(df_islands[col_surface].quantile(0.75) - df_islands[col_surface].quantile(0.25), 2)
}

# Création d'un DataFrame pour l'export
df_stats = pd.DataFrame(list(stats.items()), columns=['Paramètre', 'Valeur'])

# Exportation C.S.V.
df_stats.to_csv(r'./data/stats_islands.csv', sep=';', index=False, encoding='utf-8-sig')

# Exportation Excel
try:
    df_stats.to_excel(r'./data/stats_islands.xlsx', index=False)
    print("✅ Exportation C.S.V. et Excel réussie.")
except ImportError:
    print("⚠️ Pour Excel, installez openpyxl : pip install openpyxl")


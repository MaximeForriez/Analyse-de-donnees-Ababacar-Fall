#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open(r"./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)

# Mettre dans un commentaire le numéro de la question
# ...
# Question 5 : Afficher sur le terminal exécutant

# Afficher les 5 premières lignes avec la méthode .head()
print(contenu.head())

# Question 6 

# Calcul du nombre de lignes
nb_lignes = len(contenu)

# Calcul du nombre de colonnes (on mesure la longueur de la liste des colonnes)
nb_colonnes = len(contenu.columns)

# Affichage sur le terminal
print(f"Nombre de lignes : {nb_lignes}")
print(f"Nombre de colonnes : {nb_colonnes}")

#  Question 7 

# On récupère les types de colonnes
types_colonnes = contenu.dtypes

print("--- Analyse des types de colonnes ---")
# On boucle sur chaque colonne pour afficher son nom et son type simplifié
for col, dtype in types_colonnes.items():
    # Traduction des types Pandas vers les types demandés (str, int, float)
    type_python = ""
    if dtype == 'object':
        type_python = "str"
    elif 'int' in str(dtype):
        type_python = "int"
    elif 'float' in str(dtype):
        type_python = "float"
    elif 'bool' in str(dtype):
        type_python = "bool"
    
    print(f"Colonne : {col} | Type : {type_python}")
chemin = r"./data/resultats-elections-presidentielles-2022-1er-tour.csv"

contenu = pd.read_csv(chemin, sep=";", encoding="ISO-8859-1")
print("Fichier chargé avec succès !")
import os

if os.path.exists(chemin):
    contenu = pd.read_csv(chemin, sep=";", encoding="ISO-8859-1")
    print("Le fichier a été trouvé et chargé.")
else:
    print(f"ERREUR : Le fichier est introuvable à l'adresse : {chemin}")

#  Question 8
# Affiche la 1ère ligne de données avec les noms de colonnes au-dessus

df = pd.read_csv(r'./data/resultats-elections-presidentielles-2022-1er-tour.csv', encoding='ISO-8859-1', sep=';')
print(df.head(1))

#  Question 9 
df = pd.read_csv(chemin, sep=',', encoding='utf-8')
df.columns = df.columns.str.replace('"', '').str.strip()
print("Les colonnes sont maintenant bien séparées :")
print(df.columns.tolist()[:5])

print("-" * 30)
print("Voici les premières lignes de la colonne Inscrits :")
print(df["Inscrits"].head())

#  Question 10
resultats_bizarres = []

for col in df.columns:
    somme = df[col].sum()
    resultats_bizarres.append(somme)

print("Résultats bizarres (mélange texte et nombres) :")
print(resultats_bizarres[:3]) # On affiche les 3 premiers pour voir le texte collé


# On définit d'abord la liste des colonnes quantitatives (basé sur l'exercice 1)
colonnes_quantitatives = ['Inscrits', 'Abstentions', 'Votants', 'Blancs', 'Nuls', 'Exprimés', 'Voix']

effectifs_quantitatifs = []

# Boucle avec condition
for col in df.columns:
    if col in colonnes_quantitatives:
        # On s'assure que la donnée est bien traitée comme un nombre (float ou int)
        somme = df[col].astype(float).sum()
        effectifs_quantitatifs.append(somme)

print("Sommes des colonnes quantitatives :")
print(effectifs_quantitatifs)

#  Question 11
import os
import re # Import pour le nettoyage de texte
import pandas as pd
import matplotlib.pyplot as plt

# 1. Préparation du dossier
dossier_images = "graphiques_departements"
if not os.path.exists(dossier_images):
    os.makedirs(dossier_images)

# 2. Boucle sécurisée
for nom_dept, data_dept in df.groupby("Libellé du département"):
    try:
        # --- Nettoyage du nom pour le fichier ---
        # Cette ligne remplace TOUT ce qui n'est pas une lettre ou un chiffre par un "_"
        # "Saint-Martin/Saint-Barthélemy" deviendra "Saint_Martin_Saint_Barth_lemy"
        nom_fichier_secu = re.sub(r'[^a-zA-Z0-9]', '_', nom_dept)
        
        chemin_complet = os.path.join(dossier_images, f"participation_{nom_fichier_secu}.png")

        # --- Création du graphique ---
        total_inscrits = data_dept['Inscrits'].sum()
        total_votants = data_dept['Votants'].sum()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['Inscrits', 'Votants'], [total_inscrits, total_votants], color=['#3498db', '#2ecc71'])
        ax.set_title(f"Département : {nom_dept}")
        
        # --- Sauvegarde ---
        plt.savefig(chemin_complet)
        plt.close(fig)
        print(f"✅ Graphique généré pour : {nom_dept}")

    except Exception as e:
        print(f"❌ Erreur critique pour {nom_dept} : {e}")
        if 'fig' in locals(): plt.close(fig)

# 3. Ouverture du dossier à la fin
print("\n--- TERMINE ---")
os.startfile(os.path.abspath(dossier_images))


#  Question 12

import pandas as pd
import matplotlib.pyplot as plt
import os
import re # Nécessaire pour le nettoyage "costaud" des noms de fichiers

# --- CONFIGURATION ---
# Remplacez par le chemin exact de votre fichier si différent
chemin_csv = r'./data/resultats-elections-presidentielles-2022-1er-tour.csv'
dossier_sortie = "diagrammes_circulaires_repartition"
colonnes_a_analyser = ['Blancs', 'Nuls', 'Exprimés', 'Abstentions']

# --- 1. CHARGEMENT ET NETTOYAGE PREALABLE ---
print("Chargement des données...")
try:
    # On utilise les paramètres qui ont fonctionné précédemment
    df = pd.read_csv(chemin_csv, sep=',', encoding='utf-8', quotechar='"')
    # Nettoyage des noms de colonnes (suppression guillemets et espaces)
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
except FileNotFoundError:
    print(f"ERREUR CRITIQUE : Le fichier {chemin_csv} est introuvable.")
    exit()

# --- 2. CONVERSION EN NOMBRES (Etape Cruciale) ---
print("Conversion des colonnes en données numériques...")
# On boucle sur les 4 colonnes dont on a besoin pour s'assurer qu'elles sont des nombres
for col in colonnes_a_analyser:
    # errors='coerce' transforme les textes non convertibles en NaN (Not a Number)
    # .fillna(0) remplace ces NaN par 0 pour ne pas bloquer les calculs
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- 3. PREPARATION DU DOSSIER DE SORTIE ---
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)
    print(f"Dossier '{dossier_sortie}' créé.")

# --- 4. BOUCLE PRINCIPALE DE GENERATION ---
print("\nDébut de la génération des diagrammes circulaires...")

# On groupe les données par département et on itère sur chaque groupe
for nom_dept, data_dept in df.groupby("Libellé du département"):
    try:
        # --- A. Calcul des sommes pour ce département ---
        # On fait la somme de chaque catégorie pour le département en cours
        somme_blancs = data_dept['Blancs'].sum()
        somme_nuls = data_dept['Nuls'].sum()
        somme_exprimes = data_dept['Exprimés'].sum()
        somme_abstentions = data_dept['Abstentions'].sum()

        # Préparation des listes pour Matplotlib
        labels = ['Blancs', 'Nuls', 'Exprimés', 'Abstention']
        valeurs = [somme_blancs, somme_nuls, somme_exprimes, somme_abstentions]

        # Définition de couleurs personnalisées (Gris, Gris foncé, Vert, Rouge)
        couleurs = ['#bdc3c7', '#7f8c8d', '#2ecc71', '#e74c3c']

        # --- B. Création du Pie Chart ---
        fig, ax = plt.subplots(figsize=(7, 7))

        # Création du camembert
        # autopct='%1.1f%%' : affiche le pourcentage avec 1 chiffre après la virgule
        # startangle=90 : tourne le graphique pour avoir le début à "midi"
        # wedgeprops : ajoute une petite bordure blanche pour la lisibilité
        ax.pie(valeurs, labels=labels, colors=couleurs, autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'white'})

        # Ajout du titre
        ax.set_title(f"Répartition des votes : {nom_dept}", fontsize=14, fontweight='bold')

        # Assure que le pie chart est bien un cercle parfait et non une ellipse
        ax.axis('equal')

        # --- C. Sauvegarde Sécurisée ---
        # Nettoyage robuste du nom de fichier avec Regex (garde lettres, chiffres et accents)
        nom_propre = re.sub(r'[^a-zA-Z0-9àâéèêëîïôùûçÀÂÉÈÊËÎÏÔÙÛÇ\s]', '_', nom_dept)
        # Remplace les espaces restants par des underscores
        nom_propre = nom_propre.replace(' ', '_')

        nom_fichier = f"repartition_{nom_propre}.png"
        chemin_complet = os.path.join(dossier_sortie, nom_fichier)

        # Sauvegarde de l'image (bbox_inches='tight' enlève les marges blanches inutiles)
        plt.savefig(chemin_complet, bbox_inches='tight')

        # --- D. Gestion de la mémoire (TRES IMPORTANT) ---
        plt.close(fig)
        # print(f"-> OK : {nom_dept}") # Décommentez si vous voulez voir le défilement

    except Exception as e:
        print(f"❌ Problème avec le département {nom_dept} : {e}")
        plt.close(fig) # On ferme la figure même en cas d'erreur
        continue

print(f"\n--- Terminé ! ---")
print(f"Les images sont enregistrées dans : {os.path.abspath(dossier_sortie)}")

# Optionnel : Ouvre le dossier automatiquement sous Windows
# os.startfile(os.path.abspath(dossier_sortie))


#  Question 13

# --- 1. Configuration et Chargement (Comme précédemment) ---
# Remplacez par votre chemin exact
chemin_csv = r'./data/resultats-elections-presidentielles-2022-1er-tour.csv'

print("Chargement et préparation des données...")
try:
    # Lecture avec les bons paramètres pour gérer les guillemets et virgules
    df = pd.read_csv(chemin_csv, sep=',', encoding='utf-8', quotechar='"')

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()

except FileNotFoundError:
    print(f"ERREUR : Le fichier est introuvable : {chemin_csv}")
    exit()

# --- 2. Préparation de la variable quantitative "Inscrits" ---
# ÉTAPE CRUCIALE : On convertit la colonne texte en nombres.
# 'errors='coerce'' transforme les valeurs illisibles en NaN (Not a Number)
df['Inscrits'] = pd.to_numeric(df['Inscrits'], errors='coerce')

# On supprime les éventuelles lignes vides (NaN) qui gêneraient le graphique
data_inscrits = df['Inscrits'].dropna()


# --- 3. Création de l'Histogramme avec Matplotlib ---
print("Génération de l'histogramme...")

# Création de la figure et définition de sa taille (largeur, hauteur en pouces)
plt.figure(figsize=(12, 7))

# Fonction principale pour l'histogramme : plt.hist()
# - data_inscrits : la série de nombres à analyser
# - bins=100 : Le nombre de "barres" (intervalles).
#              Plus le chiffre est élevé, plus le graphique est précis mais bruité.
#              50 à 100 est souvent un bon point de départ pour ce volume de données.
# - color et edgecolor : pour l'esthétique
plt.hist(data_inscrits, bins=100, color='#3498db', edgecolor='black', linewidth=0.5)

# --- 4. Habillage du graphique (Titres et Légendes) ---
# Titre principal
plt.title('Distribution du nombre d\'inscrits par lieu de vote\n(Élection Présidentielle 2022 - T1)',
          fontsize=16, fontweight='bold')

# Légende de l'axe X (Variable quantitative)
plt.xlabel('Nombre d\'inscrits (taille du bureau de vote/commune)', fontsize=12)

# Légende de l'axe Y (Fréquence)
plt.ylabel('Fréquence (Nombre de bureaux de vote concernés)', fontsize=12)

# Ajout d'une grille horizontale pour faciliter la lecture des hauteurs
plt.grid(axis='y', alpha=0.5, linestyle='--')

# --- 5. Affichage ---
# Ajuste automatiquement les marges pour que tout rentre bien
plt.tight_layout()

# Affiche une fenêtre interactive avec le graphique
print("Affichage du graphique. Fermez la fenêtre pour terminer le script.")
plt.show()



# Bonus

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# --- CONFIGURATION ---
chemin_csv = r'./data/resultats-elections-presidentielles-2022-1er-tour.csv'
dossier_sortie = "graphiques_candidats_par_dept"
seuil_visibilite = 3.0 # Pourcentage en dessous duquel un candidat est mis dans "Autres"

# --- 1. CHARGEMENT ET PREPARATION GLOBALE ---
print("Chargement des données...")
try:
    df = pd.read_csv(chemin_csv, sep=',', encoding='utf-8', quotechar='"')
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
except FileNotFoundError:
    print(f"ERREUR : Fichier introuvable : {chemin_csv}")
    exit()

# --- 2. IDENTIFICATION DES CANDIDATS (L'étape clé) ---
print("Identification des colonnes et des candidats...")

# On repère toutes les colonnes qui contiennent des voix et des noms
cols_voix_brutes = [c for c in df.columns if 'Voix' in c]
cols_noms_brutes = [c for c in df.columns if 'Nom' in c]

# On récupère les vrais noms des candidats depuis la PREMIÈRE ligne de données
# On utilise .astype(str) pour éviter des erreurs si un nom est lu comme un nombre
noms_candidats_officiels = [str(df[c].iloc[0]).title() for c in cols_noms_brutes]

# Conversion forcée en numérique pour toutes les colonnes de voix AVANT la boucle
print("Conversion des voix en numérique...")
for col in cols_voix_brutes:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- 3. PREPARATION DU DOSSIER ---
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

# --- 4. BOUCLE DE GENERATION PAR DEPARTEMENT ---
print("\nDémarrage de la génération des graphiques...")

for nom_dept, data_dept in df.groupby("Libellé du département"):
    fig = None # On initialise la figure pour pouvoir la fermer dans le 'finally'
    try:
        # A. Calcul des voix pour ce département
        # On fait la somme de chaque colonne "Voix.X" pour ce département
        voix_dept = [data_dept[c].sum() for c in cols_voix_brutes]
        total_votes_dept = sum(voix_dept)

        # Si le département n'a pas de votes (ex: données manquantes), on passe
        if total_votes_dept == 0:
            print(f"⚠️ Pas de votes trouvés pour {nom_dept}, on ignore.")
            continue

        # B. Filtrage pour la lisibilité (Création du groupe "Autres")
        labels_finaux = []
        valeurs_finales = []
        somme_autres = 0

        # On parcourt chaque candidat et son nombre de voix
        for voix, nom in zip(voix_dept, noms_candidats_officiels):
            pourcentage = (voix / total_votes_dept) * 100
            if pourcentage >= seuil_visibilite:
                # C'est un "gros" candidat, on le garde
                labels_finaux.append(nom)
                valeurs_finales.append(voix)
            else:
                # C'est un "petit" candidat, on l'ajoute à la somme "Autres"
                somme_autres += voix

        # Si la catégorie "Autres" n'est pas vide, on l'ajoute à la fin
        if somme_autres > 0:
            labels_finaux.append("Autres candidats (<3%)")
            valeurs_finales.append(somme_autres)

        # C. Création du Diagramme Circulaire
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Utilisation d'une palette de couleurs distinctes
        couleurs = plt.cm.tab20(np.linspace(0, 1, len(labels_finaux)))

        # Le camembert
        wedges, texts, autotexts = ax.pie(valeurs_finales, 
                                         labels=labels_finaux,
                                         autopct='%1.1f%%', # Affiche le % avec 1 décimale
                                         startangle=140,
                                         colors=couleurs,
                                         wedgeprops={'edgecolor': 'white', 'linewidth': 1})

        ax.set_title(f"Répartition des votes : {nom_dept}\n(Total exprimés : {int(total_votes_dept):,})", 
                     fontsize=14, fontweight='bold')
        plt.axis('equal')

        # D. Sauvegarde sécurisée
        nom_secu = re.sub(r'[^a-zA-Z0-9àâéèêëîïôùûçÀÂÉÈÊËÎÏÔÙÛÇ\s]', '_', nom_dept).replace(' ', '_')
        chemin = os.path.join(dossier_sortie, f"candidats_{nom_secu}.png")
        
        plt.savefig(chemin, bbox_inches='tight')
        # print(f"✅ {nom_dept} : OK") # Décommentez pour voir le défilement

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE sur {nom_dept} : {e}")

    finally:
        # E. Gestion de la mémoire : on ferme la figure quoi qu'il arrive
        if fig:
            plt.close(fig)

print(f"\n--- Terminé ! Les graphiques sont dans '{dossier_sortie}' ---")
try:
    os.startfile(os.path.abspath(dossier_sortie))
except Exception:
    pass # Ne fait rien sur Mac/Linux




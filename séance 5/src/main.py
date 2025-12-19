#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(r"C:\test\Analyse d edonnées\séance 5\src\data\Echantillonnage-100-Echantillons.csv") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv"))

#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")


#  Question 1

import pandas as pd
import numpy as np

#  Étape 1 : Fonction d'ouverture locale 
def ouvrirUnFichier(adresse):
    """Ouvre le fichier CSV en utilisant le gestionnaire de contexte 'with'."""
    try:
        # L'encodage 'utf-8-sig' est utilisé pour gérer les caractères français
        with open(adresse, mode='r', encoding='utf-8-sig') as fichier:
            df = pd.read_csv(fichier, sep=None, engine='python')
        # Nettoyage des colonnes (suppression des guillemets et espaces)
        df.columns = df.columns.str.replace('"', '').str.strip()
        return df
    except FileNotFoundError:
        print(f"Erreur : Fichier introuvable à l'adresse : {adresse}")
        return None

# Étape 2 : Chargement et calculs 
# Utilisation du chemin absolu pour corriger l'erreur système
chemin_absolu = r"C:\test\Analyse d edonnées\séance 5\src\data\Echantillonnage-100-Echantillons.csv"
donnees = ouvrirUnFichier(chemin_absolu)

if donnees is not None:
    # Calcul des moyennes par colonne (arrondi à l'entier le plus proche)
    moyennes = donnees.mean().round(0).astype(int)
    
    # Calcul des fréquences de l'échantillon (f)
    n_moyen = moyennes.sum() # Taille totale moyenne d'un échantillon
    frequences_obs = (moyennes / n_moyen).round(2)
    
    # Fréquences réelles de la population mère (p)
    pop_mere = {'Pour': 852, 'Contre': 911, 'Sans opinion': 422}
    total_pop = 2185
    frequences_reelles = {k: round(v / total_pop, 2) for k, v in pop_mere.items()}

    #  Étape 3 : Calcul de l'intervalle de fluctuation à 95 % 
    # Formule : f +/- 1.96 * sqrt( (f * (1-f)) / n )
    zc = 1.96
    resultats = []
    
    for opinion in moyennes.index:
        f = frequences_obs[opinion]
        p_reel = frequences_reelles[opinion]
        
        # Calcul de la marge d'erreur
        marge = zc * np.sqrt((f * (1 - f)) / n_moyen)
        borne_inf = round(f - marge, 2)
        borne_sup = round(f + marge, 2)
        
        resultats.append({
            'Opinion': opinion,
            'Fréq. Obs (f)': f,
            'Fréq. Réelle (p)': p_reel,
            'Intervalle [95%]': f"[{borne_inf} ; {borne_sup}]"
        })

    # Affichage sur le terminal
    print("\n--- RÉSULTATS DE L'ÉCHANTILLONNAGE (SÉANCE 5) ---")
    print(pd.DataFrame(resultats))


# Question 2 
import pandas as pd
import numpy as np

# --- 1. Extraction du premier échantillon (Ligne 0) ---
# Note : on utilise 'donnees' qui est le nom défini lors du chargement
if 'donnees' in locals() and donnees is not None:
    # Sélection de la première ligne avec iloc
    premiere_ligne_pandas = donnees.iloc[0]
    
    # Conversion en liste native Python (Casting)
    valeurs_echantillon = list(premiere_ligne_pandas)
    noms_colonnes = list(donnees.columns)
    
    # --- 2. Calcul de l'effectif et des fréquences de l'échantillon ---
    n_isole = sum(valeurs_echantillon) # Somme de la ligne
    frequences_isolees = [v / n_isole for v in valeurs_echantillon]
    
    # --- 3. Calcul des Intervalles de Confiance (Seuil 95%, zc = 1.96) ---
    zc = 1.96
    resultats_estimation = []
    
    for i in range(len(frequences_isolees)):
        f = frequences_isolees[i]
        # Formule de l'Intervalle de Confiance : f +/- zc * sqrt(f*(1-f)/n)
        marge = zc * np.sqrt((f * (1 - f)) / n_isole)
        ic_inf = round(f - marge, 2)
        ic_sup = round(f + marge, 2)
        
        resultats_estimation.append({
            'Opinion': noms_colonnes[i],
            'Effectif': valeurs_echantillon[i],
            'Fréq. Échantillon (f)': round(f, 2),
            'Intervalle de Confiance [95%]': f"[{ic_inf} ; {ic_sup}]"
        })

    # Affichage des résultats
    print(f"\n--- THÉORIE DE L'ESTIMATION (Échantillon n°1, n={n_isole}) ---")
    df_estimation = pd.DataFrame(resultats_estimation)
    print(df_estimation)
else:
    print("Erreur : Le DataFrame 'donnees' n'est pas chargé.")


#  Question 3

import pandas as pd
from scipy.stats import shapiro

# --- Chemins des fichiers (Vérifiez qu'ils pointent bien vers des fichiers différents) ---
chemin_test1 = r"C:\test\Analyse d edonnées\séance 5\src\data\Loi-normale-Test-1.csv"
chemin_test2 = r"C:\test\Analyse d edonnées\séance 5\src\data\Loi-normale-Test-2.csv"

def tester_normalite_corrige(chemin, etiquette):
    try:
        # On force le délimiteur à la virgule pour éviter l'erreur de détection
        df_test = pd.read_csv(chemin, sep=',', engine='python', encoding='utf-8-sig')
        
        # On extrait les données et on retire les valeurs vides (NaN)
        donnees = df_test.iloc[:, 0].dropna()
        
        # Test de Shapiro-Wilk
        stat, p_value = shapiro(donnees)
        
        print(f"\n--- ANALYSE : {etiquette} ---")
        print(f"p-value : {p_value:.5e}")
        
        if p_value > 0.05:
            print("=> Résultat : La distribution semble NORMALE (H0 acceptée).")
        else:
            print("=> Résultat : La distribution n'est PAS NORMALE (H0 rejetée).")
        return p_value
    except Exception as e:
        print(f"Erreur lors de la lecture de {etiquette} : {e}")
        return None

# Exécution des deux tests
p1 = tester_normalite_corrige(chemin_test1, "Test n°1")
p2 = tester_normalite_corrige(chemin_test2, "Test n°2")










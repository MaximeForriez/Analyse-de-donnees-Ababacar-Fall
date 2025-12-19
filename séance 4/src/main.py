#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)




# Question 1 
# les distributions statistiques de variables discrètes suivantes :
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Loi de Dirac (on la simule car elle est très spécifique)
def plot_dirac():
    x = np.arange(-2, 8)
    y = [1 if i == 3 else 0 for i in x] # Masse concentrée en x=3
    plt.stem(x, y)
    plt.title("Loi de Dirac (en x=3)")
    plt.show()

# 2. Loi Uniforme Discrète (randint dans votre liste)
def plot_uniforme():
    low, high = 1, 10
    x = np.arange(low, high + 1)
    y = stats.randint.pmf(x, low, high + 1)
    plt.bar(x, y, color='skyblue')
    plt.title(f"Loi Uniforme Discrète [1, 10]")
    plt.show()

# 3. Loi Binomiale (binom)
def plot_binomiale():
    n, p = 20, 0.4
    x = np.arange(0, n + 1)
    y = stats.binom.pmf(x, n, p)
    plt.bar(x, y, color='green', alpha=0.6)
    plt.title(f"Loi Binomiale (n={n}, p={p})")
    plt.show()

# 4. Loi de Poisson (poisson)
def plot_poisson():
    mu = 5
    x = np.arange(0, 15)
    y = stats.poisson.pmf(x, mu)
    plt.bar(x, y, color='purple', alpha=0.6)
    plt.title(f"Loi de Poisson (lambda={mu})")
    plt.show()

# 5. Loi de Zipf-Mandelbrot (zipfian)
def plot_zipfian():
    a, n = 1.5, 10 # Paramètre de forme et nombre de catégories
    x = np.arange(1, n + 1)
    y = stats.zipfian.pmf(x, a, n)
    plt.bar(x, y, color='orange')
    plt.title(f"Loi de Zipf-Mandelbrot (a={a}, n={n})")
    plt.show()

# Appel des fonctions
plot_dirac()
plot_uniforme()
plot_binomiale()
plot_poisson()
plot_zipfian()


# les distributions statistiques de variables discrètes suivantes :

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Configuration du dossier de sortie
output_dir = "img_seance4_continues"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def finaliser_et_sauvegarder(titre):
    plt.title(titre, fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    nom_fichier = titre.replace(" ", "_").replace("²", "2").lower() + ".png"
    plt.savefig(os.path.join(output_dir, nom_fichier))
    plt.show()

# 1. LOI DE POISSON (Discrète - Rappel)
def visu_poisson(mu=5):
    x = np.arange(0, 20)
    y = stats.poisson.pmf(x, mu)
    plt.figure(figsize=(8, 4))
    plt.bar(x, y, color='purple', alpha=0.5, label=f'Poisson ($\lambda$={mu})')
    finaliser_et_sauvegarder("Loi de Poisson")

# 2. LOI NORMALE
def visu_normale(mu=0, sigma=1):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'r-', lw=2, label=f'Normale ($\mu$={mu}, $\sigma$={sigma})')
    plt.fill_between(x, y, color='red', alpha=0.1)
    finaliser_et_sauvegarder("Loi Normale")

# 3. LOI LOG-NORMALE
def visu_lognormale(s=0.95):
    x = np.linspace(0, 10, 200)
    y = stats.lognorm.pdf(x, s)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'g-', lw=2, label=f'Log-normale (s={s})')
    plt.fill_between(x, y, color='green', alpha=0.1)
    finaliser_et_sauvegarder("Loi Log-normale")

# 4. LOI UNIFORME (Continue)
def visu_uniforme(a=0, b=1):
    x = np.linspace(a - 0.5, b + 0.5, 200)
    y = stats.uniform.pdf(x, loc=a, scale=b-a)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b-', lw=2, label=f'Uniforme [{a}, {b}]')
    plt.fill_between(x, y, color='blue', alpha=0.1)
    finaliser_et_sauvegarder("Loi Uniforme")

# 5. LOI DU CHI-DEUX (χ²)
def visu_chi2(df=5):
    x = np.linspace(0, 20, 200)
    y = stats.chi2.pdf(x, df)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'm-', lw=2, label=f'Chi-deux (ddl={df})')
    plt.fill_between(x, y, color='magenta', alpha=0.1)
    finaliser_et_sauvegarder("Loi du Chi-deux")

# 6. LOI DE PARETO
def visu_pareto(b=2.62):
    x = np.linspace(1, 5, 200)
    y = stats.pareto.pdf(x, b)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'darkorange', lw=2, label=f'Pareto (b={b})')
    plt.fill_between(x, y, color='orange', alpha=0.1)
    finaliser_et_sauvegarder("Loi de Pareto")

# Exécution
if __name__ == "__main__":
    visu_poisson()
    visu_normale()
    visu_lognormale()
    visu_uniforme()
    visu_chi2()
    visu_pareto()

# Question 2
import numpy as np
from scipy import stats

def calculer_moments_discrets():
    print("--- MOMENTS DES DISTRIBUTIONS DISCRÈTES ---")
    
    # 1. Dirac (simulée : moyenne = point de concentration, écart-type = 0)
    print(f"Dirac (x=3)       | Moyenne: 3.00 | Écart-type: 0.00")

    # 2. Uniforme Discrète [1, 10]
    m, v = stats.randint.stats(1, 11, moments='mv')
    print(f"Uniforme Disc.    | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 3. Binomiale (n=20, p=0.4)
    m, v = stats.binom.stats(20, 0.4, moments='mv')
    print(f"Binomiale         | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 4. Poisson (lambda=5)
    m, v = stats.poisson.stats(5, moments='mv')
    print(f"Poisson           | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 5. Zipfian (a=1.5, n=10)
    m, v = stats.zipfian.stats(1.5, 10, moments='mv')
    print(f"Zipf-Mandelbrot   | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}\n")

def calculer_moments_continus():
    print("--- MOMENTS DES DISTRIBUTIONS CONTINUES ---")

    # 1. Normale (mu=0, sigma=1)
    m, v = stats.norm.stats(0, 1, moments='mv')
    print(f"Normale           | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 2. Log-Normale (s=0.95)
    m, v = stats.lognorm.stats(0.95, moments='mv')
    print(f"Log-Normale       | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 3. Uniforme Continue [0, 1]
    m, v = stats.uniform.stats(0, 1, moments='mv')
    print(f"Uniforme Cont.    | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 4. Chi-deux (df=5)
    m, v = stats.chi2.stats(5, moments='mv')
    print(f"Chi-deux          | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

    # 5. Pareto (b=2.62)
    m, v = stats.pareto.stats(2.62, moments='mv')
    print(f"Pareto            | Moyenne: {m:.2f} | Écart-type: {np.sqrt(v):.2f}")

if __name__ == "__main__":
    calculer_moments_discrets()
    calculer_moments_continus()


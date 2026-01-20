# Élements de corrections

## Séance 2.

### Questions

- **Question 8.** Il n'existe aucune hiérarchie entre les caractères.

### Code

- Excellent !

## Séance 3.

### Questions

- Excellent !

### Code

- Excellent !

## Séance 4

### Questions

- Excellent !

### Code

- Excellent !

## Séance 5

### Questions

- Excellent !

### Code

- Excellent !

- Effectivement, le calcul de la *p-value* pose problème dans le cas de la distribution test1. C'est probablement un problème d'échelle. Je réglerai cela pour vos successeurs.

## Séance 6

### Questions

- Excellent !

### Code

- Problème dans le calcul de corrélation des rangs. Vous avez écrit :

```
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
```

Il aurait fallu écrire quelque chose du genre :

```
    # Calcul pour la population
    rho_pop, p_s_pop = spearmanr(rangs_pop_2007, rangs_pop_2025)
    tau_pop, p_k_pop = kendalltau(rangs_pop_2007, rangs_pop_2025)

    print(f"2007 - Spearman : {rho_pop:.4f} (p={p_s_pop:.2e})")
    print(f"2007 - Kendall  : {tau_pop:.4f} (p={p_k_pop:.2e})")

    # Calcul pour la densité
    rho_dens, p_s_dens_ = spearmanr(rangs_dens_2007, rangs_dens_2025)
    tau_dens, p_k_dens = kendalltau(rangs_dens_2007, rangs_dens_2025)

    print(f"2025 - Spearman : {rho_dens:.4f} (p={p_s_dens:.2e})")
    print(f"2025 - Kendall  : {tau_dens:.4f} (p={p_k_dens:.2e})")
```

## Humanités numériques

- Aucun rendu.

## Remarques générales

- Aucun dépôt régulier sur `GitHub`.

- Pourquoi avoir envoyé la séance 2 avec un format `*.zip`. Ce n'est pas gênant, mais cela me force à faire une manipulation qui prend un peu de temps.

- Attention ! Il ne faut pas utiliser les adresses absolues `"C:\test\Analyse d edonnées\séance 2\src\data\resultats-elections-presidentielles-2022-1er-tour.csv"`, mais les adresses relatives `"./data/resultats-elections-presidentielles-2022-1er-tour.csv"`.

- Attention ! `import` ne s'utilise qu'une fois en début de fichier.

- Attention ! En copiant certaines formules, vous avez copié un code `LaTeX`.

- Travail sérieux et excellent !

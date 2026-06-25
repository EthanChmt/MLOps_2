# Rapport d’optimisation post-déploiement

## 1. Objectif

L’objectif de cette analyse est d’évaluer les performances de l’API de prédiction après déploiement, d’identifier les principaux goulots d’étranglement, puis de tester une optimisation permettant d’améliorer le temps de réponse sans dégrader le comportement du modèle.

Les métriques observées sont :

- latence moyenne des requêtes ;
- utilisation CPU ;
- mémoire RAM ;
- utilisation ou non du GPU ;
- profiling du temps d’exécution ;
- compatibilité avec l’environnement de production ;
- absence de régression après optimisation.

## 2. Méthodologie

Les mesures ont été réalisées avec un script de benchmark simulant des appels vers l’endpoint `/predict`.

La latence mesurée correspond au temps de réponse global de l’API. Elle inclut donc la réception du fichier CSV, sa lecture avec Pandas, le feature engineering, l’alignement des colonnes, l’inférence du modèle et la réponse API.

L’utilisation CPU et la mémoire RAM ont également été relevées. Un profiling avec `cProfile` a été utilisé pour identifier les parties du traitement les plus coûteuses.

Deux versions ont été comparées :

- version initiale avec modèle `.pkl` ;
- version optimisée avec modèle ONNX exécuté par ONNX Runtime.

## 3. Goulot d’étranglement identifié

Le principal goulot d’étranglement identifié est l’inférence du modèle au format `.pkl`.

Ce format est pratique pour sauvegarder un pipeline scikit-learn, mais il reste dépendant de l’environnement Python et de la logique d’exécution scikit-learn. Dans cette API, il entraîne un temps de réponse moyen supérieur à 1,5 seconde par requête.

Le profiling et le benchmark montrent que le ralentissement principal ne vient pas d’une saturation CPU ou d’une consommation mémoire élevée, mais du temps nécessaire pour exécuter le modèle au moment de la prédiction.

## 4. Résultats avant / après optimisation

| Métrique | Version `.pkl` | Version ONNX | Évolution |
| --- | ---: | ---: | ---: |
| Latence moyenne | 1,5151 s | 0,0260 s | -98,3 % |
| Utilisation CPU | 4,10 % | 4,30 % | +0,20 point |
| Mémoire RAM | 85,78 MB | 86,04 MB | +0,26 MB |

Le passage du modèle `.pkl` au format ONNX réduit la latence moyenne de **1,5151 s à 0,0260 s**.

Cela représente un gain d’environ **58 fois** et une réduction du temps de réponse d’environ **98,3 %**.

La consommation CPU et la mémoire RAM restent quasiment stables. L’optimisation améliore donc fortement le temps de réponse sans augmenter significativement la consommation de ressources.

## 5. Analyse CPU / GPU / hardware

La configuration finale repose sur une exécution CPU avec ONNX Runtime et `CPUExecutionProvider`.

Le GPU n’est pas utilisé. Ce choix est volontaire : le modèle est suffisamment léger pour être exécuté efficacement sur CPU, et le passage à ONNX Runtime permet déjà d’obtenir une latence moyenne de **0,0260 s**.

L’utilisation d’un GPU aurait complexifié l’environnement de production sans bénéfice nécessaire pour ce cas d’usage.

| Ressource | Choix retenu | Justification |
| --- | --- | --- |
| CPU | Utilisé | Suffisant pour l’API après optimisation ONNX |
| GPU | Non utilisé | Non nécessaire au vu des performances obtenues |
| RAM | Stable | Passage de 85,78 MB à 86,04 MB |

## 6. Validation de non-régression

L’optimisation ne doit pas modifier le comportement fonctionnel du modèle.

Les prédictions du modèle `.pkl` et du modèle ONNX sont comparées sur les mêmes données d’entrée avec une tolérance numérique, par exemple `np.allclose(..., atol=1e-5)`.

Cette vérification permet de s’assurer que l’amélioration du temps de réponse ne se fait pas au détriment de la qualité des prédictions.

## 7. Compatibilité CI/CD

La version optimisée est compatible avec le pipeline CI/CD.

Le modèle ONNX est récupéré depuis Hugging Face au démarrage de l’API. Le déploiement ne dépend donc plus d’un fichier local non versionné.

Les tests existants ont été relancés avant intégration afin de vérifier que les modifications n’introduisent pas de régression dans l’API.

## 8. Limites et améliorations futures

Les résultats doivent être interprétés comme un benchmark simulé. La latence mesurée correspond au temps de réponse global de l’API, et pas uniquement au temps strict d’inférence du modèle.

Les mesures CPU et RAM donnent une indication utile, mais ne remplacent pas un monitoring complet de production avec collecte continue des métriques.

Les améliorations futures identifiées sont :

- intégrer automatiquement la validation `.pkl` / ONNX dans la CI/CD ;
- modulariser le fichier `api/main.py` ;
- séparer les dépendances de production, de test et de profiling ;
- enrichir le monitoring avec des métriques plus détaillées ;
- suivre régulièrement le Data Drift et la stabilité des prédictions.

## 9. Conclusion

L’optimisation principale réalisée est le passage du modèle `.pkl` vers un modèle ONNX exécuté avec ONNX Runtime.

Cette optimisation permet de faire passer la latence moyenne de **1,5151 s** à **0,0260 s**, soit une réduction d’environ **98,3 %** du temps de réponse.

Le principal goulot d’étranglement était donc l’inférence du modèle au format `.pkl`.

La configuration finale repose sur une exécution CPU. Le GPU n’a pas été retenu, car les performances obtenues avec ONNX Runtime sur CPU sont déjà suffisantes pour cette API de scoring.

La version optimisée est plus adaptée à un usage en production, reste compatible avec le pipeline CI/CD et conserve une consommation CPU/RAM stable.
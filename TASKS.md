## Pipeline 

Extraction des features -> (fusion?) -> selection des features -> (fusion?) -> modèle -> fusion 

## Extraction 
- [ ] Penser à prendre les matrices avec des thresholds. Les autres groupes ont trouvé:
  - [ ] ALPHA 20% + ALPHA 100% -> Anova 30 features
- [ ] Graphe 
  - [ ] Moyenne des clusters par région (frontal, etc.) + BETA à 50%
  - [ ] Global efficiency par région
  - [ ] **Complétude de certains groupes avant d'autres sur THETA (quand on monte le threshold, on complète plus tôt sur certains groupes)**
  - [ ] Small wordness avec THETA à 100%

## Selection
- [ ] Essayer de limite le nombre de features à environ 15
- [ ] PCA to keep 90% of the information, KMEANS multi dim, silhouette, ARI more interesting
- [ ] Informations mutuelle au lieu de anova peut être ça va marcher sur les nouvelles méthodes de threshold et fusion
- [ ] **Bruit c'est pas du bruit c'est important -> peut-être pca puis on remove les premières dim et on se concentre sur le bruit**
- [ ] Pas que des méthodes de feature selection univariés -> multivarié 
- [ ] Essayer de trouver une sélection des features qui ne dépend pas du modèle 

## Fusion
- [ ] Il faut des basses fréquences pour chopper sci ad -> fusionner les bandes, et prendre THETA DELTA


## Misc
- [ ] Ne pas dire le mot bruit ni outlyer, c'est juste des points loins de la moyenne, et rien est bruit sauf si spécialiste le dit

## Présentation
- [ ] Multi classe -> show performances class by class 
- [ ] **Tableaux avec ROC AUC (entre 0 et 1), sensitivity (% et quelle classe), specificity (% et quelle classe), accuracy (%)**
- [ ] **Pas de balanced accuracy -> downsampling + acc, roc, sensitivity, specificity**
- [ ] Mettre la pipeline entière au début des slides
- [ ] Attention, PCA ce n'est pas selection c'est dim reduction
- [ ] Mettre **UNIQUEMENT LA PIPELINE FINALE, TOUT TRUC QUI N'A PAS MARCHE NE DOIT PAS ETRE MONTRE** (on met juste ce qui marche quoi, et on justifie rapidement). Par défaut, on a toujours les meilleurs hyperparamètres : pas besoin de le dire, et surtout, pas dire "Y'aurait peut-être mieux avec des meilleurs hyperparamètres". 

## Modèles 
- [ ] SVM non-linéaire
- [ ] Modèles binaires d'abord (entre classe mci classe ad, 2 à 2), puis fusion

## Fusion après
- [ ] Faire gaffe aux confiances -> la scale sera pas la même pour les modèles 


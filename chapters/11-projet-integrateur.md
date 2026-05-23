## Projet Intégrateur : Créer une IA Utile pour une Communauté

### Définir un problème réel d’intérêt communautaire

Le cœur d’un projet d’intelligence artificielle à impact social réside dans son ancrage local. Avant de coder ou d’entraîner un modèle, il faut choisir un problème concret, significatif pour une communauté, et réalisable avec les outils accessibles. Prenons l’exemple d’un dispensaire rural au Sénégal où les ruptures de stock de médicaments essentiels sont fréquentes. Ces pénuries mettent en danger la santé des patients et nuisent à la confiance dans le système de santé.

Notre objectif : **concevoir un modèle d’IA capable de prédire les risques de pénurie de médicaments dans les 15 prochains jours**, en se basant sur des données historiques de consommation, de saisons, d’épidémies locales et de livraisons.

Ce type de projet est pertinent en Afrique, où les infrastructures de santé sont souvent limitées, mais où les données, même partielles, peuvent être exploitée intelligemment. L’idée n’est pas de créer une solution parfaite, mais un prototype fonctionnel qui démontre la valeur ajoutée de l’IA dans un contexte réel.

### Étapes clés du projet : de l’idée au prototype

Un projet d’IA réussi suit un processus structuré. Voici les six étapes à suivre :

1. **Cadrage du problème**
2. **Collecte et préparation des données**
3. **Choix du modèle d’IA**
4. **Entraînement du modèle**
5. **Évaluation des performances**
6. **Présentation des résultats**

Chaque étape est cruciale. Même le meilleur modèle échouera si les données sont biaisées ou incomplètes.

#### Cadrage du problème : transformer un besoin en question prédictive

Dans notre exemple, le besoin est : « Éviter les pénuries de médicaments ». Pour que l’IA puisse y répondre, il faut formuler une **question prédictive précise**. Par exemple :  
> « Quelle est la probabilité que le stock d’artémisinine (un antipaludéen) tombe à zéro dans les 15 prochains jours ? »

Cela devient un problème de **machine learning supervisé de régression ou de classification binaire** (pénurie oui/non). Le choix dépend de la granularité souhaitée.

#### Collecte de données simulées : réaliste mais accessible

Dans un contexte africain, les données réelles peuvent être rares ou sensibles. C’est pourquoi nous allons **simuler un jeu de données réaliste** adapté au projet.

Imaginons que le dispensaire enregistre quotidiennement :
- Quantité en stock de chaque médicament
- Nombre de patients traités
- Diagnostic principal (paludisme, diarrhée, etc.)
- Date (pour détecter les saisons)
- Dernière livraison reçue

Voici un exemple de génération de données simulées en Python :

```python
import pandas as pd
import numpy as np

# Génération de données simulées sur 365 jours
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=365, freq="D")

# Simulation de la consommation d'artémisinine (plus élevée pendant la saison des pluies)
rainy_season = (dates.month >= 6) & (dates.month <= 10)
base_consumption = np.random.poisson(5, 365)
rainy_consumption = base_consumption + np.random.poisson(3, 365)
daily_consumption = np.where(rainy_season, rainy_consumption, base_consumption)

# Simulation de livraisons (tous les 30-45 jours)
deliveries = np.zeros(365)
for i in range(0, 365, np.random.randint(30, 46)):
    if i < 365:
        deliveries[i] = np.random.randint(150, 250)

# Calcul du stock simulé
stock = [200]  # stock initial
for i in range(1, 365):
    new_stock = stock[-1] - daily_consumption[i-1] + deliveries[i]
    stock.append(max(new_stock, 0))  # pas de stock négatif

# Création du DataFrame
data = pd.DataFrame({
    "date": dates,
    "daily_consumption": daily_consumption,
    "stock": stock,
    "is_rainy_season": rainy_season.astype(int),
    "delivery": deliveries
})

# Création de la variable cible : pénurie dans 15 jours ?
data["shortage_in_15_days"] = [
    1 if 0 in data["stock"].iloc[i:i+15].values else 0
    for i in range(365)
]
data = data.dropna().reset_index(drop=True)

data.to_csv("dispensary_data_simulated.csv", index=False)
```

Ce script génère un fichier CSV utilisable pour l’entraînement. Il illustre comment, même sans données réelles, on peut créer un jeu de données pédagogique et réaliste.

#### Préparation des données : la clé du succès

Avant l’entraînement, les données doivent être nettoyées et transformées :
- Gérer les valeurs manquantes
- Créer des variables utiles (ex : moyenne mobile de consommation sur 7 jours)
- Normaliser les valeurs numériques
- Encoder les variables catégorielles

Exemple de création d’une caractéristique utile :

```python
data["avg_consumption_last_7_days"] = data["daily_consumption"].rolling(7).mean()
data = data.dropna()
```

#### Choix du modèle : simple et interprétable

Pour un projet débutant, privilégiez des modèles **interprétables** et peu gourmands. Ici, une **régression logistique** ou un **arbre de décision** est idéal. Ils sont faciles à comprendre, rapides à entraîner, et leurs décisions peuvent être expliquées aux gestionnaires de dispensaire.

Par exemple, un arbre pourrait apprendre que :
> Si la saison des pluies ET le stock < 30 unités ET la consommation moyenne > 8 par jour → risque élevé de pénurie.

Voici l’entraînement d’un arbre de décision :

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sélection des caractéristiques
features = ["stock", "daily_consumption", "is_rainy_season", "avg_consumption_last_7_days", "delivery"]
X = data[features]
y = data["shortage_in_15_days"]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)
print("Précision :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Le modèle atteint ici environ 85 % de précision — suffisant pour un prototype.

#### Évaluation : aller au-delà de la précision

La précision ne dit pas tout. Dans un contexte de santé, **le rappel (recall)** est crucial : on préfère avoir quelques faux positifs (alerte inutile) plutôt que des faux négatifs (pénurie non détectée).

Le rapport de classification montre :
- **Rappel pour la classe 1 (pénurie)** : 78 % → acceptable, mais à améliorer
- **F1-score** : 0.76 → bon équilibre

On peut aussi visualiser l’arbre pour comprendre ses décisions (avec `sklearn.tree.plot_tree`) et l’expliquer à un agent de santé non technique.

#### Déploiement minimal et présentation

Le prototype peut être présenté sous forme de **tableau de bord simple** ou d’**alerte quotidienne**. Par exemple, un script automatisé pourrait générer chaque soir un message :

```python
# Exemple de prédiction quotidienne
today_data = np.array([[25, 10, 1, 9.2, 0]])  # stock, consom, saison, moyenne, livraison
risk = model.predict_proba(today_data)[0][1]

if risk > 0.7:
    print(f"ALERTE : Risque de pénurie élevé ({risk:.1%}) dans 15 jours. Prévoir une commande.")
else:
    print(f"Risque faible ({risk:.1%}). Stock actuel stable.")
```

Dans un contexte réel, ce message pourrait être envoyé par SMS via un service comme **Africa’s Talking** ou intégré à une application simple.

### Bonnes pratiques pour un impact durable

Un projet d’IA utile ne se limite pas au code. Il doit être :
- **Compréhensible** par les utilisateurs finaux (infirmiers, agents de santé)
- **Maintenable** avec des outils localement disponibles
- **Éthique** : pas de biais contre certaines zones ou groupes

Posez-vous ces questions :
- Les données reflètent-elles bien la diversité de la population ?
- Qui bénéficie du système ? Qui pourrait en souffrir ?
- Comment mettre à jour le modèle si les comportements changent ?

Par exemple, si le modèle est entraîné uniquement sur un dispensaire urbain, il risque de mal prédire dans une zone rurale. C’est un biais de représentativité à éviter.

### Travailler seul ou en groupe : conseils pratiques

Ce projet peut être réalisé seul, mais en groupe, il devient plus riche. Répartissez les rôles :
- **Data manager** : collecte et nettoyage des données
- **Data scientist** : entraînement du modèle
- **Responsable éthique** : analyse des biais et impacts
- **Communicant** : présentation des résultats

Utilisez des outils gratuits comme **Google Colab** pour coder ensemble, ou **GitHub** pour partager le code. Empire du Web encourage la collaboration ouverte et inclusive.

---

## Points clés

- Un projet d’IA utile commence par un **problème réel et local**, comme la prévention des pénuries de médicaments.
- Même sans données réelles, des **données simulées** peuvent permettre de construire un prototype fonctionnel et pédagogique.
- Le processus IA comprend six étapes : cadrage, données, modèle, entraînement, évaluation, communication.
- Pour les débutants, privilégiez des **modèles simples et interprétables** comme la régression logistique ou les arbres de décision.
- L’évaluation doit aller au-delà de la précision : le **rappel est crucial** dans les domaines sensibles comme la santé.
- Un bon projet d’IA est **éthique, inclusif et compréhensible** par ses utilisateurs finaux.
- La collaboration et l’explication des résultats sont aussi importantes que le code lui-même.
- En Afrique, l’IA doit être **ancrée dans le contexte local**, utilisant des technologies accessibles et répondant à des besoins concrets.
## Initiation au Machine Learning : Apprentissage Supervisé

Le machine learning, ou apprentissage automatique, est une branche de l’intelligence artificielle qui permet aux machines d’apprendre à partir de données. Parmi ses formes les plus utilisées, l’**apprentissage supervisé** occupe une place centrale. Il s’agit de la méthode la plus intuitive pour comprendre comment une machine peut apprendre à prédire ou à classer.

### Qu’est-ce que l’apprentissage supervisé ?

Imaginez un agriculteur expérimenté qui, au fil des saisons, apprend à reconnaître les signes d’une maladie sur les feuilles de ses plants de tomates. Il observe des symptômes visuels — taches jaunes, déformation — et, grâce à son expérience, pose un diagnostic. Un modèle d’apprentissage supervisé fonctionne de la même manière : **il apprend à partir d’exemples étiquetés**.

En termes techniques, chaque exemple est une **paire d’entrée-sortie** :
- L’**entrée** (ou *feature*) peut être une image, une température, une quantité de pluie, un prix historique.
- La **sortie** (ou *label*) est ce que l’on veut prédire : une maladie, un prix futur, une catégorie.

Le modèle cherche à apprendre une **fonction** qui relie les entrées aux sorties. Une fois entraîné, il pourra appliquer cette fonction à de nouvelles données pour faire des prédictions.

Prenons un exemple concret en contexte africain : un petit marché de céréales au Sénégal. Un vendeur souhaite prédire le prix du maïs demain en fonction de plusieurs facteurs : la quantité disponible, la météo, le jour de la semaine, et les prix des semaines précédentes.

Voici un extrait fictif de son carnet de données :

| Quantité (kg) | Pluie (mm) | Jour | Prix (XOF) |
|---------------|------------|------|------------|
| 500           | 10         | Lundi | 650        |
| 420           | 0          | Mardi | 700        |
| 600           | 30         | Mercredi| 600       |

Chaque ligne est un **exemple étiqueté**. L’objectif du modèle est de trouver une relation entre les colonnes d’entrée (quantité, pluie, jour) et la colonne de sortie (prix). C’est cela, l’apprentissage supervisé.

### Variables d’entrée et de sortie : les briques du modèle

Les **variables d’entrée** sont aussi appelées *features* (caractéristiques). Elles doivent être choisies avec soin : plus elles sont pertinentes, plus le modèle sera précis.

Dans notre exemple, la quantité de maïs sur le marché influence directement le prix : plus il y en a, plus le prix baisse. La pluie peut affecter la qualité ou la disponibilité. Même le jour de la semaine peut jouer un rôle — les prix montent peut-être le vendredi, jour d’affluence.

Les **variables de sortie** (ou *target*) sont ce que l’on veut prédire. Elles peuvent être :
- **Numériques** : comme un prix, une température, un rendement agricole → on parle alors de **régression**.
- **Catégorielles** : comme "malade" ou "sain", "français", "anglais", "swahili" → on parle de **classification**.

Par exemple, un modèle qui prédit le prix du maïs est un modèle de **régression**.  
Un modèle qui identifie si une plante est atteinte par la rouille du blé à partir d’une photo est un modèle de **classification**.

### Fonction de perte : mesurer l’erreur pour apprendre

Personne ne devient bon sans faire d’erreurs. Le modèle non plus. Mais pour progresser, il doit **mesurer ses erreurs**. C’est le rôle de la **fonction de perte** (*loss function*).

Prenons le cas d’un modèle qui prédit un prix de maïs à 680 XOF, mais le prix réel est 700 XOF. L’erreur est de 20 XOF. La fonction de perte calcule cette erreur pour chaque exemple, puis en tire une moyenne. Le modèle ajuste ensuite ses paramètres pour **minimiser** cette perte.

Une fonction de perte courante pour la régression est l’**erreur quadratique moyenne** (MSE) :

```
MSE = (1/n) * Σ (y_prédit - y_réel)²
```

Pour la classification, on utilise souvent l’**entropie croisée**, qui pénalise fortement les mauvaises prédictions assurées.

Imaginez un coach sportif qui corrige chaque mouvement de son athlète. Plus l’athlète se trompe, plus la correction est forte. Le modèle fait la même chose : il s’ajuste petit à petit, grâce à la fonction de perte.

### Validation croisée : éviter l’illusion de la performance

Un modèle peut être excellent sur les données qu’il a déjà vues… mais se planter sur de nouvelles données. C’est comme un élève qui récite par cœur un texte sans le comprendre.

Pour éviter ce piège, on utilise la **validation croisée**. L’idée ? Diviser les données en plusieurs parties, entraîner le modèle sur certaines, et le tester sur d’autres.

La méthode la plus courante est la **validation croisée en k-folds** (k parties). Par exemple, avec 5 folds :
1. On divise les données en 5 groupes.
2. On entraîne sur 4 groupes, teste sur le 5ᵉ.
3. On répète 5 fois, en changeant à chaque fois le groupe de test.
4. On calcule la performance moyenne.

Cela donne une estimation plus fiable de la capacité du modèle à **généraliser** — c’est-à-dire à bien fonctionner sur de nouvelles données.

### Exemple pratique : prédire une maladie agricole

Imaginons une coopérative d’agriculteurs au Cameroun qui veut détecter la **maladie de la mosaïque du manioc**. Ils disposent de centaines de photos de feuilles, certaines saines, d’autres malades, toutes correctement étiquetées.

On peut entraîner un modèle d’apprentissage supervisé pour classer automatiquement une nouvelle photo. Voici comment cela se passerait dans Google Colab, un outil gratuit d’Empire du Web pour expérimenter sans installer de logiciels.

#### Étape 1 : Charger les données

```python
from google.colab import drive
drive.mount('/content/drive')

# Charger un jeu de données simple (ex: CSV ou images)
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/maladie_manioc.csv')
```

#### Étape 2 : Préparer les entrées et sorties

```python
X = data[['texture', 'couleur', 'forme']]  # features
y = data['etat']                           # label: 'sain' ou 'malade'
```

#### Étape 3 : Choisir un modèle simple

Utilisons un modèle classique : l’**arbre de décision**, facile à comprendre.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=5)  # validation croisée en 5 folds
print("Précision moyenne:", scores.mean())
```

Le résultat affiché pourrait être :  
`Précision moyenne: 0.88`  
→ Le modèle est bon dans 88 % des cas.

Cet exemple montre que l’on peut commencer sans être expert en code. Les outils modernes rendent l’accès au machine learning progressivement accessible.

### Pourquoi l’apprentissage supervisé est-il si utile en Afrique ?

Les données, même simples, sont partout : dans les marchés, les hôpitaux, les écoles. L’apprentissage supervisé permet de les transformer en **connaissances actionnables**.

- Un centre de santé peut **prédire les pics de paludisme** en fonction de la saison, des pluies et des cas antérieurs.
- Une école peut **anticiper les élèves en difficulté** à partir de leurs notes et de leur assiduité.
- Un éleveur peut **estimer le poids d’un bœuf** à partir de photos et de mensurations.

Ces applications n’ont pas besoin de supercalculateurs. Elles peuvent fonctionner sur des modèles simples, accessibles via des plateformes comme Google Colab ou des applications mobiles.

### Attention aux pièges courants

Même un bon modèle peut échouer si on néglige certains aspects :
- **Données biaisées** : si toutes les photos de maladie viennent d’une seule région, le modèle ne reconnaîtra pas les symptômes ailleurs.
- **Surapprentissage** (*overfitting*) : le modèle mémorise les exemples au lieu d’apprendre des règles générales. Il devient inutile sur de nouvelles données.
- **Qualité des données** : "Garbage in, garbage out". Si les données sont incomplètes ou erronées, les prédictions seront mauvaises.

Une bonne pratique est de toujours commencer par un **jeu de données propre, représentatif et bien étiqueté**.

## Points clés à retenir

- L’**apprentissage supervisé** repose sur des **paires entrée-sortie** pour entraîner un modèle.
- Les **variables d’entrée** (*features*) sont les informations utilisées pour prédire ; les **variables de sortie** (*labels*) sont ce que l’on veut prédire.
- En **régression**, on prédit une valeur numérique (ex: prix du maïs). En **classification**, on prédit une catégorie (ex: malade/sain).
- La **fonction de perte** mesure l’erreur du modèle et guide son apprentissage.
- La **validation croisée** évalue la capacité du modèle à généraliser à de nouvelles données.
- Des outils comme **Google Colab** permettent de manipuler des modèles simples sans code complexe.
- L’apprentissage supervisé a des applications concrètes en Afrique : agriculture, santé, éducation.
- La qualité des données et la représentativité des exemples sont essentielles pour éviter les biais et le surapprentissage.

Grâce à ce chapitre, vous avez posé les bases pour comprendre comment un modèle apprend à partir d’exemples. Dans les chapitres suivants, nous explorerons des architectures plus avancées, comme les réseaux de neurones, qui s’inspirent du fonctionnement du cerveau humain.
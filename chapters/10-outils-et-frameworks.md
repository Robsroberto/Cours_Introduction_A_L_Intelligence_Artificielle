## Outils Pratiques : Découverte des Frameworks d'IA

### Les frameworks d’IA : des moteurs invisibles de l’innovation

Les frameworks d’intelligence artificielle sont des bibliothèques logicielles qui simplifient la conception, l’entraînement et le déploiement de modèles d’IA. Ils agissent comme des outils puissants qui permettent aux développeurs de se concentrer sur la logique de leurs algorithmes, sans avoir à tout réinventer à chaque projet. En Afrique, où les ressources techniques et matérielles peuvent être limitées, choisir le bon framework est une étape cruciale pour réussir un projet d’IA accessible, efficace et évolutif.

Trois frameworks dominent aujourd’hui le paysage de l’IA : **TensorFlow**, **PyTorch** et **Scikit-learn**. Chacun a ses forces, ses cas d’usage privilégiés, et un écosystème qui les entoure. Comprendre leurs spécificités permet de faire des choix éclairés, surtout lorsqu’on débute.

---

### TensorFlow : l’architecte industriel

Développé par Google, **TensorFlow** est un framework robuste, particulièrement adapté aux applications industrielles et à grande échelle. Il excelle dans le déploiement de modèles sur mobile, web ou serveurs.

Un exemple pertinent pour les développeurs africains : imaginez un système qui reconnaît les maladies des plantes à partir de photos prises avec un téléphone. TensorFlow Lite, une version allégée du framework, permet de déployer ce modèle directement sur des smartphones, même sans connexion internet constante — une solution idéale pour les zones rurales.

Voici un extrait simple pour charger un modèle prédéfini dans TensorFlow :

```python
import tensorflow as tf

# Charger un modèle pré-entraîné
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Prédire sur une image (prétraitée)
predictions = model.predict(image_array)
```

TensorFlow est puissant, mais son apprentissage peut sembler plus rigide, surtout pour les débutants. Son avantage majeur ? Une excellente documentation, une communauté mondiale active, et un support fort sur les plateformes cloud comme Google Cloud.

---

### PyTorch : le laboratoire du chercheur

Créé par Facebook (Meta), **PyTorch** est devenu le favori des chercheurs et des étudiants. Il est plus flexible et intuitif, avec une syntaxe proche de Python pur. Cela le rend idéal pour expérimenter rapidement, par exemple lors d’un projet scolaire ou d’un prototype dans un incubateur technologique local.

Supposons que vous souhaitiez créer un modèle pour classer différentes variétés de manioc selon leurs racines. Avec PyTorch, vous pouvez modifier facilement chaque couche du réseau de neurones, ajuster les paramètres d’entraînement, et visualiser les résultats en temps réel.

Exemple de création d’un simple réseau de neurones :

```python
import torch
import torch.nn as nn

class ClassifieurManioc(nn.Module):
    def __init__(self):
        super(ClassifieurManioc, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 classes : manioc blanc, jaune, rouge

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ClassifieurManioc()
```

PyTorch est parfait pour apprendre, mais peut exiger plus de ressources pour le déploiement industriel. Il gagne toutefois en popularité grâce à TorchScript et TorchServe, qui facilitent la mise en production.

---

### Scikit-learn : l’ami du débutant

Si vous débutez, **Scikit-learn** est votre meilleur allié. Ce framework ne s’occupe pas des réseaux de neurones complexes, mais il maîtrise parfaitement les algorithmes classiques de machine learning : régression, classification, clustering.

Par exemple, vous pouvez utiliser Scikit-learn pour prédire les rendements agricoles en fonction de données simples : pluviométrie, type de sol, variété de semence. Il fonctionne très bien avec des données structurées — celles que l’on trouve souvent dans les tableurs Excel ou les bases de données locales.

Voici comment entraîner un classifieur d’espèces de mil avec Scikit-learn :

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Données : longueur et largeur des épis
X = [[10, 2], [12, 3], [8, 1], [15, 4]]  # caractéristiques
y = [0, 1, 0, 1]  # 0 = mil rouge, 1 = mil blanc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Précision :", accuracy_score(y_test, predictions))
```

Simple, rapide, et efficace. Scikit-learn est le socle idéal pour comprendre les bases du machine learning avant de passer à des modèles plus complexes.

---

### Plateformes cloud : accéder à la puissance sans matériel coûteux

En Afrique, avoir un ordinateur puissant n’est pas toujours possible. La solution ? Utiliser des plateformes cloud gratuites qui offrent accès à des GPU (processeurs graphiques) nécessaires pour entraîner des modèles d’IA.

**Google Colab** est une des meilleures options. Il s’agit d’un environnement de programmation en ligne, basé sur Jupyter Notebook, qui fonctionne directement dans le navigateur. Pas besoin d’installer quoi que ce soit. Et surtout, il propose une version gratuite avec accès à GPU et TPU (un processeur spécial pour l’IA), idéal pour les zones à faible bande passante.

Voici comment activer un GPU dans Google Colab :

1. Ouvrir un nouveau notebook sur [colab.research.google.com](https://colab.research.google.com)
2. Aller dans **Modifier > Préférences de notebook > Accélérateur matériel > GPU**
3. Exécuter une cellule avec :

```python
import tensorflow as tf
print("GPU Disponible:", tf.config.list_physical_devices('GPU'))
```

Google Colab permet de lancer des projets d’IA même avec un vieux smartphone ou un ordinateur bas de gamme. C’est une révolution pour les développeurs africains qui veulent apprendre sans investir des milliers de francs CFA dans du matériel.

---

### Projet tutoriel : classifieur d’images de produits locaux

Vous allez maintenant créer un classifieur d’images capable de reconnaître trois produits agricoles courants : **manioc**, **mil** et **arachide**. Ce projet est conçu pour être réalisable avec des ressources minimales.

#### Étape 1 : Collecte des données

Prenez 20 à 30 photos de chaque produit avec un téléphone. Vous pouvez les prendre sur un marché local, dans un champ ou à la maison. Nommez les dossiers :
- `mancioc/`
- `mil/`
- `arachide/`

Chargez-les sur Google Drive, puis montez le Drive dans Colab :

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Étape 2 : Chargement et préparation

Utilisez TensorFlow/Keras pour charger les images :

```python
import tensorflow as tf

data_dir = '/content/drive/MyDrive/produits_locaux'
batch_size = 16
img_size = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=img_size,
  batch_size=batch_size)
```

#### Étape 3 : Création du modèle

On utilise un modèle pré-entraîné (MobileNetV2), adapté aux petits écrans et faibles ressources :

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

#### Étape 4 : Entraînement

```python
model.fit(train_ds, epochs=5)
```

#### Étape 5 : Test et interface simple

Après entraînement, testez le modèle avec une nouvelle photo. Vous pouvez même créer une interface basique avec `gr.Interface` (bibliothèque Gradio) pour que d’autres utilisateurs puissent tester :

```python
import gradio as gr

def predire_produit(img):
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    prediction = model.predict(img[None, ...])
    classes = ['Manioc', 'Mil', 'Arachide']
    return {classes[i]: float(prediction[0][i]) for i in range(3)}

interface = gr.Interface(fn=predire_produit, inputs="image", outputs="label")
interface.launch()
```

En quelques clics, vous avez créé une application web interactive, accessible depuis un téléphone.

---

## Points clés à retenir

- **TensorFlow** est idéal pour le déploiement, surtout sur mobile ou web. Il est robuste et bien documenté, mais peut être plus rigide pour les débutants.
- **PyTorch** est flexible et parfait pour apprendre, expérimenter et développer des prototypes. Sa syntaxe proche de Python en fait un excellent choix pédagogique.
- **Scikit-learn** est incontournable pour les algorithmes classiques de machine learning. Il est simple, rapide, et parfait pour traiter des données structurées comme celles de l’agriculture ou de la santé.
- **Google Colab** permet d’accéder à des GPU gratuits sans installer de logiciels. C’est une ressource précieuse pour les développeurs africains avec peu de matériel.
- Vous pouvez créer un classifieur d’images avec des données locales (manioc, mil, arachide) en quelques étapes simples : collecte, entraînement avec un modèle pré-entraîné, et déploiement via une interface.
- L’IA n’a pas besoin de matériel coûteux pour être utile. Avec les bons outils, un smartphone et une connexion occasionnelle suffisent pour démarrer.

Chez Empire du Web, nous croyons que l’avenir de l’IA en Afrique se construit avec des outils accessibles, des projets concrets et une communauté engagée. Ce chapitre vous a donné les clés pour passer de la théorie à l’action — maintenant, à vous de créer.
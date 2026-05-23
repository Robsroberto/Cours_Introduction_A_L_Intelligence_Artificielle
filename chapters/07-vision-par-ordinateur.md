## Vision par Ordinateur : L'IA qui Voit

La vision par ordinateur (Computer Vision) est une branche de l’intelligence artificielle qui permet aux machines de “voir” et d’interpréter le monde visuel comme le font les humains. Grâce à elle, un ordinateur peut analyser une image, détecter des objets, reconnaître des visages, ou encore comprendre une scène complexe. Ce domaine est aujourd’hui au cœur de nombreuses innovations, des voitures autonomes aux applications agricoles passant par la santé numérique.

Contrairement à ce que l’on pourrait croire, la vision par ordinateur ne nécessite pas obligatoirement du matériel coûteux ou des supercalculateurs. Grâce à des modèles pré-entraînés et des frameworks accessibles, même un développeur africain travaillant depuis un cybercafé avec un ordinateur modeste peut commencer à expérimenter cette technologie.

### Comment une machine "voit-elle" une image ?

Pour un humain, reconnaître un chat sur une photo est instantané. Pour une machine, une image n’est qu’un tableau de nombres. Chaque pixel est représenté par des valeurs numériques correspondant à ses couleurs (rouge, vert, bleu – le format RGB). Une image de 224x224 pixels contient donc 224 × 224 × 3 = 150 528 nombres.

Le défi de la vision par ordinateur est de transformer ce tableau de nombres en une compréhension sémantique : *qu’y a-t-il dans l’image ? Où sont les objets ? Quelle est la scène ?*

Les premières approches de la vision par ordinateur reposaient sur des algorithmes manuels. Par exemple, pour détecter les contours dans une image, on appliquait des filtres mathématiques comme le filtre de Sobel ou de Canny. Ces méthodes analysent les changements brusques d’intensité entre pixels voisins — un bon indicateur d’un bord.

```python
# Exemple de détection de contours avec OpenCV (Python)
import cv2
import numpy as np

# Charger une image en niveaux de gris
image = cv2.imread('champ_mais.jpg', cv2.IMREAD_GRAYSCALE)

# Appliquer le détecteur de contours de Canny
contours = cv2.Canny(image, 100, 200)

# Afficher le résultat
cv2.imshow('Contours', contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Ce code permet de visualiser les contours dans une image de champ agricole. Même si cela ne suffit pas à reconnaître une maladie de maïs, c’est une première étape vers la compréhension visuelle.

### De la détection à la reconnaissance : entrée dans le deep learning

Les méthodes traditionnelles ont leurs limites. Elles ne peuvent pas facilement reconnaître un objet sous différents angles, éclairages ou occlusions. C’est là que le deep learning entre en jeu, notamment avec les réseaux de neurones convolutifs (CNN).

Un CNN apprend automatiquement les caractéristiques pertinentes à partir de milliers d’images étiquetées. Il commence par détecter des formes simples (lignes, courbes), puis des formes plus complexes (yeux, roues), et enfin des objets entiers (visage, voiture). Ce processus d’apprentissage hiérarchique est ce qui rend les CNN si puissants.

### Reconnaissance d’objets et classification d’images

La classification d’images consiste à attribuer une étiquette à une image entière. Par exemple : *cette photo contient-elle une tomate saine ou malade ?* La détection d’objets va plus loin : elle localise chaque objet dans l’image (avec un rectangle) et indique ce qu’il est.

Imaginons un agriculteur au Sénégal qui prend une photo de sa parcelle de tomates avec son smartphone. Grâce à une application mobile utilisant la vision par ordinateur, il peut instantanément savoir si ses plants sont attaqués par un champignon. Cette détection précoce permet d’intervenir rapidement, réduisant les pertes et préservant l’environnement grâce à une utilisation ciblée de traitements.

Un autre exemple africain pertinent est l’utilisation de drones équipés de caméras pour surveiller des cultures de cacao en Côte d’Ivoire. Les images aériennes sont analysées pour détecter les zones de stress hydrique ou les infestations d’insectes, permettant une gestion précise des ressources.

### MobileNet : un modèle adapté aux réalités africaines

Dans de nombreux pays africains, l’accès à des serveurs cloud puissants ou à une connexion internet haut débit reste limité. C’est pourquoi des modèles légers comme **MobileNet** sont particulièrement adaptés.

Développé par Google, MobileNet est conçu pour fonctionner efficacement sur des appareils mobiles et embarqués. Il utilise une architecture appelée *convolutions séparables par profondeur* qui réduit considérablement le nombre de calculs nécessaires, sans sacrifier trop de précision.

MobileNet a été pré-entraîné sur ImageNet, une base de données de 14 millions d’images réparties en 1000 catégories (chiens, voitures, fruits, etc.). Cela signifie qu’on peut l’utiliser immédiatement pour reconnaître des objets courants.

Voici un exemple d’application simple avec TensorFlow Lite (la version légère de TensorFlow) :

```python
# Chargement et inférence avec MobileNetV2 (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger le modèle pré-entraîné
model = MobileNetV2(weights='imagenet')

# Charger une image depuis le smartphone
img_path = 'mais_malade.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Faire une prédiction
predictions = model.predict(x)
decoded = decode_predictions(predictions, top=3)[0]

# Afficher les résultats
for classe, description, score in decoded:
    print(f"{description}: {score:.2f}")
```

Supposons que cette photo montre une feuille de maïs attaquée par la rouille. Le modèle pourrait afficher :

```
corn: 0.68
plant: 0.19
leaf: 0.07
```

Même si le modèle ne reconnaît pas directement la maladie, il identifie correctement la présence de maïs. C’est un point de départ. En affinant le modèle avec des images locales de maladies de cultures africaines, on peut atteindre une précision bien supérieure.

### Adapter la vision par ordinateur aux besoins locaux

La puissance de la vision par ordinateur ne réside pas dans l’usage de modèles globaux, mais dans leur adaptation locale.

Par exemple, au Kenya, des startups comme **UjuziKilimo** utilisent la vision par ordinateur pour analyser les sols et les cultures à partir de photos. En Tanzanie, des projets pilotes permettent aux petits agriculteurs d’envoyer des photos de leurs plants via WhatsApp, et un système d’IA répond automatiquement avec un diagnostic et des conseils.

Ces solutions fonctionnent souvent en combinant :

- Des modèles pré-entraînés (comme MobileNet)
- Des jeux de données locales (photos de cultures africaines prises avec des smartphones bas de gamme)
- Une interface simple (SMS, WhatsApp, app légère)

Un développeur à Abidjan ou à Dakar peut donc :

1. Collecter des images de plants malades via une campagne communautaire.
2. Retravailler un modèle MobileNet avec ces nouvelles données (c’est le *transfer learning*).
3. Déployer l’application sur un serveur local ou une app Android.

Ce genre de projet peut être réalisé avec un budget minimal, surtout grâce aux outils open source et aux ressources gratuites comme Google Colab.

### Limites et précautions

La vision par ordinateur n’est pas infaillible. Les erreurs peuvent survenir à cause :

- De la mauvaise qualité des images (flou, mauvais éclairage)
- D’un manque de données locales (le modèle n’a jamais vu un type de maladie de manioc du Congo)
- De biais dans les données d’entraînement (trop d’images prises en Europe, peu en Afrique)

Il est donc essentiel de tester les modèles dans des conditions réelles, avec des utilisateurs finaux. Une erreur de diagnostic peut coûter cher à un agriculteur.

De plus, la confidentialité des données est cruciale. Les photos envoyées par les agriculteurs contiennent souvent des informations sensibles (localisation du champ, type de culture). Il faut s’assurer qu’elles sont traitées de manière sécurisée et éthique.

---

## Points clés

- La vision par ordinateur permet aux machines d’interpréter des images, grâce à des techniques allant de la détection de contours au deep learning.
- Les réseaux de neurones convolutifs (CNN) sont au cœur des avancées récentes, car ils apprennent automatiquement à reconnaître des objets à partir d’images.
- Des modèles légers comme **MobileNet** sont idéaux pour les contextes africains, car ils fonctionnent sur des appareils peu puissants et nécessitent peu de bande passante.
- La classification d’images et la détection d’objets ont des applications concrètes en Afrique, notamment dans l’agriculture (diagnostic de maladies de cultures) et la santé (analyse de radiographies).
- Le **transfer learning** permet d’adapter des modèles pré-entraînés à des besoins locaux, avec peu de données et de ressources.
- Des outils comme TensorFlow Lite et OpenCV rendent la mise en œuvre accessible, même à distance avec un ordinateur basique.
- Les défis restent réels : qualité des données, biais, accessibilité, éthique. L’implication des communautés locales est essentielle pour réussir.
- Un développeur africain peut dès aujourd’hui créer des solutions d’IA utiles, pertinentes et durables, simplement en combinant un smartphone, une connexion internet limitée et des compétences en programmation.
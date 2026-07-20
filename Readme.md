# 🔢 Reconnaissance de Chiffres Manuscrits

Projet de stage **Machine Learning** chez **CodeAlpha** — Reconnaissance de chiffres manuscrits avec Deep Learning.

---

## 📋 Description

Ce projet implémente un modèle de **Deep Learning (CNN)** capable de reconnaître les chiffres manuscrits (0-9) à partir d'images. Le modèle est entraîné sur le dataset **MNIST** et atteint une accuracy de **99.50%**.

### 🎯 Objectifs

- Prétraiter et normaliser les images de chiffres manuscrits
- Construire un réseau de neurones convolutif (CNN)
- Entraîner et évaluer le modèle
- Prédire le chiffre dans une image personnalisée

---

## 🛠️ Technologies utilisées

| Technologie | Version |
|-------------|---------|
| Python | 3.11 |
| TensorFlow | 2.x |
| Keras | Intégré à TensorFlow |
| NumPy | 1.x |
| Matplotlib | 3.x |
| OpenCV | 4.x |
| Scikit-learn | 1.x |
| Streamlit | 1.x |

---

## 📁 Structure du projet
CodeAlpha_Handwritten/
├── data/                          # Images et visualisations
│   ├── test_image.png             # Image de test personnalisée
│   ├── training_history.png       # Courbes d'entraînement
│   ├── predictions.png            # Exemples de prédictions
│   ├── confusion_matrix.png       # Matrice de confusion
│   └── errors.png                 # Images mal prédites
├── models/
│   ├── best_model.h5              # Modèle CNN entraîné (99.5%)
│   └── rf_model.pkl               # Modèle Random Forest léger
├── src/
│   ├── data_preprocessing.py      # Chargement et prétraitement MNIST
│   ├── model.py                   # Architecture CNN
│   ├── train.py                   # Script d'entraînement CNN
│   ├── predict.py                 # Prédiction sur image personnalisée
│   ├── visualize.py               # Visualisation des résultats
│   └── train_light_model.py       # Entraînement modèle léger
├── app.py                         # Application Streamlit interactive
├── requirements.txt               # Dépendances
└── README.md                      # Ce fichier


---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Neriah249-alt/CodeAlpha_Handwritten.git
cd CodeAlpha_Handwritten

2. Créer l'environnement virtuel

bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

3. Installer les dépendances

bash
pip install -r requirements.txt

🏋️ Entraînement du modèle CNN
bash
cd src
python train.py

Résultat obtenu :
Epochs : 22/30 (EarlyStopping)
Accuracy test : 99.50%
Modèle sauvegardé dans models/best_model.h5

🔮 Prédiction sur une image
1. Préparer une image
Créer une image 28×28 pixels
Fond noir, chiffre blanc
📊 Visualisation des résultats
bash
cd src
python visualize.py

Génère :
✅ Courbes d'entraînement (accuracy/loss)
✅ Prédictions sur 16 images
✅ Matrice de confusion
✅ Images mal prédites

🌐 Application interactive (Streamlit)
bash
streamlit run app.py

Puis ouvrir : http://localhost:8501
Les utilisateurs peuvent :
Uploader une image de chiffre manuscrit
Voir la prédiction en temps réel
Consulter les probabilités pour chaque chiffre

🧠 Architecture du modèle CNN
Couche                  | Sortie
------------------------|------------------
Conv2D (32 filtres)     | (28, 28, 32)
BatchNormalization      | (28, 28, 32)
Conv2D (32 filtres)     | (28, 28, 32)
MaxPooling2D            | (14, 14, 32)
Dropout (25%)           | (14, 14, 32)

Conv2D (64 filtres)     | (14, 14, 64)
BatchNormalization      | (14, 14, 64)
Conv2D (64 filtres)     | (14, 14, 64)
MaxPooling2D            | (7, 7, 64)
Dropout (25%)           | (7, 7, 64)

Flatten                 | (3136)
Dense (256 neurones)    | (256)
BatchNormalization      | (256)
Dropout (50%)           | (256)
Dense (10 classes)      | (10) — Softmax

Paramètres totaux : ~870 000

📈 Résultats
| Métrique | Valeur                             |
| -------- | ---------------------------------- |
| Accuracy | **99.50%**                         |
| Loss     | ~0.02                              |
| Epochs   | 22 (EarlyStopping)                 |
| Dataset  | MNIST (60 000 train / 10 000 test) |

## ⚠️ Limitations connues

Le modèle fonctionne correctement dans la plupart des cas, mais peut présenter des **limitations** :

| Situation | Performance | Cause |
|-----------|-------------|-------|
| Chiffre bien centré, trait épais | ✅ Excellent (95%+) | Condition idéale |
| Chiffre décentré ou incliné | ⚠️ Moyen (60-80%) | Prétraitement à améliorer |
| Trait fin ou mal formé | ❌ Faible (<50%) | Différent de MNIST |
| Fond complexe ou bruité | ❌ Faible | Pas de débruitage avancé |

### Améliorations possibles

- 🔄 Utiliser l'augmentation de données (Data Augmentation)
- 📊 Entraîner sur plus d'époques
- 🎯 Ajouter un prétraitement plus robuste
- 🖼️ Utiliser un modèle pré-entraîné (Transfer Learning)



🌐 Démo en ligne
🔗 Page de présentation : https://huggingface.co/spaces/neriah249/codealpha-handwritten
> ⚠️ **Important** : La démo en ligne est un **dashboard de présentation** qui affiche :
> - Les performances du modèle (accuracy, epochs)
> - L'architecture du réseau de neurones
> - Des exemples de résultats

**Pour tester l'application avec vos propres images**, vous devez exécuter l'application **en local** :

```bash
streamlit run app.py

🔧 Améliorations possibles
[ ] Passer à EMNIST pour reconnaître les lettres (A-Z)
[ ] Implémenter un CRNN pour reconnaître des mots complets
[ ] Ajouter une interface web avancée
[ ] Déployer le modèle sur le cloud avec API

📝 Auteur
Nom : OLAFA Maurica Nériah Mondjissiola
Stage : CodeAlpha Machine Learning Internship
LinkedIn : Mauricia Olafa
Date : 20 Juillet 2026
🙏 Remerciements
Merci à CodeAlpha pour cette opportunité de stage et l'accompagnement tout au long du projet.

<div align="center">
  <p>🎓 Projet réalisé dans le cadre du stage Machine Learning — CodeAlpha</p>
</div>




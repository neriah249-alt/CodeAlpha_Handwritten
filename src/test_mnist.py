import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import os
import sys

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
# Obtenir le chemin absolu du dossier courant
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = '../models/best_model.h5'

print(f"🔍 Recherche du modèle dans : {model_path}")

# Vérifier si le fichier existe
if not os.path.exists(model_path):
    print(f"❌ ERREUR : Le fichier {model_path} n'existe pas !")
    print("📁 Fichiers disponibles dans 'models/' :")
    try:
        print(os.listdir(os.path.join(current_dir, 'models')))
    except:
        print("   Dossier 'models/' introuvable")
    sys.exit(1)

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
try:
    # Essayer de charger le modèle
    model = load_model(model_path)
    print(f"✅ Modèle chargé avec succès")
    print(f"📊 Architecture : {model.summary()}")
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    sys.exit(1)

# ==========================================
# 3. CHARGEMENT DE MNIST
# ==========================================
print("\n📦 Chargement de MNIST...")
(_, _), (x_test, y_test) = mnist.load_data()
print(f"✅ Dataset chargé : {len(x_test)} images de test")

# ==========================================
# 4. ÉVALUATION SUR 10 IMAGES
# ==========================================
print("\n" + "="*50)
print("🔬 TESTS SUR 10 IMAGES MNIST")
print("="*50)

correct = 0
total = 10

for i in range(total):
    # Image originale
    img = x_test[i]
    true_label = y_test[i]
    
    # Afficher l'image en ASCII (optionnel)
    if i < 5:  # Afficher les 5 premières en ASCII
        print(f"\nImage {i} (ASCII) :")
        for row in img[:10, :10]:  # Afficher seulement 10x10 pixels
            print(''.join(['█' if p > 128 else '░' for p in row[:10]]))
    
    # ==========================================
    # PRÉTRAITEMENT IDENTIQUE À L'ENTRAÎNEMENT
    # ==========================================
    # 1. Normalisation [0,1]
    img_processed = img.astype('float32') / 255.0
    
    # 2. Ajout des dimensions (batch_size, height, width, channels)
    img_processed = np.expand_dims(img_processed, axis=0)  # batch
    img_processed = np.expand_dims(img_processed, axis=-1) # channel
    
    # 3. Vérification des dimensions
    if img_processed.shape != (1, 28, 28, 1):
        print(f"❌ Format invalide : {img_processed.shape}")
        print(f"   Attendu : (1, 28, 28, 1)")
        continue
    
    # ==========================================
    # PRÉDICTION
    # ==========================================
    try:
        pred = model.predict(img_processed, verbose=0)
        predicted = np.argmax(pred[0])
        confidence = np.max(pred[0])
        
        # Vérifier si la prédiction est correcte
        is_correct = (predicted == true_label)
        if is_correct:
            correct += 1
        
        # Afficher le résultat
        status = "✅" if is_correct else "❌"
        print(f"{status} Image {i+1:2d}: Vrai={true_label} | Prédit={predicted} | Confiance={confidence:.2%}")
        
        # Afficher les top 3 prédictions
        top3_idx = np.argsort(pred[0])[-3:][::-1]
        top3_probs = [pred[0][idx] for idx in top3_idx]
        print(f"   Top 3 : {top3_idx} → {[f'{p:.2%}' for p in top3_probs]}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {e}")

# ==========================================
# 5. STATISTIQUES
# ==========================================
print("\n" + "="*50)
print("📊 STATISTIQUES")
print("="*50)

accuracy = correct / total
print(f"✅ Précision sur {total} images : {accuracy:.2%}")

if accuracy == 1.0:
    print("\n🎉 PERFECT ! Le modèle fonctionne parfaitement sur MNIST !")
elif accuracy >= 0.8:
    print("\n👍 Bonne performance ! Le modèle est correct.")
elif accuracy >= 0.5:
    print("\n⚠️ Performance moyenne. Le modèle a besoin d'être amélioré.")
else:
    print("\n❌ Mauvaise performance. Le modèle est probablement corrompu ou mal entraîné.")

# ==========================================
# 6. TEST SUR UNE IMAGE PERSONNALISÉE
# ==========================================
print("\n" + "="*50)
print("🖼️ TEST SUR UNE IMAGE DU DOSSIER 'data/'")
print("="*50)

# Chercher une image dans le dossier data
data_dir = os.path.join(current_dir, 'data')
if os.path.exists(data_dir):
    image_files = [f for f in os.listdir(data_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if image_files:
        from PIL import Image
        
        for img_file in image_files[:2]:  # Tester les 2 premières images
            img_path = os.path.join(data_dir, img_file)
            print(f"\n📷 Test sur : {img_file}")
            
            try:
                # Charger l'image
                img = Image.open(img_path)
                
                # Convertir en niveaux de gris
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Redimensionner à 28x28
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convertir en array et normaliser
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Inverser si nécessaire (MNIST = fond noir, chiffre blanc)
                if np.mean(img_array) > 0.5:
                    img_array = 1.0 - img_array
                
                # Reshape pour le modèle
                img_array = np.expand_dims(img_array, axis=(0, -1))
                
                # Prédire
                pred = model.predict(img_array, verbose=0)
                predicted = np.argmax(pred[0])
                confidence = np.max(pred[0])
                
                print(f"   Prédit : {predicted}")
                print(f"   Confiance : {confidence:.2%}")
                
            except Exception as e:
                print(f"   ❌ Erreur : {e}")
    else:
        print("ℹ️ Aucune image trouvée dans 'data/'")
else:
    print("ℹ️ Dossier 'data/' introuvable")

# ==========================================
# 7. DIAGNOSTIC DU PROBLÈME
# ==========================================
if accuracy < 0.8:
    print("\n" + "="*50)
    print("🔧 DIAGNOSTIC - Le modèle ne reconnaît pas les chiffres")
    print("="*50)
    print("""
    Causes possibles :

    1️⃣ **Le modèle n'est pas le bon**
       → Vérifiez que 'models/best_model.h5' est bien le modèle entraîné sur MNIST
       → Vérifiez sa taille (doit faire ~1-2 MB)
    
    2️⃣ **Le modèle n'a pas été entraîné correctement**
       → Re-entraînez le modèle avec 'train_mnist.py'
       → Assurez-vous d'avoir une précision > 95%
    
    3️⃣ **Problème de prétraitement**
       → Vérifiez que vous utilisez le même prétraitement qu'à l'entraînement
       → Normalisation [0,1], reshape (28,28,1)
    
    4️⃣ **Le fichier est corrompu**
       → Vérifiez la taille du fichier
       → Essayez de le recharger ou de ré-entraîner le modèle
    
    5️⃣ **Problème de compatibilité TensorFlow**
       → Vérifiez votre version : tf.__version__
       → Essayez : pip install --upgrade tensorflow
    """)

print("\n" + "="*50)
print("🏁 Fin du test")
print("="*50)
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import sys

print("✅ Script démarré", file=sys.stderr)

def preprocess_image(image_path):
    print(f"🔍 Chargement de : {image_path}", file=sys.stderr)
    
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        print(f"❌ FICHIER NON TROUVÉ : {image_path}", file=sys.stderr)
        print(f"📁 Dossier actuel : {os.getcwd()}", file=sys.stderr)
        print(f"📁 Contenu de ../data/ :", file=sys.stderr)
        try:
            for f in os.listdir('../data/'):
                print(f"   - {f}", file=sys.stderr)
        except:
            print("   (impossible de lister)", file=sys.stderr)
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    
    # Lire l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"📊 Dimensions image : {img.shape if img is not None else 'None'}", file=sys.stderr)
    
    if img is None:
        raise ValueError("Impossible de lire l'image")
    
    # Redimensionner
    img = cv2.resize(img, (28, 28))
    print(f"📊 Après resize : {img.shape}", file=sys.stderr)
    
    # Inverser les couleurs
    img = 255 - img
    print(f"📊 Valeurs min/max après inversion : {img.min()}/{img.max()}", file=sys.stderr)
    
    # Normaliser
    img = img.astype('float32') / 255.0
    
    # Ajouter dimensions
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    print(f"📊 Shape finale : {img.shape}", file=sys.stderr)
    
    return img

def predict(image_path):
    print("🔍 Chargement du modèle...", file=sys.stderr)
    model = load_model('../models/best_model.h5')
    print("✅ Modèle chargé", file=sys.stderr)
    
    img = preprocess_image(image_path)
    
    print("🔍 Prédiction en cours...", file=sys.stderr)
    prediction = model.predict(img, verbose=0)
    print(f"✅ Prédiction faite : {prediction[0]}", file=sys.stderr)
    
    predicted_digit = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    # Afficher le résultat
    print("\n" + "="*50)
    print("🎯 RÉSULTAT DE LA PRÉDICTION")
    print("="*50)
    print(f"Chiffre prédit : {predicted_digit}")
    print(f"Confiance : {confidence:.2%}")
    print("="*50)
    
    print("\n📊 Probabilités par chiffre :")
    for i, prob in enumerate(prediction[0]):
        bar = "█" * int(prob * 50)
        print(f"   {i} : {prob:.4f} {bar}")
    
    return predicted_digit, confidence

if __name__ == "__main__":
    image_path = '../data/test_image.png'
    try:
        digit, conf = predict(image_path)
    except Exception as e:
        print(f"\n❌ ERREUR : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
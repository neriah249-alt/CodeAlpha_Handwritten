import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Télécharger MNIST manuellement si besoin
def load_mnist():
    # Essayer de charger depuis le cache Keras
    cache_dir = os.path.expanduser('~/.keras/datasets')
    mnist_file = os.path.join(cache_dir, 'mnist.npz')
    
    if os.path.exists(mnist_file):
        with np.load(mnist_file, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    else:
        print("❌ Dataset MNIST non trouvé.")
        print(f"📁 Télécharge-le ici : https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
        print(f"📁 Place-le dans : {cache_dir}")
        raise FileNotFoundError("MNIST manquant")

# Charger MNIST
print("📦 Chargement de MNIST...")
(x_train, y_train), (x_test, y_test) = load_mnist()

# Aplatir les images (28x28 → 784)
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Entraîner un Random Forest (léger et rapide)
print("🌲 Entraînement du modèle Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
model.fit(x_train[:10000], y_train[:10000])  # 10K images suffisent

# Évaluer
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
print(f"✅ Accuracy : {acc:.4f} ({acc*100:.2f}%)")

# Sauvegarder
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("💾 Modèle sauvegardé : rf_model.pkl")
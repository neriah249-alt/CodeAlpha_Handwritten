import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_training_history():
    """
    Affiche les courbes d'entraînement si l'historique existe
    """
    history_path = '../models/history.pkl'
    
    if not os.path.exists(history_path):
        print("⚠️ Historique non trouvé. Entraînement terminé à epoch 22/30.")
        print("   Modifie train.py pour sauvegarder l'historique si tu veux les courbes.")
        return
    
    import pickle
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Entraînement', color='blue', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation', color='orange', linewidth=2)
    axes[0].set_title('Accuracy au fil des epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.9, 1.0])
    
    # Loss
    axes[1].plot(history['loss'], label='Entraînement', color='blue', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation', color='orange', linewidth=2)
    axes[1].set_title('Loss au fil des epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Graphique sauvegardé : data/training_history.png")

def plot_predictions(model, x_test, y_test, num_images=16):
    """
    Affiche des prédictions sur un échantillon de test
    """
    predictions = model.predict(x_test[:num_images], verbose=0)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    correct = 0
    
    for i in range(num_images):
        axes[i].imshow(x_test[i].squeeze(), cmap='gray')
        pred_label = np.argmax(predictions[i])
        true_label = np.argmax(y_test[i])
        
        if pred_label == true_label:
            correct += 1
            color = 'green'
        else:
            color = 'red'
        
        axes[i].set_title(f'Prédit: {pred_label}\nRéel: {true_label}', 
                         color=color, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Prédictions sur {num_images} images\n'
                f'✅ Correctes: {correct}/{num_images} ({correct/num_images*100:.1f}%)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('../data/predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Prédictions sauvegardées : data/predictions.png")

def plot_confusion_matrix(model, x_test, y_test):
    """
    Affiche la matrice de confusion
    """
    print("🔍 Calcul des prédictions sur 10 000 images de test...")
    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculer l'accuracy
    accuracy = np.mean(y_pred == y_true)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    
    plt.title(f'Matrice de Confusion\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../data/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Matrice de confusion sauvegardée : data/confusion_matrix.png")

def plot_errors(model, x_test, y_test, num_errors=16):
    """
    Affiche les images mal prédites
    """
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Trouver les erreurs
    errors = np.where(y_pred != y_true)[0]
    
    if len(errors) == 0:
        print("🎉 Aucune erreur trouvée ! Modèle parfait.")
        return
    
    print(f"🔍 {len(errors)} erreurs trouvées sur {len(y_test)} images")
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_errors, len(errors))):
        idx = errors[i]
        axes[i].imshow(x_test[idx].squeeze(), cmap='gray')
        pred = y_pred[idx]
        true = y_true[idx]
        conf = np.max(predictions[idx])
        axes[i].set_title(f'Prédit: {pred}\nRéel: {true}\nConf: {conf:.2%}', 
                         color='red', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Images Mal Prédites ({len(errors)} erreurs au total)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../data/errors.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Erreurs sauvegardées : data/errors.png")

def main():
    print("="*50)
    print("📊 VISUALISATION DES RÉSULTATS")
    print("="*50)
    
    # Charger les données
    print("\n🔍 Chargement du dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Charger le modèle
    print("🔍 Chargement du modèle...")
    model = load_model('../models/best_model.h5')
    
    # Évaluation
    print("\n📊 Évaluation du modèle...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Accuracy sur le test : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Loss sur le test : {test_loss:.4f}")
    
    # Visualisations
    print("\n" + "="*50)
    print("🎨 Génération des visualisations...")
    print("="*50)
    
    # 1. Historique d'entraînement (si disponible)
    print("\n1️⃣  Courbes d'entraînement...")
    plot_training_history()
    
    # 2. Prédictions sur échantillon
    print("\n2️⃣  Prédictions sur 16 images...")
    plot_predictions(model, x_test, y_test, num_images=16)
    
    # 3. Matrice de confusion
    print("\n3️⃣  Matrice de confusion...")
    plot_confusion_matrix(model, x_test, y_test)
    
    # 4. Images mal prédites
    print("\n4️⃣  Images mal prédites...")
    plot_errors(model, x_test, y_test, num_errors=16)
    
    print("\n" + "="*50)
    print("✅ TOUTES LES VISUALISATIONS SONT PRÊTES !")
    print("="*50)
    print("\n📁 Fichiers générés dans data/ :")
    print("   - training_history.png")
    print("   - predictions.png")
    print("   - confusion_matrix.png")
    print("   - errors.png")

if __name__ == "__main__":
    main()
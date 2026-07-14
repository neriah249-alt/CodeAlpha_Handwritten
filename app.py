import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="🔢 Reconnaissance de Chiffres Manuscrits",
    page_icon="🔢",
    layout="centered"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTitle { color: #1f77b4; text-align: center; }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .digit-display {
        font-size: 100px;
        font-weight: bold;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CHARGEMENT DU MODÈLE
# ==========================================
@st.cache_resource
def load_model_cached():
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        st.error("❌ Modèle non trouvé. Vérifiez le déploiement.")
        st.stop()
    return load_model(model_path)

# ==========================================
# PRÉTRAITEMENT
# ==========================================
def preprocess_image(image):
    img = np.array(image)
    
    # Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Inverser si fond clair
    if np.mean(img) > 127:
        img = 255 - img
    
    # Redimensionner
    img = cv2.resize(img, (28, 28))
    
    # Normaliser
    img = img.astype('float32') / 255.0
    
    # Reshape
    img = np.expand_dims(img, axis=(0, -1))
    
    return img

# ==========================================
# INTERFACE
# ==========================================
def main():
    # Header
    st.title("🔢 Reconnaissance de Chiffres Manuscrits")
    st.markdown("""
        <p style='text-align: center; color: #666;'>
        Projet <b>CodeAlpha Machine Learning</b> — Deep Learning CNN
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charger modèle
    with st.spinner("🧠 Chargement du modèle CNN..."):
        model = load_model_cached()
    st.success("✅ Modèle chargé ! Accuracy : 99.50%")
    
    st.markdown("---")
    
    # Upload
    st.subheader("📤 Télécharge une image de chiffre")
    
    uploaded_file = st.file_uploader(
        "Format accepté : PNG, JPG, JPEG",
        type=['png', 'jpg', 'jpeg']
    )
    
    # Option exemple
    use_example = st.checkbox("🎯 Utiliser un exemple MNIST")
    
    if use_example and not uploaded_file:
        # Générer un chiffre exemple (le 7)
        from tensorflow.keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
        example_idx = 0  # Chiffre 7
        img_array = x_test[example_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_array, caption="Exemple MNIST", use_column_width=True)
        
        with col2:
            # Prédire
            img_processed = img_array.astype('float32') / 255.0
            img_processed = np.expand_dims(img_processed, axis=(0, -1))
            pred = model.predict(img_processed, verbose=0)
            
            digit = np.argmax(pred[0])
            conf = np.max(pred[0])
            
            st.markdown(f"""
                <div class="result-box">
                    <p style="font-size: 20px; margin: 0;">Chiffre prédit</p>
                    <p class="digit-display">{digit}</p>
                    <p style="font-size: 18px;">Confiance : {conf:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
    
    elif uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ton image", use_column_width=True)
        
        with col2:
            # Prédire
            img_processed = preprocess_image(image)
            pred = model.predict(img_processed, verbose=0)
            
            digit = np.argmax(pred[0])
            conf = np.max(pred[0])
            
            st.markdown(f"""
                <div class="result-box">
                    <p style="font-size: 20px; margin: 0;">Chiffre prédit</p>
                    <p class="digit-display">{digit}</p>
                    <p style="font-size: 18px;">Confiance : {conf:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Probabilités
        st.markdown("---")
        st.subheader("📊 Probabilités par chiffre")
        
        cols = st.columns(10)
        for i, prob in enumerate(pred[0]):
            with cols[i]:
                st.metric(label=str(i), value=f"{prob:.2%}")
                st.progress(float(prob))
    
    # Footer
    st.markdown("---")
    st.caption("""
        💡 **Conseil** : Pour de meilleurs résultats, utilise une image avec 
        fond noir et chiffre blanc, bien centré.
        
        🔗 [CodeAlpha](https://www.codealpha.tech) | 
        📧 services@codealpha.tech
    """)

if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os

st.set_page_config(page_title="CodeAlpha - Testez !", page_icon="🔢")

st.title("🔢 Reconnaissance de Chiffres Manuscrits")
st.markdown("### Projet CodeAlpha — Machine Learning")

st.info("""
📌 **Note** : Ce modèle fonctionne mieux avec des chiffres bien formés, 
centrés et avec un bon contraste.
""")

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
@st.cache_resource
def load_model():
    model_path = 'src/rf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("❌ Modèle non trouvé.")
    st.stop()

# ============================================
# PRÉTRAITEMENT SIMPLE MAIS EFFICACE
# ============================================
def preprocess_simple(image):
    """
    Prétraitement simple et robuste
    """
    # 1. Convertir en niveaux de gris
    if image.mode != 'L':
        image = image.convert('L')
    
    # 2. Redimensionner à 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 3. Convertir en array numpy
    img_array = np.array(image, dtype=np.float32)
    
    # 4. Normaliser [0, 1]
    img_array = img_array / 255.0
    
    # 5. Vérifier si inversion nécessaire (fond blanc)
    mean_val = np.mean(img_array)
    if mean_val > 0.5:
        # Fond blanc → inverser
        img_array = 1.0 - img_array
    
    # 6. Aplatir pour Random Forest
    img_flat = img_array.flatten()
    
    return img_flat, img_array

# ============================================
# INTERFACE
# ============================================
uploaded_file = st.file_uploader("📎 Choisis une image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Ton image", use_container_width=True)
    
    with col2:
        # Prétraitement
        img_flat, img_processed = preprocess_simple(image)
        img_flat = img_flat.reshape(1, -1)
        
        # Prédire
        prediction = model.predict(img_flat)[0]
        proba = model.predict_proba(img_flat)[0]
        confidence = np.max(proba)
        
        # Afficher le résultat
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 20px; text-align: center; color: white;">
            <p style="font-size: 20px; margin: 0;">Chiffre prédit</p>
            <p style="font-size: 100px; font-weight: bold; margin: 0;">{prediction}</p>
            <p style="font-size: 18px;">Confiance : {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statut selon confiance
        if confidence > 0.5:
            status_color = "#4CAF50"
            status_text = "✅ Prédiction fiable"
        elif confidence > 0.3:
            status_color = "#FF9800"
            status_text = "⚠️ Prédiction modérée"
        else:
            status_color = "#F44336"
            status_text = "❌ Prédiction faible"
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; 
                    background-color: {status_color}20; border: 2px solid {status_color};
                    text-align: center; margin: 10px 0;">
            <span style="font-size: 20px; font-weight: bold; color: {status_color};">
                {status_text}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Barres de probabilités
        st.markdown("### 📊 Probabilités")
        
        cols = st.columns(10)
        for i in range(10):
            with cols[i]:
                prob = proba[i]
                height = int(prob * 100)
                color = "#4CAF50" if i == prediction else "#2196F3"
                
                st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="width: 20px; height: {height}px; 
                         background: {color}; border-radius: 3px 3px 0 0; 
                         min-height: 3px;">
                    </div>
                    <div style="font-size: 14px; font-weight: {'bold' if i == prediction else 'normal'}; 
                                color: {'#4CAF50' if i == prediction else 'black'};">
                        {i}
                    </div>
                    <div style="font-size: 9px; color: #888;">
                        {prob:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ À propos")
    st.markdown("""
    **Modèle :** Random Forest
    
    **Dataset :** MNIST (70 000 images)
    
    **Prétraitement :**
    - Niveaux de gris
    - Redimensionnement 28x28
    - Normalisation [0,1]
    - Inversion automatique
    """)
    
    st.markdown("---")
    st.subheader("📊 Seuils")
    st.markdown("""
    - ✅ > 50% : Fiable
    - ⚠️ 30-50% : Modérée
    - ❌ < 30% : Faible
    """)
    
    st.markdown("---")
    st.subheader("💡 Conseils")
    st.markdown("""
    1. **Fond blanc** + trait noir
    2. Chiffre bien **centré**
    3. Photo **bien éclairée**
    4. Éviter les **ombres**
    """)
    
    st.markdown("---")
    st.caption("🎓 CodeAlpha Internship 2026 | Tâche 3")
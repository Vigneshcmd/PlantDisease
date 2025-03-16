import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# ✅ Set Streamlit page config as the FIRST command
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔹 Function to Set Background Color
def set_background_color(bg_color="#1E1E1E"):
    """Set a solid background color for the app."""
    bg_style = f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {bg_color} !important;
        }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# ✅ Apply Background Color
set_background_color("#1E1E1E")  # Dark Gray Background (Change if needed)

# 🔹 Load the pre-trained model
@st.cache_resource  # ✅ Caches model for efficiency
def load_model():
    return tf.keras.models.load_model('plant_disease_cnn_model.keras')

model = load_model()

# 🔹 Function to preprocess & predict plant disease
def model_predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# 🔹 Class Labels
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry_Powdery_mildew', 'Cherry_healthy', 'Corn_Cercospora_leaf_spot',
    'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn_healthy', 'Grape_Black_rot',
    'Grape_Esca', 'Grape_Leaf_blight', 'Grape_healthy', 'Orange_Citrus_greening',
    'Peach_Bacterial_spot', 'Peach_healthy', 'Pepper_Bacterial_spot', 'Pepper_healthy',
    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy',
    'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
]

# 🔹 Sidebar Navigation
with st.sidebar:
    st.title('🌱 Plant Health Checker')
    st.markdown("### 🌍 Empowering Sustainable Agriculture")
    app_mode = st.radio('', ['🏡 Home', '🔍 Detect Disease'])

# 🔹 Home Page
if app_mode == '🏡 Home':
    st.title("🌿 Plant Disease Detector")
    st.write("An AI-powered tool to diagnose plant diseases from images.")

    # ✅ Add a homepage image
    st.image("detect.jpg", caption="🌱 Plant Health Monitoring", use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎯 Accurate")
        st.write("Powered by AI for precise disease detection")

    with col2:
        st.subheader("⚡ Fast")
        st.write("Get instant results within seconds")

    with col3:
        st.subheader("🌍 Easy")
        st.write("User-friendly for farmers and gardeners")

# 🔹 Disease Recognition Page
elif app_mode == '🔍 Detect Disease':
    st.title("📷 Upload an Image for Diagnosis")

    uploaded_file = st.file_uploader("Upload your plant image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file
        save_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ Image Uploaded Successfully")

        if st.button("🔬 Analyze Image"):
            with st.spinner('Analyzing image...'):
                result_index = model_predict(save_path)
            
            st.subheader("🔍 Prediction Result:")
            st.write(f"### **{class_names[result_index]}**")

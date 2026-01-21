import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras


# Page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 0.8rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .severity-nondemented {
        background-color: #C8E6C9;
        border-left: 5px solid #4CAF50;
    }
    .severity-verymild {
        background-color: #FFF9C4;
        border-left: 5px solid #FFEB3B;
    }
    .severity-mild {
        background-color: #FFE0B2;
        border-left: 5px solid #FF9800;
    }
    .severity-moderate {
        background-color: #FFCDD2;
        border-left:  5px solid #F44336;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_LABELS = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

CLASS_INFO = {
    'NonDemented': {
        'description': 'No signs of Alzheimer detected.',
        'severity': 'Normal',
        'color': '#4CAF50',
        'css_class': 'severity-nondemented'
    },
    'VeryMildDemented': {
        'description':  'Early signs of Alzheimer with very mild symptoms.',
        'severity': 'Very Mild',
        'color': '#FFEB3B',
        'css_class': 'severity-verymild'
    },
    'MildDemented': {
        'description': 'Clear signs of Alzheimer, but still mild.',
        'severity': 'Mild',
        'color': '#FF9800',
        'css_class': 'severity-mild'
    },
    'ModerateDemented': {
        'description': 'More pronounced symptoms of Alzheimer, moderate severity.',
        'severity': 'Moderate',
        'color': '#F44336',
        'css_class': 'severity-moderate'
    }
}


@st.cache_resource
def load_model():
    """Load the trained Keras model."""
    try:
        model = keras.models.load_model("alzheimer_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model:  {str(e)}")
        return None


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Resize with padding to maintain aspect ratio
    img_tensor = tf. image.resize_with_pad(img_tensor, 224, 224)
    
    # Add batch dimension
    img_tensor = tf. expand_dims(img_tensor, 0)
    
    return img_tensor


def predict(model, image):
    """Make prediction on the preprocessed image."""
    predictions = model.predict(image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = CLASS_LABELS[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def display_results(predicted_class, confidence, all_probabilities):
    """Display prediction results with visualizations."""
    class_info = CLASS_INFO[predicted_class]
    
    # Main prediction result
    st.markdown(f"""
        <div class="prediction-box {class_info['css_class']}">
            <h3 style="margin:  0; color: #333;">Prediction: {predicted_class}</h2>
            <p style="margin: 0.3rem 0; font-size: 1.2rem;">
                <strong>Severity Level:</strong> {class_info['severity']}
            </p>
            <p style="margin: 0.3rem 0; font-size: 1.2rem;">
                <strong>Confidence: </strong> {confidence:.2f}%
            </p>
            <p style="margin: 0.3rem 0; color: #555;">
                {class_info['description']}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Probability distribution
    st.subheader("📊 Probability Distribution")
    
    # Create horizontal bar chart
    for i, (label, prob) in enumerate(zip(CLASS_LABELS, all_probabilities)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob), text=f"{label}")
        with col2:
            st. write(f"{prob * 100:.2f}%")


def main():
    # Header
    st.markdown('<p class="main-header">🧠 Alzheimer\'s Disease Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a Patient\'s MRI scan to predict the severity of Alzheimer\'s disease</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a deep learning model based on **EfficientNet-B0** 
        to classify any Patient\'s MRI scans into four categories of Alzheimer's disease severity.
        """)
        
        st.header("📋 Classification Categories")
        for label, info in CLASS_INFO.items():
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; border-left: 3px solid {info['color']}; padding-left: 0.5rem;">
                <strong>{label}</strong><br>
                <small>{info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational and research purposes only. 
        It should NOT be used as a substitute for professional medical diagnosis.  
        Always consult qualified healthcare professionals for medical advice.
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st. error("Failed to load the model.")
        return
    
    # File uploader
    st.header("📤 Upload MRI Scan")
    uploaded_file = st. file_uploader(
        "Choose a brain MRI image.. .",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a Patient\'s MRI scan image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1,2], border=True)
        
        with col1:
            st.subheader("🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, width="stretch")
            
            # Display image info
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
            st.caption(f"Color mode: {image. mode}")
        
        with col2:
            st.subheader("🔍 Analysis Result")
            
            with st.spinner("Please wait: Analyzing MRI scan..."):
                # Preprocess and predict
                processed_image = preprocess_image(image)
                predicted_class, confidence, all_probs = predict(model, processed_image)
            
            # Display results
            display_results(predicted_class, confidence, all_probs)
    
    else:
        # Show placeholder when no image is uploaded
        st. info("👆 Please upload a Patient\'s MRI scan image to get started.")
        
        # Example section
        st.header("📚 How to Use")
        st.markdown("""
        1. **Upload** a Patient\'s MRI scan image using the file uploader above
        2. **Wait** for the model to analyze the image
        3. **View** the prediction results and probability distribution
        
        ### Supported Image Formats
        - JPEG (.jpg, . jpeg)
        - PNG (.png)
        
        ### Tips for Best Results
        - Use clear, high-quality MRI scan images
        - Ensure the brain is clearly visible in the scan
        - Avoid heavily compressed or distorted images
        """)


if __name__ == "__main__":
    main()
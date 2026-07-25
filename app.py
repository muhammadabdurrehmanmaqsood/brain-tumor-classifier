import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title("Brain Tumor MRI Classifier")
st.write("Upload an MRI scan to detect potential tumors. Note: This is an educational MLOps prototype, not a medical diagnostic tool.")

# 2. Cache the model load so it doesn't run from scratch on every click
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    # Ensure map_location='cpu' is set since free servers don't have GPUs
    model.load_state_dict(torch.load('resnet50_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. Define Transformations and Labels
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# 4. User Interface
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    st.write("Classifying...")
    
    # Load model and predict
    try:
        model = load_model()
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Display Results
        st.subheader("Results:")
        for i in range(4):
            st.write(f"**{labels[i]}**: {float(probabilities[i]) * 100:.2f}%")
            
    except FileNotFoundError:
        st.error("Model weights file (resnet50_best.pth) not found. Please ensure it is uploaded to the repository.")

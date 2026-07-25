import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# 1. Page Configuration
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title("Brain Tumor MRI Classifier")
st.write("Upload an MRI scan to detect potential tumors. Note: This is an educational MLOps prototype, not a medical diagnostic tool.")

# 2. Download and Cache the Model
@st.cache_resource
def load_model():
    model_path = 'resnet50_best.pth'
    
    # Check if the weights file exists, if not, download it directly from Google Drive
    if not os.path.exists(model_path):
        st.info("Downloading model weights... this may take a minute.")
        
        file_id = '1HkvKj-FFnxaChUUxtFEGZJ0ZpW38HOiI' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)

    # Initialize the ResNet50 architecture
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    # Load the downloaded weights (mapped to CPU for the Streamlit free tier)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. Define Transformations and Labels
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names must match the alphabetical order from your PyTorch ImageFolder
labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# 4. User Interface and Inference Pipeline
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    st.write("Classifying...")
    
    # Execute inference
    try:
        model = load_model()
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            # Convert raw logits to probability percentages using Softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Display formatted results
        st.subheader("Results:")
        for i in range(4):
            st.write(f"**{labels[i]}**: {float(probabilities[i]) * 100:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")

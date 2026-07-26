import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# 1. Page Configuration
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="wide")
st.title("Multimodal Brain Tumor Classifier")
st.write("Upload an MRI or CT scan to detect potential tumors. Note: This is an educational MLOps prototype, not a medical diagnostic tool.")

# 2. Sidebar for Modality Selection
st.sidebar.header("Configuration")
modality = st.sidebar.selectbox("Select Scan Modality", ["MRI", "CT"])

# 3. Model Configuration Mappings
MODEL_CONFIGS = {
    "MRI": {
        "weights_file": "resnet50_best.pth",
        "file_id": "1HkvKj-FFnxaChUUxtFEGZJ0ZpW38HOiI",
        "num_classes": 4,
        "labels": ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    },
    "CT": {
        "weights_file": "resnet50_ct_best.pth",
        "file_id": "1jE-pIz3flMkjKwXemesd2ykakpS90aJU",
        "num_classes": 2,
        "labels": ['Healthy', 'Tumor']
    }
}

config = MODEL_CONFIGS[modality]

# 4. Download and Cache the Selected Model
@st.cache_resource(show_spinner=False)
def load_model(weights_file, file_id, num_classes):
    if not os.path.exists(weights_file):
        st.info(f"Downloading {weights_file} weights from cloud storage... this may take a minute.")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, weights_file, quiet=False)

    # Initialize the ResNet50 architecture dynamically
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the downloaded weights (mapped to CPU for the Streamlit free tier hosting)
    model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
    model.eval()
    return model

# 5. Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. User Interface and Inference Pipeline
uploaded_file = st.file_uploader(f"Choose a {modality} image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption=f'Uploaded {modality} Scan', width=400)
    
    st.write("Classifying...")
    
    # Execute inference
    try:
        model = load_model(config["weights_file"], config["file_id"], config["num_classes"])
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            # Convert raw logits to probability percentages using Softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Display formatted results dynamically based on modality configuration
        st.subheader("Results:")
        for i in range(config["num_classes"]):
            st.write(f"**{config['labels'][i]}**: {float(probabilities[i]) * 100:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")

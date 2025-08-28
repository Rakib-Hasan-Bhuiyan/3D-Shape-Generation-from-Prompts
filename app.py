import streamlit as st
import base64
import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import torchvision.transforms as T
from model import Image2PCgen
from evaluate import text_inference

# Path to saved model checkpoint
CHECKPOINT_PATH = "1024_aiplane_car.pth"
# Path to save predicted point clouds as .npy
SAVE_PATH = "test_samples"

# Define the image preprocessing transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# --- Model Loading and Caching ---
@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Image2PCgen(num_points=1024)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        st.error(f"Model checkpoint not found at {checkpoint_path}. Using un-trained model.")

    model.to(device)
    model.eval()
    return model, device

def plot_point_cloud(predicted_pc_np):
    fig = go.Figure(data=[go.Scatter3d(
        x=predicted_pc_np[:, 0],
        y=predicted_pc_np[:, 1],
        z=predicted_pc_np[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color='royalblue',
            opacity=1
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            aspectmode='data'
        ),
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="Multi-Modal prompt to 3D", layout="wide")
    # Get the Base64 string of your image file
    image_base64 = get_base64_image("images/app_background.jpg")

    # Custom CSS for background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.title("How it Works")
        st.markdown("---")
        st.write("We use an image encoder (Vision Transformer, ViT B16) to extract features from the images and pass it to our transformer decoder that generates a 3D point cloud based on learned queries on the features.")
        
        # Load and display the model architecture image
        st.image("images/model_architecture.png", caption="Model Architecture", use_container_width=True)

    model, device = load_model(CHECKPOINT_PATH)

    col1, col2 = st.columns([1, 1])

    # Left column content
    with col1:
        st.title("3D Shape Generation from Multi-Modal Prompts")
        st.markdown("Input an image or a text prompt and the model will generate a 3D point cloud.")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["png", "jpg", "jpeg"]
        )
        
        # Change st.text_area to st.text_input to make it a single line
        text_prompt = st.text_input("Or enter a text prompt:")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            # Make the image preview smaller by using a specific width
            st.image(image, caption="Uploaded Image", width=150)
            
            # Display the text prompt if one was provided
            if text_prompt:
                st.write(f"**Text Prompt:** {text_prompt}")

    # Right column content
    with col2:
        plot_container = st.empty()
        predicted_pc_np = None
        st.markdown("<br>", unsafe_allow_html=True)

        filename = None
        if uploaded_file is not None:
            with st.spinner('Generating point cloud from image...'):
                processed_image = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted_pc = model(processed_image)
                predicted_pc_np = predicted_pc.squeeze(0).cpu().numpy()
                base_name = os.path.splitext(uploaded_file.name)[0]
                filename = f"/{base_name}_pred.npy"
                
        elif text_prompt:
            st.info(f"Using text prompt: '{text_prompt}'")
            with st.spinner('Retrieving best image and generating point cloud...'):
                predicted_pc_tensor = text_inference(model, device, text_prompt)
            if predicted_pc_tensor is not None:
                predicted_pc_np = predicted_pc_tensor.squeeze(0).cpu().numpy()
                # Create a filename from the text prompt
                safe_prompt = text_prompt.replace(" ", "_").replace("/", "").replace("\\", "")
                filename = f"/{safe_prompt}_pred.npy"
            else:
                st.warning("Could not generate a point cloud from the text prompt.")
                predicted_pc_np = None
                
        if predicted_pc_np is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            fig = plot_point_cloud(predicted_pc_np)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            if st.button("Save predicted point cloud"):
                try:
                    if not os.path.exists(SAVE_PATH):
                        os.makedirs(SAVE_PATH)
                    np.save(SAVE_PATH + filename, predicted_pc_np)
                    st.success(f"File '{filename}' has been saved.")
                except Exception as e:
                    st.error(f"Failed to save file: {e}")
            
        else:
            plot_container.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
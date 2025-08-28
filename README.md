# 3D Shape Generation from Multi-Modal Prompts
<p align="center">
    <img src="images/model_architecture.png" alt="Overview">
</p>

Our model, **"Image2PCgen"** takes an single RGB image and generates a 3D point cloud. A text-to-image retrieval pipeline is used to generate point clouds from text propmts. The model is designed with pre-trained vision transformer (ViT base_patch_16) as the image encoder and a transformer-based point cloud decoder.

## Installation
This repository is tested on `Python 3.9.11`, `PyTorch 2.6.0` and `CUDA 12.4`

Clone the repository using:
```
git clone https://github.com/Rakib-Hasan-Bhuiyan/Image2PCgen.git
```
Navigate to project directory and install:
```
pip install streamlit
pip install torchvision
pip install numpy
pip install Pillow
pip install plotly
pip install tqdm
pip install timm
pip install clip
pip install kaolin
pip install wheel
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Dataset
Download the ShapeNet (partial with three categories) dataset from here and place it in the project directory:
```
https://drive.google.com/file/d/1fqBnyboDD9mPGMM8QKGiaRsqJZe0-KUZ/view?usp=sharing
```

## Training
Train the model using:
```
python train.py
```

## Pretrained Model
Download the checkpoint from our model trained on 'Airplane' and 'Car' category from the ShapeNet dataset
```
https://drive.google.com/file/d/1Ri7HFgZiDoY7mAJFuEXSauhEfJNZc348/view?usp=sharing
```

## Inference
Launch the streamlit application using:
```
streamlit run app.py
```
The application provides an easy to use UI and support for multi-modal inputs. 


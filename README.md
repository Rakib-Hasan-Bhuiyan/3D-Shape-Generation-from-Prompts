# 3D Shape Generation from Multi-Modal Prompts
<p align="center">
    <img src="images/model_architecture.png" alt="Overview">
</p>

Our model, **"Image2PCgen"** takes an single RGB image and generates a 3D point cloud. A text-to-image retrieval pipeline is used to generate point clouds from text propmts. The model is designed with pretrained vision transformer (ViT base_patch_16) as the image encoder and a transformer-based point cloud decoder.

## Installation
This repository is tested on `Python 3.9.11`, `PyTorch 2.6.0` and `CUDA 12.4`

Clone the repository using:
```
git clone https://github.com/Rakib-Hasan-Bhuiyan/Image2PCgen.git
```
Navigate to project directory and install the dependencies:
```
pip install -r requirements.txt
```

## Dataset
Download the ShapeNet (partial with three categories) dataset from here and extract it in the project directory:
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
For text-to-3D, download image_embeddings.pkl and place it in the project directory:
```
https://drive.google.com/file/d/1lWTyT2hi4Gw-bn1Yoa_gFzU5mqIJOtAL/view?usp=sharing
```
Or, compute the paths and embeddings from the dataset using:
```
python utils.py compute_image_embeddings
```
Navigate to project directory and launch the streamlit application using:
```
streamlit run app.py
```



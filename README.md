# Alzheimer's Disease Classification using Deep Learning

A deep learning project that classifies the severity of Alzheimer's disease from brain MRI scans using transfer learning with EfficientNet-B0.

Link to the Kaggle Notebook: https://www.kaggle.com/code/griittyy/alzheimer-s-multiclass-dataset-prediction-model
- Be sure to enable the GPU Runtime.

You can Run or Test the LIVE MODEL here too:
[https://alzheimermodel.streamlit.app](https://alzheimermodel.streamlit.app)

## 🎯 Project Objective

Build a Neural Network Model to predict the severity of a patient's Alzheimer's disease given the medical image of the patient's MRI scan.

## 📊 Dataset

The project uses the **[Alzheimer's Disease Multiclass Dataset](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented)** containing approximately **44,000 MRI images** categorized into four distinct classes based on disease severity:

| Class | Description | Number of Images |
|-------|-------------|------------------|
| NonDemented | No signs of dementia | 12,800 |
| VeryMildDemented | Early signs, very mild symptoms | 11,200 |
| MildDemented | Clear signs of dementia, but still mild | 10,000 |
| ModerateDemented | More pronounced symptoms, moderate severity | 10,000 |

### Image Details
- **Total Images**: 44,000
- **Format**: MRI scans as JPG files
- **Image Sizes**: Mixed (200×190, 180×180, 176×208)
- **Color Modes**: RGB (33,984 images) and Grayscale (10,016 images)

## 🏗️ Model Architecture

The model leverages **transfer learning** with **EfficientNet-B0** as the backbone:

```
EfficientNet-B0 (frozen, pre-trained on ImageNet)
    ↓
Global Average Pooling 2D
    ↓
Batch Normalization
    ↓
Dropout
    ↓
Dense Layer (4 classes, Softmax)
```

### Preprocessing Pipeline
- **Image Resizing**: All images resized to 224×224 with aspect ratio preservation and zero-padding
- **Color Conversion**:  Grayscale images converted to RGB (3 channels)
- **Stratified Splitting**:  Ensures balanced class distribution across splits

### Data Split
| Split | Size | Percentage |
|-------|------|------------|
| Training | 35,200 images | 80% |
| Validation | 4,400 images | 10% |
| Test | 4,400 images | 10% |

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **TensorFlow / Keras** - Deep learning framework
- **EfficientNet-B0** - Pre-trained CNN for transfer learning
- **Pandas** - Data manipulation
- **Matplotlib / Seaborn** - Visualization
- **Scikit-learn** - Train/test splitting, metrics
- **PIL (Pillow)** - Image processing
- **tqdm** - Progress bars

## 📁 Repository Structure

```
Alzheimer-Disease-Classification/
├── alzheimers-prediction-model.ipynb    # Training notebook
├── alzheimer_model.keras                # Trained model weights
├── app.py                               # Streamlit application
├── requirements. txt                    # Python dependencies for Streamlit Application
├── notebook_reqs.txt                    # Python dependencies for Training notebook
└── README.md                            # Project documentation
```

## 🚀 Getting Started
### Usage

1. Clone the repository: 
```bash
git clone https://github.com/GRIITTYY/Alzheimer-Disease-Classification.git
cd Alzheimer-Disease-Classification
```

2. Install Libraries
```bash
pip install -r notebook_reqs.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook alzheimers-prediction-model.ipynb
```

4. To use the pre-trained model for inference:
```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('alzheimer_model.keras')

# Class labels
labels = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Preprocess and predict
img = tf.io.read_file('path/to/mri_scan.jpg') # Replace with path to the single MRI scan Image
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize_with_pad(img, 224, 224)
img = tf.cast(img, tf. float32)
img = tf.expand_dims(img, 0)

prediction = model.predict(img)
predicted_class = labels[tf.argmax(prediction[0]).numpy()]
print(f"Predicted:  {predicted_class}")
```

## 📈 Key Features

- **Transfer Learning**:  Utilizes EfficientNet-B0 pre-trained on ImageNet for robust feature extraction
- **Bias Audit**: Dataset verified to be robust against metadata shortcuts (image size doesn't correlate with labels)
- **Efficient Data Pipeline**: Uses `tf.data` with parallel processing and prefetching for optimal GPU utilization
- **Stratified Sampling**:  Maintains consistent class distribution across train/val/test splits

## 📋 Notebook Sections

1. **Data Exploration** - Visualizing sample MRI images from each class
2. **Data Analysis** - Class distribution and image metadata analysis
3. **Bias Audit** - Ensuring model can't exploit metadata shortcuts
4. **Preprocessing** - Image resizing, normalization, and data pipeline creation
5. **Model Building** - EfficientNet-B0 transfer learning architecture
6. **Training** - Model training with validation monitoring
7. **Evaluation** - Performance metrics and confusion matrix

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset sourced from [Kaggle's Alzheimer's Multiclass Dataset](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented)
- EfficientNet architecture by Google Research
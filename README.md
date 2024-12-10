# Image Classification with Machine Learning Models

This repository demonstrates image classification using machine learning models implemented with TensorFlow and a user-friendly interface built with Streamlit. The project includes two models: **MobileNetV2 (ImageNet)** for general image classification and a **custom-trained CIFAR-10 model** for specific CIFAR-10 dataset classification.

## 🚀 Features

- **MobileNetV2 (Pre-trained on ImageNet)**  
  Classifies a wide range of objects with high accuracy using a pre-trained MobileNetV2 model.  
- **CIFAR-10 (Custom-trained model)**  
  Recognizes 10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.  
- **Interactive Web App**  
  Streamlit-powered interface for image upload, model selection, and classification visualization.  

---

## 🛠 Prerequisites

Ensure you have the following installed on your system:
- Python 3.7–3.11
- TensorFlow
- Streamlit
- Pillow
- NumPy

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Implementation-of-ML-Model-for-Image-Classification
   ```
2. Install the required dependencies:
   ```bash
   streamlit run app.py
   ```

---
## 🚀 Usage

1. Run the Streamlit application:
   ```bash
    streamlit run app.py
   ```
2. Open your web browser and navigate to the provided local URL.
3. Steps to classify an image:
   - Upload an image in JPG or PNG format.
   - Select the desired model (MobileNetV2 or CIFAR-10).
   - View the predicted class along with the confidence score.

---

##🧠 Models
1. MobileNetV2
    - Pre-trained on the ImageNet dataset.
    - Used for general-purpose object classification.
2. CIFAR-10
   - Custom-trained TensorFlow model for classifying CIFAR-10 dataset images.
   - Categories include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

---
  ##📂 Directory Structure
   - app.py: Main Streamlit app file
   - model111.h5: Pre-trained CIFAR-10 model
   - requirements.txt: Required Python dependencies
   - README.md: Project documentation
   - assets/: Optional folder to store images or other assets

---
  ## 🛠 Future Improvements
   - Add more pre-trained models for diverse classification tasks.
   - Enhance the CIFAR-10 model for better accuracy.
   - Implement model training scripts for reproducibility.

---
  ## 🤝 Contributing

  Contributions are welcome! Feel free to submit issues or pull requests for any enhancements or bug fixes.
  



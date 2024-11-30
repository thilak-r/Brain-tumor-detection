# Brain Tumor Detection 🧠🔍

This project is a machine learning application designed to assist in the detection of brain tumors using medical imaging. The system leverages deep learning models to analyze MRI images and predict whether a brain tumor is present, helping to support early diagnosis and treatment.

---

## ✨ Features

- **Brain Tumor Classification**: Detects the presence of brain tumors in MRI scans.
- **Deep Learning Model**: Uses a trained neural network for high-accuracy predictions.
- **Web Interface**: Provides an easy-to-use web-based platform for uploading and analyzing images.

---

## 📂 Project Structure

The repository includes the following key components:


- `templates/`: HTML templates for rendering the web interface.
- `app.py`: The main Flask application script for running the web server.
- `brain tumor.ipynb`: Jupyter notebook for exploratory data analysis (EDA) and model development.
- `brain_tumor_model.pth`: The trained PyTorch model for brain tumor detection.
- `README.md`: Project documentation and setup instructions.

---

## 🚀 Getting Started

### Prerequisites

To set up and run this project, ensure you have the following installed:

- Python 3.x
- Required libraries: `Flask`, `PyTorch`, `Pillow`, `numpy`, etc.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/thilak-r/Brain-tumor-detection.git
   cd Brain-tumor-detection
   ```

2. **Create and Activate a Virtual Environment**:

   ```bash
   python -m venv venv
   # Activate the virtual environment:
   source venv/bin/activate    # For Linux/macOS
   venv\Scripts\activate       # For Windows
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask Application**:

   ```bash
   python app.py
   ```

5. **Access the Application**:

   Open your browser and navigate to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 💡 Usage

1. Upload an MRI image through the web interface.
2. Click **"Analyze"** to process the image.
3. View the prediction result, indicating whether a brain tumor is detected.

---

## 🧠 Model Details

- **Architecture**: The model is based on a convolutional neural network (CNN) trained to classify MRI images.
- **File Descriptions**:
  - `brain_tumor_model.pth`: PyTorch file containing the pre-trained model.
  - `brain tumor.ipynb`: Notebook for training, evaluation, and visualization.

---


## 🙏 Acknowledgments

-[Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu).


---

Feel free to contribute, raise issues, or suggest features to thilak22005@gmail.com

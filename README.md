# 🛡️ Face Mask Detection using CNN & OpenCV

This project uses a **Convolutional Neural Network (CNN)** to detect whether a person is wearing a mask or not in **real-time** via a webcam.  
The model is trained in **Google Colab**, saved as `.h5`, and then deployed locally using **OpenCV**.

---

## 📌 Features
- Trained CNN model (`.h5`) for mask detection
- Real-time face detection using Haar Cascade
- Color-coded bounding boxes:
  - 🟢 **Green**: Mask detected
  - 🔴 **Red**: No mask detected
- Lightweight & works on CPU

---

## 📂 Project Structure
```

face-mask-detector/
│
├── face\_mask\_model.h5                   # Trained CNN model
├── haarcascade\_frontalface\_default.xml   # Haar Cascade file for face detection
├── detect\_mask.py                        # Python script for webcam detection
└── README.md                             # Project documentation

````

---

## ⚙️ Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Face-Mask-Detection.git
cd Face-Mask-Detection
````

### 2️⃣ Install dependencies

```bash
pip install tensorflow opencv-python numpy
```

---

## ▶️ Usage

### 1️⃣ Run the detection script

```bash
python detect_mask.py
```

### 2️⃣ Controls

* **Press `q`** → Quit the webcam window.

---

## 🧠 Model Details

* Model: Custom CNN with Conv2D, MaxPooling, Dense layers
* Input size: **128×128×3**
* Normalization: Pixel values scaled to **\[0,1]**
* Labels:

  * `0` → Mask
  * `1` → No Mask

---

## 📜 How It Works

1. **Face Detection**: OpenCV Haar Cascade detects faces in the webcam feed.
2. **Preprocessing**: Extracted face is resized to `128×128` and normalized.
3. **Prediction**: The CNN model classifies it as **Mask** or **No Mask**.
4. **Display**: The result is shown with a bounding box and label.

---

## 📌 Requirements

* Python 3.7+
* TensorFlow 2.x
* OpenCV
* NumPy

---

## 🏗️ Future Improvements

* Train with more diverse datasets for higher accuracy
* Use MobileNetV2 for better performance on low-end devices
* Deploy as a web or mobile application

---

## 🙌 Acknowledgements

* [OpenCV](https://opencv.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Kaggle Dataset: Face Mask Detection](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

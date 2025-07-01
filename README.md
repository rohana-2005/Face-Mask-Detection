# 🛡️ Face Mask Detection (Machine Learning)

A real-time Face Mask Detection system built using **Random Forest Classifier** with **OpenCV** and **scikit-learn**. This project classifies faces as **with mask** 😷 or **without mask** 😐 using webcam input.

## 📁 Project Structure
```
face-mask-detection/
│
├── dataset/
│   ├── with_mask/         # Images with mask
│   └── without_mask/      # Images without mask
│
├── detect_mask_webcam.py  # Webcam-based real-time detection
├── train_model.py         # Trains Random Forest model
├── face_mask_model.pkl    # Trained ML model (optional, can be regenerated)
├── scaler.pkl             # Feature scaler (optional)
├── requirements.txt       # Dependencies
└── README.md              # Project overview
```

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧠 Training the Model
Run the following script to train the model on the image dataset:
```bash
python train_model.py
```
This will create:
- `face_mask_model.pkl`: the trained Random Forest model  
- `scaler.pkl`: the fitted feature scaler

## 📷 Real-time Mask Detection
Use your webcam to test detection in real time:
```bash
python detect_mask_webcam.py
```
The script opens your webcam and predicts whether the person is wearing a mask.

## 📈 Accuracy
- Initial model achieved **~80% accuracy** with Random Forest.
- Accuracy can be improved using:
  - CNNs or transfer learning
  - Data augmentation
  - Better preprocessing

## ✅ Requirements
- Python 3.7+
- scikit-learn
- OpenCV
- numpy

## 📌 Notes
- Only a few sample images are included in `dataset/` for demonstration.
- You can expand the dataset for better accuracy.
- If `face_mask_model.pkl` or `scaler.pkl` is missing, re-run `train_model.py`.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgements
Inspired by the need for mask compliance during the COVID-19 pandemic. Built for educational purposes using simple ML techniques.

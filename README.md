
## 🛠️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/kiranshinde24/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Install dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install opencv-python mediapipe tensorflow numpy
```

### 3. Add the model

Place your trained MobileNet model file inside the `model/` folder and name it:

```
asl_model.h5
```

### 4. Run the project

```bash
python main.py
```

> Press `q` to quit the webcam window.

---

## 🔍 How It Works

* Webcam captures frames using OpenCV.
* Mediapipe detects hand landmarks in real-time.
* The hand region is extracted and resized to 224x224.
* The image is normalized and passed to the MobileNet model.
* The model predicts the ASL letter (A–Z).
* The output is displayed on the frame in real time.

---

## 📦 Dependencies

Listed in `requirements.txt`:

```
opencv-python  
mediapipe  
tensorflow  
numpy
```

---

## ✍️ Author

**Kiran Shinde**
GitHub: [kiranshinde24](https://github.com/kiranshinde24)

---

## 📄 License

This project is open source and available under the **MIT License**.

---

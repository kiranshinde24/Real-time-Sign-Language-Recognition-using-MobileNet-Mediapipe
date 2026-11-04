# Real-Time Sign Language Recognition (MobileNet + MediaPipe)

Detects and classifies American Sign Language (ASL) hand signs (A–Z) from a webcam feed in real time. MediaPipe tracks the hand, and a MobileNetV2-based classifier predicts the letter.

<!-- Add a screenshot or GIF here, e.g. ![Demo](Screenshots/your-file.png) -->

## Results

Dataset: [ASL Alphabet (Kaggle, grassknoted)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), using the 26 letter classes (`nothing`, `space` and `del` removed). 62,400 training and 15,600 validation images at 224×224 (80/20 split).

**Backbone comparison** (frozen ImageNet base, same classifier head, 10 epochs, see `Sign_Language.ipynb`):

| Model | Validation accuracy |
|-------|--------------------|
| EfficientNetB0 | 3.85% (did not learn) |
| ResNet50 | 19.60% |
| **MobileNetV2** | **84.29%** (train 97.0%) |

**Shipped model (`asl_model.h5`):** MobileNetV2 + GlobalAveragePooling + Dropout(0.5) + Dense(128) + softmax(26), trained in `Untitled0.ipynb`. Its saved training history shows roughly 81% validation accuracy.

> The validation set is a random split of the same dataset (same signers and backgrounds), so these numbers are optimistic for real webcams.

## How It Works

1. OpenCV captures webcam frames.
2. MediaPipe detects hand landmarks in real time.
3. The hand region is cropped and resized to 224×224.
4. The image is normalized and passed to the MobileNetV2 model.
5. The predicted letter is drawn on the frame.

## Project Structure

```
├── realtime_asl.py       # real-time webcam inference (run locally)
├── asl_model.h5          # trained model
├── Sign_Language.ipynb   # backbone comparison (EfficientNetB0 / ResNet50 / MobileNetV2)
├── Untitled0.ipynb       # training run that produced asl_model.h5
├── deploy/               # Gradio app for Hugging Face Spaces
├── requirements.txt
└── Screenshots/
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kiranshinde24/Real-time-Sign-Language-Recognition-using-MobileNet-Mediapipe.git
cd Real-time-Sign-Language-Recognition-using-MobileNet-Mediapipe
```

### 2. Install dependencies

Use a virtual environment (Python 3.10–3.12).

```bash
pip install -r requirements.txt
```

Versions are pinned on purpose: the newest `mediapipe` releases no longer include the `mp.solutions` API this project uses.

### 3. Run

`asl_model.h5` is already in the repo root, so no extra setup is needed.

```bash
python realtime_asl.py
```

Press `q` to quit the webcam window.

## Live Demo

A browser version lives in [`deploy/`](deploy/) (Gradio app for Hugging Face Spaces).

<!-- Paste your Hugging Face Space link here after deploying -->

## Limitations

- **Lighting:** poor or uneven lighting reduces accuracy.
- **Background:** cluttered backgrounds or skin-toned objects can confuse hand detection.
- **J and Z:** these letters involve motion, and a single-frame classifier cannot capture it, so they are unreliable.
- **Single hand, static signs:** only one hand is tracked, and there is no word or sentence recognition.

## Future Work

- Sequence model (LSTM or temporal CNN) to handle J and Z
- Train on more varied backgrounds and lighting
- Word and sentence formation

## Author

**Kiran Shinde** · [GitHub](https://github.com/kiranshinde24)

## License

Released under the [MIT License](LICENSE).

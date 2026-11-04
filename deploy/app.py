import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model("asl_model.h5")
class_names = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)


def predict(frame):
    if frame is None:
        return None, "No frame"

    out = frame.copy()
    h, w, _ = frame.shape
    results = hands.process(frame)  # Gradio frames are already RGB

    if not results.multi_hand_landmarks:
        return out, "No hand detected"

    lms = results.multi_hand_landmarks[0].landmark
    margin = 30
    x_min = max(0, int(min(lm.x for lm in lms) * w) - margin)
    y_min = max(0, int(min(lm.y for lm in lms) * h) - margin)
    x_max = min(w, int(max(lm.x for lm in lms) * w) + margin)
    y_max = min(h, int(max(lm.y for lm in lms) * h) + margin)

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return out, "No hand detected"

    crop = cv2.resize(crop, (224, 224))
    x = np.expand_dims(crop, axis=0) / 255.0
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    label = f"{class_names[idx]} ({pred[idx]:.0%})"

    cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(out, class_names[idx], (x_min, max(20, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return out, label


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=[gr.Image(label="Detection"), gr.Textbox(label="Predicted sign")],
    live=True,
    title="Real-Time ASL Recognition",
    description="Show an ASL letter (A-Z) to your webcam. J and Z involve motion and are unreliable.",
)

if __name__ == "__main__":
    demo.launch()

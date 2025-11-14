import tensorflow as tf
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === CONFIG ===
MODEL_PATH = "model_unquant.tflite"

# CHANGE THESE LABELS to match your Teachable Machine classes:
CLASS_NAMES = ["Normal", "Abnormal"]   # <-- EDIT THIS BASED ON YOUR MODEL

# === FILE PICKER ===
Tk().withdraw()
image_path = askopenfilename(title="Select an image")
print("Selected:", image_path)

# === LOAD + PREPROCESS IMAGE ===
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (224, 224))
input_data = np.expand_dims(image_resized.astype(np.float32) / 255.0, axis=0)

# === LOAD TFLITE MODEL ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === RUN INFERENCE ===
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# === GET CLASS RESULT ===
predicted_class = np.argmax(predictions)
label = CLASS_NAMES[predicted_class]
confidence = predictions[predicted_class]

print("\nPrediction:", label, f"({confidence:.2f})")

# === DRAW RESULT ON IMAGE ===
output_image = image.copy()
cv2.putText(output_image,
            f"{label} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if label == "Normal" else (0, 0, 255),
            2)

# Show result window
cv2.imshow("Prediction", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

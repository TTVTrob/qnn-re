import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import tensorflow as tf

# ---------------------------
# CLASS LABELS (YOUR MODEL)
# ---------------------------
CLASS_MAP = {
    0: ("xray_normal", "Normal"),
    1: ("xray_abnormal", "Abnormal"),
    2: ("mri_normal", "Normal"),
    3: ("mri+abnormal", "Abnormal"),
    4: ("ct_abnormal", "Abnormal"),
    5: ("ct_normal", "Normal"),
}

MODEL_PATH = "model_unquant.tflite"

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1]

# ---------------------------
# Modern UI App Window
# ---------------------------
root = tk.Tk()
root.title("Medical Scan Classifier")
root.geometry("650x850")
root.configure(bg="#121212")

# Smooth shadow background
shadow = tk.Frame(root, bg="#000000", width=520, height=650)
shadow.place(x=65, y=80)

# Main white card
card = tk.Frame(root, bg="#1f1f1f", width=520, height=650)
card.place(x=60, y=60)

# Title
title_label = tk.Label(root, text="Medical Scan Classifier",
                       font=("Segoe UI", 26, "bold"),
                       bg="#121212", fg="white")
title_label.pack(pady=20)

# Image Display
img_label = tk.Label(card, bg="#1f1f1f")
img_label.place(relx=0.5, y=160, anchor="center")

# Prediction label
result_label = tk.Label(card, text="Upload a scan",
                        font=("Segoe UI", 18),
                        bg="#1f1f1f",
                        fg="#bbbbbb")
result_label.place(relx=0.5, y=350, anchor="center")

# Detailed result
details_label = tk.Label(card, text="",
                         font=("Segoe UI", 14),
                         bg="#1f1f1f",
                         fg="#bbbbbb")
details_label.place(relx=0.5, y=400, anchor="center")

# ---------------------------
# Prediction Logic
# ---------------------------
def classify_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    array = np.array(img).astype(np.float32)
    array = np.expand_dims(array / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    full_label, simplified = CLASS_MAP[idx]
    return simplified, full_label, confidence


# ---------------------------
# Image Upload Button Action
# ---------------------------
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
    if not file_path:
        return

    # Show image
    img = Image.open(file_path).resize((380, 380))
    tk_img = ImageTk.PhotoImage(img)
    img_label.configure(image=tk_img)
    img_label.image = tk_img

    # Predict
    simplified, full_label, conf = classify_image(file_path)

    # Color based on normal/abnormal
    color = "#4CAF50" if simplified == "Normal" else "#FF5252"

    result_label.config(
        text=f"{simplified}",
        fg=color
    )

    details_label.config(
        text=f"{full_label}\nConfidence: {conf:.2f}%",
        fg=color
    )


# ---------------------------
# Beautiful Gradient Button
# ---------------------------
def gradient_button(master, text, command):
    btn = tk.Canvas(master, width=200, height=60, bg="#1f1f1f", highlightthickness=0)
    btn.place(relx=0.5, y=520, anchor="center")

    # Gradient rectangle
    btn.create_rectangle(0, 0, 200, 60, fill="", outline="")

    # Text
    btn.create_text(100, 30, text=text, font=("Segoe UI", 18), fill="white")

    # Click event
    def on_click(event):
        command()

    btn.bind("<Button-1>", on_click)

    return btn


gradient_button(card, "Select Image", upload_image)

root.mainloop()

"""
app.py — Flask web app for Crack Detection
Run: python app.py
Then open: http://localhost:5000
"""

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, url_for
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import base64
from io import BytesIO
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = 'crack_unet_best.pth'   # put your model file next to app.py
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_MEAN  = (0.485, 0.456, 0.406)
IMAGE_STD   = (0.229, 0.224, 0.225)
THRESHOLD   = 0.5
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# ─────────────────────────────────────────────
def load_model():
    model = smp.Unet(
        encoder_name    = 'resnet34',
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f'Model loaded on {DEVICE}')
    return model

model = load_model()

TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    # Load image
    image_bgr = cv2.imread(image_path)
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess
    tensor = TRANSFORM(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output   = model(tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        mask     = (prob_map > THRESHOLD).astype(np.uint8) * 255

    # Resize back to original size
    mask_full = cv2.resize(mask, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)

    # Create red overlay
    overlay = image_rgb.copy()
    overlay[mask_full > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    # Crack coverage
    crack_pct = round((mask_full > 0).sum() / mask_full.size * 100, 2)

    # Severity level
    if crack_pct < 5:
        severity = 'Low'
        sev_color = 'green'
    elif crack_pct < 15:
        severity = 'Medium'
        sev_color = 'amber'
    else:
        severity = 'High'
        sev_color = 'red'

    # Convert images to base64 for sending to browser
    def to_base64(img_array):
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        buffer  = BytesIO()
        img_pil.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Prob map as heatmap
    prob_colored = cv2.applyColorMap(
        (prob_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    prob_rgb = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)

    return {
        'original'   : to_base64(image_rgb),
        'probability': to_base64(prob_rgb),
        'overlay'    : to_base64(blended),
        'crack_pct'  : crack_pct,
        'severity'   : severity,
        'sev_color'  : sev_color,
        'width'      : orig_w,
        'height'     : orig_h,
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400

    # Save uploaded file
    filename  = str(uuid.uuid4()) + '.' + \
                file.filename.rsplit('.', 1)[1].lower()
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Run prediction
    result = predict_image(save_path)

    # Clean up uploaded file
    os.remove(save_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

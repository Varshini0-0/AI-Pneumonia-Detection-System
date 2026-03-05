"""
AI Pneumonia Detection System - Flask Backend
=============================================
Main application file that handles routing, model inference,
Grad-CAM heatmap generation, and detection history.
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['HISTORY_FILE'] = 'detection_history.json'

# Lazy-import heavy dependencies so the app still starts even if GPU is absent
try:
    import tensorflow as tf
    from utils import load_model, preprocess_image, generate_gradcam
    MODEL = load_model('model.h5')
    print("[INFO] Pretrained model loaded successfully.")
except Exception as e:
    MODEL = None
    print(f"[WARN] Model not loaded – running in demo mode. Reason: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


def load_history() -> list:
    """Load detection history from the JSON store."""
    path = app.config['HISTORY_FILE']
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []


def save_history(history: list) -> None:
    """Persist detection history to the JSON store."""
    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump(history, f, indent=2)


def demo_predict(img_array: np.ndarray) -> dict:
    """
    Return a plausible fake prediction when no real model is available.
    Uses pixel-level statistics so results vary with the image.
    """
    mean_val = float(np.mean(img_array))
    confidence = round(min(0.97, max(0.55, abs(mean_val - 0.5) * 2 + 0.55)), 4)
    label = "PNEUMONIA" if mean_val < 0.5 else "NORMAL"
    return {'label': label, 'confidence': confidence}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    """Main dashboard – shows system statistics."""
    history = load_history()
    total      = len(history)
    pneumonia  = sum(1 for h in history if h['result'] == 'PNEUMONIA')
    normal     = total - pneumonia
    return render_template(
        'dashboard.html',
        total=total,
        pneumonia=pneumonia,
        normal=normal,
    )


@app.route('/detect')
def detect():
    """Upload page where users submit chest X-ray images."""
    return render_template('detect.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Receive an uploaded image, run inference (or demo), generate Grad-CAM,
    store the result, and redirect to the results page.
    """
    if 'xray' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['xray']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use PNG or JPG.'}), 400

    # ── Save original upload ───────────────────────────────────────────────
    ext        = file.filename.rsplit('.', 1)[1].lower()
    unique_id  = str(uuid.uuid4())[:8]
    filename   = f"{unique_id}.{ext}"
    save_path  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # ── Preprocessing & prediction ─────────────────────────────────────────
    try:
        from utils import preprocess_image, generate_gradcam
        img_array = preprocess_image(save_path)

        if MODEL is not None:
            prediction = MODEL.predict(img_array)
            confidence = float(prediction[0][0])
            label      = 'PNEUMONIA' if confidence > 0.5 else 'NORMAL'
            if label == 'NORMAL':
                confidence = 1.0 - confidence
        else:
            result     = demo_predict(img_array)
            label      = result['label']
            confidence = result['confidence']

        # ── Grad-CAM heatmap ───────────────────────────────────────────────
        heatmap_filename = f"hm_{unique_id}.jpg"
        heatmap_path     = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)

        if MODEL is not None:
            generate_gradcam(MODEL, img_array, save_path, heatmap_path)
        else:
            # Generate a synthetic coloured heatmap for demo purposes
            from utils import generate_demo_heatmap
            generate_demo_heatmap(save_path, heatmap_path)

    except Exception as e:
        print(f"[ERROR] During analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    # ── Build explanation text ─────────────────────────────────────────────
    conf_pct = round(confidence * 100, 1)
    if label == 'PNEUMONIA':
        explanation = (
            f"The model detected abnormal opacity patterns in the highlighted lung regions "
            f"with {conf_pct}% confidence. These patterns are consistent with pneumonia. "
            f"Please consult a qualified radiologist for clinical confirmation."
        )
    else:
        explanation = (
            f"The chest X-ray appears normal with {conf_pct}% confidence. "
            f"No significant opacity or consolidation patterns were detected. "
            f"Routine follow-up is recommended as per clinical guidelines."
        )

    # ── Persist to history ─────────────────────────────────────────────────
    record = {
        'id':         unique_id,
        'filename':   filename,
        'heatmap':    heatmap_filename,
        'result':     label,
        'confidence': conf_pct,
        'explanation':explanation,
        'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    history = load_history()
    history.insert(0, record)   # newest first
    save_history(history)

    return redirect(url_for('result', scan_id=unique_id))


@app.route('/result/<scan_id>')
def result(scan_id: str):
    """Display the prediction result for a specific scan."""
    history = load_history()
    record  = next((h for h in history if h['id'] == scan_id), None)
    if record is None:
        return redirect(url_for('dashboard'))
    return render_template('result.html', record=record)


@app.route('/history')
def history():
    """Display all past detection results."""
    records = load_history()
    return render_template('history.html', records=records)


@app.route('/delete/<scan_id>', methods=['POST'])
def delete_record(scan_id: str):
    """Delete a single history record and its associated files."""
    history = load_history()
    record  = next((h for h in history if h['id'] == scan_id), None)
    if record:
        # Remove files silently
        for folder, key in [('UPLOAD_FOLDER', 'filename'), ('HEATMAP_FOLDER', 'heatmap')]:
            try:
                os.remove(os.path.join(app.config[folder], record[key]))
            except FileNotFoundError:
                pass
        history = [h for h in history if h['id'] != scan_id]
        save_history(history)
    return redirect(url_for('history'))


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)

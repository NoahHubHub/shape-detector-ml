"""
server.py — Flask Web-Server (Render.com kompatibel)

Stellt die Web-App bereit und empfängt Kamera-Frames vom Handy.
Für jeden Frame:
  1. Base64-Bild dekodieren
  2. Features extrahieren (features.py)
  3. Decision Tree klassifiziert die Form
  4. Ergebnis als JSON zurück ans Handy
"""

import os
import cv2
import numpy as np
import base64
import joblib

from flask import Flask, request, jsonify, render_template
from features import extract_features

app = Flask(__name__)

# Modell laden (muss vorher train.py ausgeführt worden sein)
try:
    modell = joblib.load('model/model.pkl')
    print("✓ Modell geladen: model/model.pkl")
except FileNotFoundError:
    print("✗ Kein Modell gefunden! Bitte zuerst: python train.py")
    exit(1)

# Farben pro Form (BGR für OpenCV, wird als Hex ans Frontend gesendet)
FORM_FARBEN = {
    'Kreis':    '#6bffb8',
    'Dreieck':  '#ff6b6b',
    'Quadrat':  '#ffd93d',
    'Rechteck': '#ffa500',
    'Fünfeck':  '#a29bfe',
    'Sechseck': '#74b9ff',
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Nimmt ein Kamera-Bild als Base64-String entgegen,
    erkennt Formen darin und gibt die Ergebnisse als JSON zurück.
    """
    daten = request.json
    if not daten or 'image' not in daten:
        return jsonify({'formen': []})

    # Base64 → NumPy-Array → OpenCV-Bild
    try:
        img_bytes = base64.b64decode(daten['image'].split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'formen': []})

    if img is None:
        return jsonify({'formen': []})

    # Vorverarbeitung: Graustufen → Blur → Kanten
    grau = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grau, (5, 5), 0)
    kanten = cv2.Canny(blur, 50, 150)

    # Konturen schließen
    kernel = np.ones((3, 3), np.uint8)
    kanten = cv2.dilate(kanten, kernel)

    konturen, _ = cv2.findContours(kanten, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    erkannte_formen = []

    for kontur in konturen:
        features = extract_features(kontur)
        if features is None:
            continue

        # Decision Tree klassifiziert die Form
        vorhersage = modell.predict([features])[0]
        wahrscheinlichkeit = modell.predict_proba([features]).max()

        # Nur ausgeben wenn Konfidenz > 60%
        if wahrscheinlichkeit < 0.60:
            continue

        x, y, w, h = cv2.boundingRect(kontur)
        erkannte_formen.append({
            'name':       vorhersage,
            'konfidenz':  round(float(wahrscheinlichkeit) * 100),
            'farbe':      FORM_FARBEN.get(vorhersage, '#ffffff'),
            'bbox':       {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })

    return jsonify({'formen': erkannte_formen})


if __name__ == '__main__':
    # PORT wird von Render.com als Umgebungsvariable gesetzt
    port = int(os.environ.get('PORT', 5000))
    print(f"\nServer läuft auf Port {port}")
    print(f"  Lokal:  https://localhost:{port}")
    print(f"  Handy:  https://<deine-IP>:{port}  (gleiches WLAN)")
    app.run(host='0.0.0.0', port=port, debug=False, ssl_context='adhoc')

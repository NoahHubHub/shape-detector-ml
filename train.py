"""
train.py — Trainingsdaten generieren & Decision Tree trainieren

Ablauf:
  1. Synthetische Form-Bilder erzeugen (kein manuelles Labeln nötig)
  2. Features aus jedem Bild extrahieren (via features.py)
  3. Decision Tree mit scikit-learn trainieren
  4. Modell speichern (model/model.pkl)
  5. Baum als Bild visualisieren (model/decision_tree.png)

Ausführen: python train.py
"""

import matplotlib
matplotlib.use('Agg')  # Headless-Backend für Server ohne Display (z.B. Render)
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import joblib

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from features import extract_features

# ── Konfiguration ─────────────────────────────────────────────────────────────

FORMEN = ['Kreis', 'Dreieck', 'Quadrat', 'Rechteck', 'Fünfeck', 'Sechseck']
SAMPLES_PRO_FORM = 600   # Wie viele Trainingsbilder pro Form
IMG_SIZE = 200           # Bildgröße in Pixel

# ── Synthetische Bilder erzeugen ───────────────────────────────────────────────

def bild_erzeugen(form, groesse=IMG_SIZE):
    """
    Erzeugt ein synthetisches Bild einer geometrischen Form.
    Zufällige Variation bei Größe, Position und Rotation macht das Modell robuster.
    """
    img = np.ones((groesse, groesse), dtype=np.uint8) * 255  # weißer Hintergrund
    cx, cy = groesse // 2, groesse // 2
    r = int(groesse * 0.35)

    # Zufällige Variation für mehr Robustheit im Training
    offset_x = np.random.randint(-25, 25)
    offset_y = np.random.randint(-25, 25)
    r_var = int(r * np.random.uniform(0.65, 1.0))
    winkel = np.random.uniform(0, 2 * np.pi)

    if form == 'Kreis':
        cv2.circle(img, (cx + offset_x, cy + offset_y), r_var, 0, -1)

    elif form == 'Dreieck':
        pts = []
        for i in range(3):
            a = winkel + i * 2 * np.pi / 3
            pts.append([int(cx + offset_x + r_var * np.cos(a)),
                        int(cy + offset_y + r_var * np.sin(a))])
        cv2.fillPoly(img, [np.array(pts)], 0)

    elif form == 'Quadrat':
        cv2.rectangle(img,
                      (cx + offset_x - r_var, cy + offset_y - r_var),
                      (cx + offset_x + r_var, cy + offset_y + r_var), 0, -1)

    elif form == 'Rechteck':
        w = int(r_var * np.random.uniform(1.5, 2.2))
        h = int(r_var * np.random.uniform(0.45, 0.70))
        cv2.rectangle(img,
                      (cx + offset_x - w, cy + offset_y - h),
                      (cx + offset_x + w, cy + offset_y + h), 0, -1)

    elif form == 'Fünfeck':
        pts = []
        for i in range(5):
            a = winkel + i * 2 * np.pi / 5
            pts.append([int(cx + offset_x + r_var * np.cos(a)),
                        int(cy + offset_y + r_var * np.sin(a))])
        cv2.fillPoly(img, [np.array(pts)], 0)

    elif form == 'Sechseck':
        pts = []
        for i in range(6):
            a = winkel + i * 2 * np.pi / 6
            pts.append([int(cx + offset_x + r_var * np.cos(a)),
                        int(cy + offset_y + r_var * np.sin(a))])
        cv2.fillPoly(img, [np.array(pts)], 0)

    # Leichtes Blur simuliert echte Kamera-Unschärfe
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def datensatz_erstellen(n=SAMPLES_PRO_FORM):
    """
    Generiert den gesamten Trainingsdatensatz.
    Gibt X (Features) und y (Labels) zurück.
    """
    X, y = [], []

    for form in FORMEN:
        erfolge = 0
        versuche = 0
        while erfolge < n and versuche < n * 4:
            versuche += 1
            img = bild_erzeugen(form)

            # Gleiche Pipeline wie im Server: Canny → Konturen → Features
            edges = cv2.Canny(img, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            kontur = max(contours, key=cv2.contourArea)
            features = extract_features(kontur)

            if features is not None:
                X.append(features)
                y.append(form)
                erfolge += 1

        print(f"  {form:12s}: {erfolge} Samples")

    return np.array(X), np.array(y)


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  Form-Erkenner — Decision Tree Training")
    print("=" * 50)

    # 1. Daten generieren
    print(f"\n[1/4] Generiere {SAMPLES_PRO_FORM} Samples pro Form...")
    X, y = datensatz_erstellen()
    print(f"      Gesamt: {len(X)} Samples, {len(FORMEN)} Klassen")

    # 2. Aufteilen in Trainings- und Testdaten (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[2/4] Aufgeteilt: {len(X_train)} Training / {len(X_test)} Test")

    # 3. Decision Tree trainieren
    # max_depth begrenzt die Tiefe → verhindert Overfitting
    print("\n[3/4] Trainiere Decision Tree...")
    modell = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    modell.fit(X_train, y_train)

    # Ergebnis auf Testdaten
    print("\nGenauigkeit auf Testdaten:")
    print(classification_report(y_test, modell.predict(X_test), target_names=FORMEN))

    # 4. Modell speichern
    os.makedirs('model', exist_ok=True)
    joblib.dump(modell, 'model/model.pkl')
    print("[4/4] Modell gespeichert: model/model.pkl")

    # 5. Baum visualisieren
    merkmal_namen = ['Zirkularität', 'Seitenverhältnis', 'Ecken', 'Solidität', 'Füllgrad']
    plt.figure(figsize=(24, 12))
    plot_tree(modell,
              feature_names=merkmal_namen,
              class_names=FORMEN,
              filled=True,
              rounded=True,
              fontsize=8,
              impurity=False)
    plt.title("Decision Tree — Wie das Modell Formen erkennt", fontsize=14)
    plt.tight_layout()
    plt.savefig('model/decision_tree.png', dpi=120, bbox_inches='tight')
    print("    Baum-Visualisierung: model/decision_tree.png")
    print("\nFertig! Starte jetzt: python server.py")

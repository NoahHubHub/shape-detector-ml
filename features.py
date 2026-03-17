"""
features.py — Feature-Extraktion aus Konturen

Wird sowohl beim Training (train.py) als auch beim Server (server.py) verwendet.
Aus jeder erkannten Kontur werden 5 mathematische Merkmale extrahiert,
die der Decision Tree dann zur Klassifikation nutzt.
"""

import cv2
import numpy as np


def extract_features(contour):
    """
    Extrahiert 5 Merkmale aus einer Kontur:

    1. Zirkularität  → wie rund ist die Form? (1.0 = perfekter Kreis)
    2. Seitenverhältnis → Breite / Höhe der Bounding Box
    3. Eckenanzahl   → wie viele Ecken hat die Form?
    4. Solidität     → Füllung vs. konvexe Hülle (niedrig = Raute)
    5. Füllgrad      → wie viel der Bounding Box ist gefüllt?

    Gibt None zurück wenn die Kontur zu klein ist (Rauschen).
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Zu kleine Konturen ignorieren (Rauschen)
    if area < 500 or perimeter == 0:
        return None

    # --- Merkmal 1: Zirkularität ---
    # Formel: 4π * Fläche / Umfang²
    # Kreis = 1.0, je eckiger desto kleiner
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # --- Merkmal 2: Seitenverhältnis ---
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0

    # --- Merkmal 3: Eckenanzahl ---
    # approxPolyDP vereinfacht die Kontur auf die wichtigsten Punkte
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)

    # --- Merkmal 4: Solidität ---
    # Fläche der Form geteilt durch Fläche der konvexen Hülle
    # Rauten / hohle Formen haben niedrige Solidität
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # --- Merkmal 5: Füllgrad ---
    # Wie viel Prozent der Bounding Box ist ausgefüllt?
    extent = area / (w * h) if (w * h) > 0 else 0

    return [circularity, aspect_ratio, vertices, solidity, extent]

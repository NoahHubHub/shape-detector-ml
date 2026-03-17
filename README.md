# Form Erkenner — ML (Decision Tree + Flask)

Erkennt geometrische Formen (Kreis, Dreieck, Quadrat, Rechteck, Fünfeck, Sechseck)
in Echtzeit über die Handy-Kamera. Python-Backend mit echtem ML (Decision Tree).

## Setup

```bash
pip install -r requirements.txt
```

## Schritt 1 — Modell trainieren

```bash
python train.py
```

Generiert synthetische Trainingsdaten, trainiert den Decision Tree und speichert:
- `model/model.pkl` — das trainierte Modell
- `model/decision_tree.png` — Visualisierung des Baums

## Schritt 2 — Server starten

```bash
python server.py
```

Der Server zeigt zwei URLs an:
- **Laptop:** `https://localhost:5000`
- **Handy:** `https://<deine-IP>:5000`

> Beim Öffnen erscheint eine Sicherheitswarnung (selbstsigniertes Zertifikat).
> Auf "Erweitert" → "Weiter zu ..." klicken.

## Architektur

```
Handy-Kamera (getUserMedia)
    ↓ JPEG Base64 (alle 300ms)
Flask Server /predict
    ↓ OpenCV: Graustufen → Blur → Canny → Konturen
    ↓ features.py: 5 Merkmale extrahieren
    ↓ Decision Tree: Vorhersage
    ↓ JSON: { name, konfidenz, bbox }
Handy-Browser (Canvas)
    → Rahmen + Label zeichnen
```

## ML-Features

| Merkmal        | Beschreibung                        | Kreis | Dreieck |
|----------------|-------------------------------------|-------|---------|
| Zirkularität   | 4π·Fläche / Umfang²                 | ~1.0  | ~0.6    |
| Seitenverhältnis | Breite / Höhe Bounding Box        | ~1.0  | ~1.0    |
| Eckenanzahl    | Ecken nach approxPolyDP             | >8    | 3       |
| Solidität      | Fläche / konvexe Hülle              | ~1.0  | ~1.0    |
| Füllgrad       | Fläche / Bounding Box               | ~0.8  | ~0.5    |

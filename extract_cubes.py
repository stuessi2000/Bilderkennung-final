#!/usr/bin/env python3
"""
extract_cubes.py

Erkennt Würfel auf Graustufen-Bildern, extrahiert die perspektivisch entzerrte Oberseite (Face)
und berechnet gleichzeitig den geometrischen Mittelpunkt jeder Würfelfläche im Originalbild.
Zusätzlich wird für jedes Bild ein Debug-Bild ausgegeben, auf dem:
  - alle 4-Ecken-Konturen in Grün eingezeichnet sind
  - alle Mittelpunkte als kleine, rote Kreise markiert sind

Die extrahierten Gesichter landen in "faces/", 
Debug-Bilder in "debug_cubes_output/" und 
alle Mittelpunkte in "cubes_centers.csv".
"""

import os
import glob
import cv2
import numpy as np
import csv

# === Einstellungen ===
PNG_DIR             = "png_cubes"    # Ordner mit Graustufen-Bildern
FACES_DIR           = "faces"            # Ordner für die extrahierten Würfel-Oberseiten
DEBUG_DIR           = "debug_cubes_output"  # Ordner für die Debug-Bilder
TARGET_W            = 200               # Pixel-Breite der entzerrten Face
TARGET_H            = 200                # Pixel-Höhe der entzerrten Face
MIN_AREA            = 3500               # Min. Konturfläche in px² (um Rauschen zu filtern)
EPS_FACTOR          = 0.02               # Epsilon-Wert für approxPolyDP (Kontur-Glättung)
THRESH_VAL          = 30                 # Threshold zum Binarisieren (0–255)
# =======================

# Ordner anlegen, falls nicht vorhanden
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Liste, um später alle Mittelpunkte zu speichern
cube_centers = []

# Schleife über alle PNGs in PNG_DIR
for img_path in sorted(glob.glob(os.path.join(PNG_DIR, "*.png"))):
    # 1) Graustufenbild laden
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"⚠ Kann {img_path} nicht laden.")
        continue

    # Erzeuge ein farbiges Debug-Bild (BGR), auf dem wir alle Konturen + Zentren einzeichnen
    debug_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # 2) Binärbild erstellen: Matte ist schwarz (0), Würfel hell (255)
    _, thresh = cv2.threshold(img_gray, THRESH_VAL, 255, cv2.THRESH_BINARY)

    # 3) Konturen extrahieren (äußere Konturen)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    basename = os.path.splitext(os.path.basename(img_path))[0]
    face_idx = 0

    # Durchlaufe jede gefundene Kontur
    for cnt in cnts:
        # 4) Filtern nach Mindestfläche
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        # 5) Kontur approximieren und prüfen, ob 4 Ecken
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPS_FACTOR * peri, True)
        if len(approx) != 4:
            continue

        # 6) Eckpunkte sortieren: tl, tr, br, bl
        pts = approx.reshape(4, 2)
        s   = pts.sum(axis=1)
        diff= np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        src = np.array([tl, tr, br, bl], dtype="float32")

        # 7) Zielrechteck definieren
        dst = np.array([
            [0,        0],
            [TARGET_W, 0],
            [TARGET_W, TARGET_H],
            [0,        TARGET_H]
        ], dtype="float32")

        # 8) Perspektivisch entzerren
        M    = cv2.getPerspectiveTransform(src, dst)
        face = cv2.warpPerspective(img_gray, M, (TARGET_W, TARGET_H))

        # 9) Face-Bild speichern
        out_name = f"{basename}_face{face_idx:02d}.png"
        out_path = os.path.join(FACES_DIR, out_name)
        cv2.imwrite(out_path, face)
        print(f"✓ {out_name}")

        # 10) Mittelpunkt der Kontur im Originalbild berechnen (Momente)
        M_mom = cv2.moments(cnt)
        if M_mom["m00"] != 0:
            cx = int(M_mom["m10"] / M_mom["m00"])
            cy = int(M_mom["m01"] / M_mom["m00"])
        else:
            # Fallback: Bounding-Rect-Center, falls Fläche ≈ 0 (sollte kaum passieren)
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

        # 11) In cube_centers-Liste aufnehmen
        cube_centers.append({
            "FaceImage":   out_name,
            "Center_X_px": cx,
            "Center_Y_px": cy
        })

        # 12) Debug-Visualisierung auf dem EINZIGEN Debug-Bild: 
        #     - Kontur-Grün (Linienstärke 2)
        #     - Mittelpunkt-Rot (Kreis mit Radius 5, gefüllt)
        cv2.drawContours(debug_color, [approx], -1, (0, 255, 0), 2)
        cv2.circle(debug_color, (cx, cy), 5, (0, 0, 255), thickness=-1)

        face_idx += 1

    # 13) EINZIGES Debug-Bild pro Bilddatei speichern
    if face_idx > 0:
        debug_name = f"{basename}_debug.png"
        debug_path = os.path.join(DEBUG_DIR, debug_name)
        cv2.imwrite(debug_path, debug_color)
        print(f"📝 Debug gespeichert: {debug_name}")

    # Falls keine Face extrahiert wurde, Ausgabe
    if face_idx == 0:
        print(f"— Keine 4-Ecke gefunden in {basename}.png\n")
    else:
        print(f"→ {face_idx} Faces aus {basename}.png extrahiert\n")

# 14) Nach aller Extraktion: Speichere die Mittelpunkte in „cubes_centers.csv"
csv_path = "cubes_centers.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["FaceImage", "Center_X_px", "Center_Y_px"])
    writer.writeheader()
    for row in cube_centers:
        writer.writerow(row)

print(f"✓ Mittelpunkte aller Würfel-Faces gespeichert in {csv_path}")
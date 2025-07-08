#!/usr/bin/env python3
import glob
import cv2

# === Hier definieren, was du im Extract-Script benutzt hast ===
TARGET_W = 200
TARGET_H = 200

# Verzeichnis, in dem deine Tiles liegen
TILES_DIR = "target_tiles"

# Alle PNG-Dateien einlesen
files = glob.glob(f"{TILES_DIR}/*.png")
mismatch = False

for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Konnte {f} nicht laden.")
        continue
    h, w = img.shape
    if (h, w) != (TARGET_H, TARGET_W):
        print(f"❌ Falsche Größe bei {f}: {h}×{w}, erwartet {TARGET_H}×{TARGET_W}")
        mismatch = True

if not mismatch:
    print(f"✅ Alle Tiles haben die korrekte Größe {TARGET_H}×{TARGET_W}.")

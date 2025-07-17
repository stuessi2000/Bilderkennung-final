#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np

# === Einstellungen ===
PNG_DIR      = "png_zielbild"             # Ordner mit deinem Ziel-PNG
DEBUG_DIR    = "debug_zielbild_output"    # Debug-Kanten- und Grid-Bilder
OUTPUT_DIR   = "extracted_target"         # Entzerrtes Vollbild
TILES_DIR    = "target_tiles"             # 3×3 Sub-Tiles

TARGET_W     = 160                        # Breite einer Kachel in Pixel
TARGET_H     = 160                        # Höhe einer Kachel in Pixel
GRID_ROWS    = 3
GRID_COLS    = 3

THRESH_VAL   = 15                         # Threshold auf 15 gesenkt
MIN_AREA     = 10000                      # Fläche etwas herabgesetzt fürs Erkennen
EPS_FACTOR   = 0.02                       # approxPolyDP: 2 % des Umfangs
# ======================

# Ordner erstellen (falls noch nicht vorhanden)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)

# (1) Lade erstes Bild im PNG_DIR (hier *.png)
all_pngs = sorted(glob.glob(os.path.join(PNG_DIR, "*.png")))
if not all_pngs:
    print("❌ Kein Zielbild gefunden in", PNG_DIR)
    exit(1)

zielbild_path = all_pngs[0]
basename      = os.path.splitext(os.path.basename(zielbild_path))[0]
print(f"🎯 Zielbild geladen: {zielbild_path}")

# (2) Graustufen einlesen und Debug-Canvas anlegen
img   = cv2.imread(zielbild_path, cv2.IMREAD_GRAYSCALE)
debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# (2a) Leichter Gauß-Blur, um Rauschen (z.B. Scanner-Körnung) etwas zu glätten
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# (3) Thresholding (hellere Pixel > THRESH_VAL → 255; sonst 0)
_, th = cv2.threshold(img_blur, THRESH_VAL, 255, cv2.THRESH_BINARY)
# Speichere das Threshold-Bild zum Debuggen
cv2.imwrite(os.path.join(DEBUG_DIR, f"{basename}_threshold.png"), th)
print(f"📝 Threshold-Bild gespeichert: {os.path.join(DEBUG_DIR, f'{basename}_threshold.png')}")

# (4) Morphologisches Closing, um kleine Löcher (Cartoonlinien) zu schließen
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
# Speichere das Closed-Bild zum Debuggen
cv2.imwrite(os.path.join(DEBUG_DIR, f"{basename}_closed.png"), closed)
print(f"📝 Closed-Bild gespeichert: {os.path.join(DEBUG_DIR, f'{basename}_closed.png')}")

# (5) Konturen finden
cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"🔎 {len(cnts)} Konturen insgesamt gefunden.")

# (6) In allen Konturen nach einem Viereck (4 Punkten) suchen
best_quad = None
best_area = 0

for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue

    peri = cv2.arcLength(cnt, True)
    # approx mit EPS_FACTOR (2 % des Perimeters)
    approx = cv2.approxPolyDP(cnt, EPS_FACTOR * peri, True)
    if len(approx) == 4 and area > best_area:
        best_quad = approx.copy()
        best_area = area

if best_quad is None:
    print("❌ Kein Viereck gefunden.")
    exit(1)

# (7) Debug: Zeichne das gefundene 4-Eck
cv2.drawContours(debug, [best_quad], -1, (0, 255, 0), 2)
dbg_path = os.path.join(DEBUG_DIR, f"{basename}_debug_quad.png")
cv2.imwrite(dbg_path, debug)
print(f"📝 Debug-Kontur (Viereck) gespeichert: {dbg_path}")

# (8) Eckpunkte sortieren: tl, tr, br, bl
pts  = best_quad.reshape(4, 2).astype("float32")
s    = pts.sum(axis=1)
diff = np.diff(pts, axis=1).ravel()
tl   = pts[np.argmin(s)]
br   = pts[np.argmax(s)]
tr   = pts[np.argmin(diff)]
bl   = pts[np.argmax(diff)]
src  = np.array([tl, tr, br, bl], dtype="float32")

# (9) Ziel-Auflösung für das Warp (3×3 Kacheln à 200×200px)
warp_w = TARGET_W * GRID_COLS
warp_h = TARGET_H * GRID_ROWS
dst = np.array([
    [0,         0],
    [warp_w-1,  0],
    [warp_w-1, warp_h-1],
    [0,        warp_h-1]
], dtype="float32")

# Homographie / Perspective-Transform
M    = cv2.getPerspectiveTransform(src, dst)
warp = cv2.warpPerspective(img, M, (warp_w, warp_h))

out_full = os.path.join(OUTPUT_DIR, f"{basename}_extracted.png")
cv2.imwrite(out_full, warp)
print(f"✓ Entzerrtes Zielbild gespeichert: {out_full}")

# (10) Gitter-Debug auf Entzerrtem bilden
grid_dbg = cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR)
for i in range(1, GRID_ROWS):
    y = i * TARGET_H
    cv2.line(grid_dbg, (0, y), (warp_w, y), (0, 0, 255), 2)
for j in range(1, GRID_COLS):
    x = j * TARGET_W
    cv2.line(grid_dbg, (x, 0), (x, warp_h), (0, 0, 255), 2)

grid_dbg_path = os.path.join(DEBUG_DIR, f"{basename}_grid_debug.png")
cv2.imwrite(grid_dbg_path, grid_dbg)
print(f"📝 Gitter-Debug gespeichert: {grid_dbg_path}")

# (11) 3×3 Sub-Tiles ausschneiden & speichern
print(f"Zerlege {basename} in {GRID_ROWS}×{GRID_COLS} Tiles:")
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        y0, y1 = r * TARGET_H, (r + 1) * TARGET_H
        x0, x1 = c * TARGET_W, (c + 1) * TARGET_W
        tile = warp[y0:y1, x0:x1]
        tile_fn = f"Ziel_{r}_{c}.png"
        tile_fp = os.path.join(TILES_DIR, tile_fn)
        cv2.imwrite(tile_fp, tile)
        print(f"✓ {tile_fn}")

print("✅ Fertig – alle Schritte abgeschlossen.")

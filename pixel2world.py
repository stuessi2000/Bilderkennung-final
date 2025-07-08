#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# --- Pfade ---
# 1) Hier eure ursprüngliche CSV mit Pixel‐Koordinaten:
INPUT_CSV = Path("cubes_centers.csv")  
#    (enthält Zeilen: FaceImage,Center_X_px,Center_Y_px)
# 2) Hier speichern wir die neue CSV mit mm‐Werten:
OUTPUT_CSV = Path("cubes_world_coords_simple.csv")

# --- Pixel‐Zu‐mm‐Skalierung ---
MM_PER_PIXEL = 0.125  # jeder Pixel entspricht 0.125 mm in X- und Y-Richtung

# --- Referenz: Pixeloffset und Welt‐Offset ---
# Wenn Pixel (0,0) in Welt (0,0) liegen soll:
u_offset = 0.0  
v_offset = 0.0  
X0 = 0.0  # Welt‐Koordinate in mm am Pixel‐Ursprung (0,0)
Y0 = 0.0  # Welt‐Koordinate in mm am Pixel‐Ursprung (0,0)
# Falls ihr stattdessen Pixel(die Bildmitte) als Referenz nehmen wollt, setzt z.B.:
# u_offset = image_width/2
# v_offset = image_height/2
# X0 = known_world_X_of_image_center
# Y0 = known_world_Y_of_image_center

# --- Hauptroutine ---
if __name__ == "__main__":
    # 1) Einlesen der CSV mit Pixel‐Mittelpunkten
    df_pix = pd.read_csv(INPUT_CSV)  
    #    Erwartetes Format:
    #    FaceImage, Center_X_px, Center_Y_px
    #    _Tag...face00.png, 612, 925
    #    ...

    world_coords = []
    for idx, row in df_pix.iterrows():
        face_name = row["FaceImage"]
        u_px = float(row["Center_X_px"])
        v_px = float(row["Center_Y_px"])
        
        # 2) Aus Pixel → Welt (mm) mit linearer Skalierung:
        #    X = X0 + (u_px - u_offset)*MM_PER_PIXEL
        #    Y = Y0 + (v_px - v_offset)*MM_PER_PIXEL
        Xw = X0 + (u_px - u_offset) * MM_PER_PIXEL
        Yw = Y0 + (v_px - v_offset) * MM_PER_PIXEL
        Zw = 0.0  # Tischoberfläche / Würfeloberseite als Z=0 in Robotersystem
        
        world_coords.append({
            "FaceImage": face_name,
            "X_mm": round(Xw, 3),
            "Y_mm": round(Yw, 3),
            "Z_mm": round(Zw, 3)
        })

    # 3) Speichern in neue CSV
    df_world = pd.DataFrame(world_coords)
    df_world.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Weltkoordinaten (mm) gespeichert in '{OUTPUT_CSV}'.")

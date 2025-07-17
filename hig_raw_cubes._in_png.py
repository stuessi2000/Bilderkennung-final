import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Einstellungen ===
hig_dir = r"\\192.168.0.1\Vision_Daten\DU_3\Würfelbilder"             # hier liegen deine .hig-Dateien
png_dir = r"png_cubes"  # PNGs werden im lokalen Ordner png_cubes gespeichert
width, height   = 1602, 1212
expected_pixels = width * height

print(f"Starte Konvertierung: Suche .hig-Dateien in {hig_dir}")
try:
    files = sorted(os.listdir(hig_dir))
    print(f"Gefundene Dateien: {len(files)}")
except Exception as e:
    print(f"Fehler beim Zugriff auf {hig_dir}: {e}")
    files = []

# === HIG zu PNG konvertieren ===
os.makedirs(png_dir, exist_ok=True)
for fname in files:
    if fname.endswith(".hig"):
        path = os.path.join(hig_dir, fname)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            print(f"Fehler beim Lesen von {path}: {e}")
            continue

        if len(data) < expected_pixels:
            print(f"❌ Too few bytes in {fname} ({len(data)} bytes)")
            continue

        try:
            y_plane = np.frombuffer(data[:expected_pixels], dtype=np.uint8).reshape((height, width))
            # Nur Spiegelung an der X-Achse (horizontal)
            y_plane = cv2.flip(y_plane, 1)  # Spiegelung an X-Achse (horizontal)
        except Exception as e:
            print(f"Fehler beim Umwandeln von {fname}: {e}")
            continue

        # Dateinamen für Windows-Dateisystem bereinigen
        safe_name = os.path.splitext(fname)[0]
        for ch in [':', ' ', '\\', '/', '?', '*', '"', '<', '>', '|']:
            safe_name = safe_name.replace(ch, '_')
        out_name = safe_name + ".png"
        out_path = os.path.join(png_dir, out_name)
        try:
            success = cv2.imwrite(out_path, y_plane)
            if success:
                print(f"✓ {fname} → {out_path}")
            else:
                print(f"❌ Fehler beim Speichern von {out_path}")
        except Exception as e:
            print(f"Fehler beim Schreiben von {out_path}: {e}")


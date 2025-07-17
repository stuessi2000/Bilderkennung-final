import os
import cv2
import numpy as np

# === Einstellungen ===
hig_dir         = r"\\192.168.0.1\Vision_Daten\DU_3\Zielbild"
png_dir = r"png_zielbild"
width, height   = 1602, 1212
expected_pixels = width * height

os.makedirs(png_dir, exist_ok=True)

for fname in sorted(os.listdir(hig_dir)):
    if fname.endswith(".hig"):
        path = os.path.join(hig_dir, fname)
        with open(path, "rb") as f:
            data = f.read()

        actual_length = len(data)
        if actual_length < expected_pixels:
            print(f"⚠️  {fname}: too short ({actual_length}), padding with zeros.")
            data = data + b'\x00' * (expected_pixels - actual_length)
        elif actual_length > expected_pixels:
            print(f"⚠️  {fname}: too long ({actual_length}), cropping extra bytes.")
            data = data[:expected_pixels]

        y_plane = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        # Nach dem Konvertieren an der X-Achse spiegeln
        y_plane = cv2.flip(y_plane, 1)  # 1 = horizontal (X-Achse)
        # Spiegelung an der X-Achse (horizontal) hinzufügen
        # y_plane = cv2.flip(y_plane, 1)  # 1 = horizontal (X-Achse)
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(png_dir, out_name)
        cv2.imwrite(out_path, y_plane)
        print(f"✓ {fname} → {out_path}")

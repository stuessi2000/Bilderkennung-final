import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
#import matplotlib.pyplot as plt
import time
import shutil

# —————————————————————————————
# Adaptive Parameter-Funktionen mit Grenzen
# —————————————————————————————
def estimate_noise_sigma(img_gray):
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    noise = img_gray.astype(np.float32) - blur.astype(np.float32)
    return np.std(noise)


def compute_clip_limit(img_gray, base=2.0, min_clip=1.0, max_clip=3.0):
    """
    Adaptive Clip-Limit basierend auf Histogramm-Ausbreitung,
    begrenzt auf [min_clip, max_clip].
    """
    hist = cv2.calcHist([img_gray], [0], None, [256], [0,256]).ravel()
    spread = np.count_nonzero(hist) / 256.0
    clip = base * spread
    return float(np.clip(clip, min_clip, max_clip))


def compute_denoise_h(img_gray, scale=2.0, min_h=5.0, max_h=20.0):
    """
    Berechnet Denoising-Parameter h = scale * sigma (Rausch-Std),
    begrenzt auf [min_h, max_h].
    """
    sigma = estimate_noise_sigma(img_gray)
    h = scale * sigma
    return float(np.clip(h, min_h, max_h))


def compute_tile_grid_size(img_gray, min_grid=4, max_grid=16):
    """
    Dynamische CLAHE-Kachelgröße: etwa 1 Kachel / 20 px Bildhöhe,
    begrenzt in [min_grid, max_grid].
    """
    h, w = img_gray.shape
    grid = int(np.clip(min(h, w) // 20, min_grid, max_grid))
    return (grid, grid)

# —————————————————————————————
# 1) Pfade & Ordner anlegen
# —————————————————————————————
#BASE_DIR     = Path(__file__).resolve().parent
#template_dir = Path(r"D:/Leon/target_tiles")
#face_dir     = Path(r"D:/Leon/Würfel")
#vis_dir      = Path(r"D:/Leon/Sift_results_new/matched_visuals")
#plot_dir     = Path(r"D:/Leon/Sift_results_new/plots")
#csv_path     = BASE_DIR / "matching_results.csv"
template_dir = Path("target_tiles")
face_dir     = Path("faces")
vis_dir      = Path("sift_vis")
# csv_path     = Path(r"D:/Leon/Koordinaten_template_matching/sift_match_results.csv")
# log_path     = Path(r"D:/Leon/Koordinaten_template_matching/sift_debug_log.csv")
vis_dir.mkdir(parents=True, exist_ok=True)

# Ordner anlegen
#vis_dir.mkdir(parents=True, exist_ok=True)
#plot_dir.mkdir(parents=True, exist_ok=True)

# unterstützte Bildformate
image_exts = ["*.png", "*.jpg", "*.jpeg"]

def gather_image_paths(directory):
    paths = []
    for ext in image_exts:
        paths.extend(directory.glob(ext))
    return sorted(paths)

templates = gather_image_paths(template_dir)
faces     = gather_image_paths(face_dir)
if not templates or not faces:
    print("Keine Bilder gefunden: Templates oder Faces nicht vorhanden.")
    sys.exit(1)
print(f"Templates: {len(templates)}, Faces: {len(faces)}")

# —————————————————————————————
# Unicode-kompatibles Einlesen
# —————————————————————————————
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = Path(path).read_bytes()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, flags)

# —————————————————————————————
# 2) Initialisierung
# —————————————————————————————
sift      = cv2.SIFT_create()
bf_sift   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
akaze     = cv2.AKAZE_create()
bf_akaze  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
orb       = cv2.ORB_create(nfeatures=2000)
bf_orb    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Matching-Parameter
good_thresh_sift  = 0.75
min_inliers_sift  = 15
good_thresh_akaze = 0.75
min_inliers_akaze = 12
good_thresh_orb   = 0.8
min_inliers_orb   = 8

# Skalierungs- und Rotationsfaktoren
scales  = [0.75, 1.0, 1.25]
angles  = [0, 90, 180, 270]

results = []

# Liste für Laufzeitmessungen
matching_times = []

# —————————————————————————————
# 3) Matching-Schleife: SIFT → AKAZE (Multiscale + Rotation) → ORB → Template
# —————————————————————————————
for t_path in templates:

     # Zeitmessung starten
    #start_time = time.time()
    #name = t_path.stem
    #print(f"Verarbeite {name}")

    # Template Preprocessing
    tpl       = imread_unicode(t_path)
    h_tpl     = compute_denoise_h(tpl)
    tpl_dn    = cv2.fastNlMeansDenoising(tpl, None, h=h_tpl,
                                          templateWindowSize=7, searchWindowSize=21)
    clahe_t   = cv2.createCLAHE(clipLimit=compute_clip_limit(tpl_dn),
                                tileGridSize=compute_tile_grid_size(tpl_dn))
    tpl_proc  = clahe_t.apply(tpl_dn)
    kp_t, des_t = sift.detectAndCompute(tpl_proc, None)

    best = {"method":None, "face":None, "inliers":0}

    # --- 3a) SIFT Matching ---
      # Zeitmessung starten
    start_time = time.time()
    name = t_path.stem
    print(f"Verarbeite {name}")

    if des_t is not None:
        for f_path in faces:
            img       = imread_unicode(f_path)
            h_img     = compute_denoise_h(img)
            img_dn    = cv2.fastNlMeansDenoising(img, None, h=h_img,
                                                 templateWindowSize=7, searchWindowSize=21)
            clahe_f   = cv2.createCLAHE(clipLimit=compute_clip_limit(img_dn),
                                        tileGridSize=compute_tile_grid_size(img_dn))
            img_proc  = clahe_f.apply(img_dn)
            kp_f, des_f = sift.detectAndCompute(img_proc, None)
            if des_f is None:
                continue
            matches = bf_sift.knnMatch(des_t, des_f, k=2)
            good = [m for pair in matches if len(pair)==2 and (m:=pair[0]).distance <
                    good_thresh_sift * pair[1].distance]
            if len(good) < min_inliers_sift:
                continue
            pts_t   = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            pts_f   = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(pts_t, pts_f, cv2.RANSAC, 5.0)
            if mask is not None:
                inl = int(mask.sum())
                if inl > best["inliers"]:
                    best.update({"method":"SIFT", "face":f_path.stem, "inliers":inl})

    # --- 3b) AKAZE Multiscale + Rotation Fallback ---
    if best["face"] is None and des_t is not None:
        kp_ta, des_ta = akaze.detectAndCompute(tpl_proc, None)
        if des_ta is not None:
            for f_path in faces:
                img       = imread_unicode(f_path)
                h_img     = compute_denoise_h(img)
                img_dn    = cv2.fastNlMeansDenoising(img, None, h=h_img,
                                                     templateWindowSize=7, searchWindowSize=21)
                clahe_b   = cv2.createCLAHE(clipLimit=compute_clip_limit(img_dn),
                                            tileGridSize=compute_tile_grid_size(img_dn))
                img_base  = clahe_b.apply(img_dn)
                h0, w0    = img_base.shape
                for s in scales:
                    sc = cv2.resize(img_base, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_LINEAR)
                    for a in angles:
                        M = cv2.getRotationMatrix2D((sc.shape[1]/2, sc.shape[0]/2), a, 1.0)
                        rot = cv2.warpAffine(sc, M, (sc.shape[1], sc.shape[0]))
                        kp_r, des_r = akaze.detectAndCompute(rot, None)
                        if des_r is None:
                            continue
                        matches = bf_akaze.knnMatch(des_ta, des_r, k=2)
                        good    = [m for pair in matches if len(pair)==2 and (m:=pair[0]).distance <
                                   good_thresh_akaze * pair[1].distance]
                        if len(good) < min_inliers_akaze:
                            continue
                        pts_t = np.float32([kp_ta[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                        pts_f = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                        H, mask = cv2.findHomography(pts_t, pts_f, cv2.RANSAC, 5.0)
                        if mask is not None:
                            inl = int(mask.sum())
                            if inl > best["inliers"]:
                                best.update({"method":"AKAZE", "face":f_path.stem, "inliers":inl})

    # --- 3c) ORB Ratio-Fallback ---
    if best["face"] is None and des_t is not None:
        kp_to, des_to = orb.detectAndCompute(tpl_proc, None)
        if des_to is not None:
            for f_path in faces:
                img       = imread_unicode(f_path)
                h_img     = compute_denoise_h(img)
                img_dn    = cv2.fastNlMeansDenoising(img, None, h=h_img,
                                                     templateWindowSize=7, searchWindowSize=21)
                clahe_o   = cv2.createCLAHE(clipLimit=compute_clip_limit(img_dn),
                                            tileGridSize=compute_tile_grid_size(img_dn))
                img_proc  = clahe_o.apply(img_dn)
                kp_fo, des_fo = orb.detectAndCompute(img_proc, None)
                if des_fo is None:
                    continue
                matches = bf_orb.knnMatch(des_to, des_fo, k=2)
                good    = [m for pair in matches if len(pair)==2 and (m:=pair[0]).distance <
                           good_thresh_orb * pair[1].distance]
                if len(good) < min_inliers_orb:
                    continue
                pts_t = np.float32([kp_to[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                pts_f = np.float32([kp_fo[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, mask = cv2.findHomography(pts_t, pts_f, cv2.RANSAC, 5.0)
                if mask is not None:
                    inl = int(mask.sum())
                    if inl > best["inliers"]:
                        best.update({"method":"ORB", "face":f_path.stem, "inliers":inl})

   
# Ausgabe der KP-Zahlen und Inliers
   # print(f"{name}: Methode={best['method']}, Template_KP={best.get('tpl_kp',0)}, Face_KP={best.get('face_kp',0)}, Inliers={best['inliers']}")

    # Ergebnisse sammeln und visualisieren
    results.append({"template":name, "face":best["face"], "method":best["method"], "inliers":best["inliers"]})
    tpl_vis = tpl_proc
    if best["face"]:
        # Face-Datei mit korrekter Extension suchen
        face_path = next((p for p in faces if p.stem == best["face"]), None)
        if face_path:
            face_img = imread_unicode(face_path)
        else:
            # Fallback: leeres Bild
            face_img = np.full_like(tpl_vis, 255)
        vis = np.hstack([tpl_vis, face_img])
    else:
        vis = np.hstack([tpl_vis, np.full_like(tpl_vis,255)])
    out = vis_dir / f"{name}_{best['method']}.png"
    cv2.imwrite(str(out), vis)

 # Zeitmessung beenden und speichern
    end_time = time.time()
    elapsed_time = end_time - start_time
    matching_times.append(elapsed_time)
    print(f"Template {name}: {elapsed_time:.2f} Sekunden")


# Koordinaten aus objekte.txt einlesen
objekte_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Koordinaten\objekte.txt")
objekte = []
with open(objekte_path, encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) >= 7:
            objekte.append({
                "id": parts[0],
                "x": parts[1],
                "y": parts[2],
                "rotation": parts[6]
            })
# Mapping: Würfelindex (1-basiert) zu Zielname (falls gematcht)
match_dict = {}
for r in results:
    # matched_face ist z.B. "1" (aus faces/1.png), template ist z.B. "Ziel_0_0"
    if r["matched_face"] is not None:
        match_dict[r["matched_face"]] = r["template"]

# TXT schreiben - ZWISCHENSPEICHER (zwi_pos.txt) - in debug_cubes_output Ordner
debug_output_dir = Path("debug_cubes_output")
debug_output_dir.mkdir(parents=True, exist_ok=True)  # Ordner erstellen falls nicht vorhanden
output_txt = debug_output_dir / "zwi_pos.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("id,x,y,rotation,ziel\n")
    last_line = None
    for obj in objekte:
        ziel = match_dict.get(obj["id"], "wende")
        line = f"{obj['id']},{obj['x']},{obj['y']},{obj['rotation']},{ziel}\n"
        print(f"DEBUG zwi_pos.txt: {line.strip()}")
        f.write(line)
        last_line = line
    
    # Letzte Zeile nochmal schreiben
    if last_line:
        f.write(last_line)
#def delete_matched_templates(match_dict, template_dir):
"""
    Löscht alle Template-Bilder aus target_tiles, die als Match in pos.txt ausgegeben wurden.
    """
   # for ziel in match_dict.values():
        # Zielname wie Ziel_0_2 → Ziel_0_2.png
       # template_file = template_dir / f"{ziel}.png"
        #if template_file.exists():
           # try:
               # template_file.unlink()
               # print(f"🗑️ Template gelöscht: {template_file}")
            #except Exception as e:
               # print(f"Fehler beim Löschen von {template_file}: {e}")#

# Nach dem Schreiben der zwi_pos.txt aufrufen:
#delete_matched_templates(match_dict, template_dir)
# 5) KPI-Berechnung und Export
# ————————————————————————————————————

# Berechne durchschnittliche Laufzeit aus tatsächlichen Messwerten
if matching_times:
    avg_time = sum(matching_times) / len(matching_times)
    print(f"\nDurchschnittliche Laufzeit für {len(matching_times)} Templates: {avg_time:.2f} Sekunden")
    
    # Datum für KPI-Datei
    current_date = time.strftime("%Y-%m-%d")
    
    # KPI-Pfade
    kpi_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\KPI\process_stats.txt")
    powerbi_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\PowerBI\process_stats.txt")
    
    try:
        # KPI-Werte berechnen
        wurfel_anzahl = len(faces)
        wendungen = sum(1 for obj in objekte if match_dict.get(obj["id"]) == "wende")
        zuruckgelegter_weg = 27159.410  # Fester Wert für zurückgelegten Weg
        gesamtzeit = sum(matching_times)
        gewendete_wurfel = wendungen
        anzahl_zyklen = 1
        
        # KPI-Zeile im vorgegebenen Format erstellen
        kpi_line = f"{wurfel_anzahl},{avg_time:.3f},{wendungen},{zuruckgelegter_weg:.3f},{gesamtzeit:.3f},{gewendete_wurfel},{anzahl_zyklen},{avg_time:.3f},{current_date}\n"
        
        # Überschriftenzeile
        header = "Bearbeitete Würfel,Durchschnittsdauer/Würfel,Gesamte Wendungen,Zurückgelegter Weg,Gesamtzeit,Gewendete Würfel,Anzhal Zyklen,Bildauswertung,Datum\n"
        
        # KPI-Datei schreiben
        with open(kpi_path, "w", encoding="utf-8") as kpi_file:
            kpi_file.write(header)
            kpi_file.write(kpi_line)
        print(f"✓ KPI-Datei gespeichert: {kpi_path}")
        
        # Kopieren in PowerBI-Ordner
        shutil.copy2(kpi_path, powerbi_path)
        print(f"✓ KPI-Datei nach PowerBI kopiert: {powerbi_path}")
        
        # Ausgabe zum Überprüfen
        print(f"\nErstellte KPI-Zeile:")
        print(f"{header.strip()}")
        print(f"{kpi_line.strip()}")
    except Exception as e:
        print(f"Fehler beim Speichern/Kopieren der KPI-Datei: {e}")
else:
    print("Keine Laufzeitdaten für KPI vorhanden.")

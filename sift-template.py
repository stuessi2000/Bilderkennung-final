#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# ————————————————————————————————————
# 1) Pfade & Ordner anlegen
# ————————————————————————————————————
template_dir = Path("target_tiles")
face_dir     = Path("faces")
vis_dir      = Path("sift_vis")
# csv_path     = Path(r"D:/Leon/Koordinaten_template_matching/sift_match_results.csv")
# log_path     = Path(r"D:/Leon/Koordinaten_template_matching/sift_debug_log.csv")
vis_dir.mkdir(parents=True, exist_ok=True)

# ————————————————————————————————————
# 2) SIFT und Matcher initialisieren (Stellschraube 1)
# ————————————————————————————————————
# Mehr Features, härterer Kontrast-Filter
sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.03)
bf   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Stellschraube 2: lockererer Ratio-Test
ratio_thresh     = 0.85
# homöopathisch gelockert
min_inliers      = 10
min_inlier_ratio = 0.15

# Skalierungen für Multi-Scale Matching (Stellschraube 3)
face_scales = [0.75, 1.0, 1.25]

results   = []
debug_log = []

# ————————————————————————————————————
# Bild laden & vorverarbeiten (CLAHE + Denoise)
# ————————————————————————————————————
def load_gray_preproc(path: Path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Kann Bild nicht dekodieren: {path}")
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img

# ————————————————————————————————————
# Augmentations: Rotation (0–315° in 45°-Schritten) + Flip
# ————————————————————————————————————
def augments(img: np.ndarray):
    h, w = img.shape
    base_mask = np.ones((h, w), dtype=np.uint8) * 255
    out = []
    for angle in range(0, 360, 45):
        M   = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderValue=255)
        mask = cv2.warpAffine(base_mask, M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
        out.append((rot,      mask, angle, False))
        out.append((cv2.flip(rot,1), mask, angle, True))
    return out

# ————————————————————————————————————
# 3) Matching-Schleife
# ————————————————————————————————————
faces = sorted(face_dir.glob("*.png"))
for t_path in sorted(template_dir.glob("*.png")):
    t_name = t_path.stem
    img1   = load_gray_preproc(t_path)
    kp1, des1 = sift.detectAndCompute(img1, None)
    if des1 is None:
        results.append({"template": t_name, "matched_face": None, "inliers": 0})
        continue

    best = {"face": None, "inliers": 0, "matchesMask": None,
            "kp2": None, "img2": None, "good": None,
            "angle": 0, "flipped": False}

    # Pro Würfelbild, zusätzlich in mehreren Skalen
    for f_path in faces:
        img2_raw0 = load_gray_preproc(f_path)
        for scale in face_scales:
            # Skalierung
            img2_raw = cv2.resize(
                img2_raw0,
                (0,0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR
            )
            # Alle Rotation/Flip-Varianten
            for aug_img2, mask2, angle, flipped in augments(img2_raw):
                kp2, des2 = sift.detectAndCompute(aug_img2, mask2)
                if des2 is None:
                    continue

                # Lowe's ratio test
                knn  = bf.knnMatch(des1, des2, k=2)
                good = [m for m,n in knn if m.distance < ratio_thresh * n.distance]
                if len(good) < min_inliers:
                    debug_log.append({
                        "template": t_name, "face": f_path.stem,
                        "scale": scale, "angle": angle, "flipped": flipped,
                        "matches": len(good), "inliers": 0
                    })
                    continue

                # Homographie + RANSAC
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
                if H is None or mask is None:
                    continue

                inliers      = int(mask.sum())
                inlier_ratio = inliers / len(good)
                debug_log.append({
                    "template": t_name, "face": f_path.stem,
                    "scale": scale, "angle": angle, "flipped": flipped,
                    "matches": len(good), "inliers": inliers,
                    "inlier_ratio": inlier_ratio
                })

                # SSIM-Abgleich als zusätzliches Filter
                h1, w1 = img1.shape
                retval, H_inv = cv2.invert(H)
                if retval == 0:
                    continue
                warp_back = cv2.warpPerspective(
                    aug_img2, H_inv, (w1,h1),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )
                score, _ = ssim(img1, warp_back, full=True)
                if score < 0.4:
                    continue

                # Harte Schwellwert-Prüfung
                if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
                    continue

                # Besten Treffer merken
                if inliers > best["inliers"]:
                    best.update(face=f_path.stem,
                                inliers=inliers,
                                matchesMask=mask.ravel().tolist(),
                                kp2=kp2,
                                img2=aug_img2,
                                good=good,
                                angle=angle,
                                flipped=flipped)

    results.append({
        "template":     t_name,
        "matched_face": best["face"],
        "inliers":      best["inliers"],
        "angle":        best["angle"],
        "flipped":      best["flipped"]
    })

    # Visualisierung speichern
    out1 = vis_dir / f"{t_name}_match.png"
    if best["face"]:
        vis = cv2.drawMatches(
            img1, kp1,
            best["img2"], best["kp2"],
            best["good"], None,
            matchColor=(0,255,0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchesMask=best["matchesMask"]
        )
    else:
        h,w = img1.shape
        vis = np.hstack([img1, np.full((h,w),255,np.uint8)])
    cv2.imwrite(str(out1), vis)

    out2 = vis_dir / f"{t_name}_sidebyside.png"
    if best["face"]:
        h1,w1 = img1.shape
        h2,w2 = best["img2"].shape
        mh = max(h1,h2)
        a1 = np.pad(img1, ((0,mh-h1),(0,0)), constant_values=255)
        a2 = np.pad(best["img2"], ((0,mh-h2),(0,0)), constant_values=255)
        side = np.hstack([a1,a2])
    else:
        h,w = img1.shape
        side = np.hstack([img1, np.full((h,w),255,np.uint8)])
    cv2.imwrite(str(out2), side)

# ————————————————————————————————————
# 4) TXT-Export im gewünschten Format
# ————————————————————————————————————

# Koordinaten aus objekte.txt einlesen
objekte_path = Path("cubes_coordinate/objekte.txt")
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

# TXT schreiben
output_txt = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Koordinaten_template_matching\pos.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("id,x,y,rotation,ziel\n")
    for obj in objekte:
        ziel = match_dict.get(obj["id"], "wende")
        f.write(f"{obj['id']},{obj['x']},{obj['y']},{obj['rotation']},{ziel}\n")

# Debug-Log bleibt als CSV
pd.DataFrame(debug_log).to_csv(log_path, index=False)

print("✓ Matching mit 3 Stellschrauben (nfeatures, Ratio, Multi-Scale) abgeschlossen. TXT-Export fertig.")
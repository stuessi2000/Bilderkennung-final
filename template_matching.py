import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# —————————————————————————————
# 1) Pfade & Ordner anlegen
# —————————————————————————————
BASE_DIR      = Path(__file__).resolve().parent
template_dir  = BASE_DIR / "target_tiles"   # Deine 9 Puzzle-Tiles
face_dir      = BASE_DIR / "faces"          # alle extrahierten Würfel-Flächen
vis_dir       = BASE_DIR / "matched_visuals"
csv_path      = BASE_DIR / "sift_match_results.csv"

vis_dir.mkdir(exist_ok=True)

# —————————————————————————————
# 2) SIFT und Matcher initialisieren
# —————————————————————————————
sift = cv2.SIFT_create()  
bf   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
ratio_thresh = 0.6       # Lowe's Ratio Threshold
min_inliers  = 8         # minimal akzeptierte Inlier für Homographie

# Ergebnisliste
results = []

# —————————————————————————————
# 3) Matching-Schleife
# —————————————————————————————
for t_path in sorted(template_dir.glob("*.png")):
    t_name = t_path.stem
    img1   = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)

    best = {
        "face":       None,
        "inliers":    0,
        "matchesMask": None,
        "kp2":        None,
        "img2":       None,
        "good":       None
    }

    # Falls Template kein Deskriptor liefert, überspringen
    if des1 is None:
        results.append({"template": t_name, "matched_face": None, "inliers": 0})
        continue

    # gegen alle Faces matchen
    for f_path in sorted(face_dir.glob("*.png")):
        img2 = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des2 is None:
            continue

        # 1) KNN-Match + Lowe-Ratio-Test
        knn_matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m,n in knn_matches if m.distance < ratio_thresh * n.distance]
        if len(good) < min_inliers:
            continue

        # 2) Homographie + RANSAC-Inlier zählen
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            continue
        inliers = int(mask.sum())

        # wenn besser als bisheriger Bestwert → merken
        if inliers > best["inliers"]:
            best.update({
                "face":        f_path.stem,
                "inliers":     inliers,
                "matchesMask": mask.ravel().tolist(),
                "kp2":         kp2,
                "img2":        img2,
                "good":        good
            })

    # 3) Ergebnis notieren
    results.append({
        "template":     t_name,
        "matched_face": best["face"],
        "inliers":      best["inliers"]
    })

    # 4) Matches visualisieren
    if best["face"] is not None:
        draw_params = dict(
            matchColor=(0,255,0),
            singlePointColor=None,
            matchesMask=best["matchesMask"],
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        vis = cv2.drawMatches(
            img1, kp1,
            best["img2"], best["kp2"],
            best["good"], None,
            **draw_params
        )
    else:
        # kein Match → nur Template links, rechts weiß
        h,w = img1.shape
        blank = np.full((h,w), 255, dtype=np.uint8)
        vis = np.hstack([img1, blank])

    cv2.imwrite(str(vis_dir / f"{t_name}_match.png"), vis)

    # 4b) Side-by-Side Visualisierung (ohne Keypoints)
    if best["face"] is not None:
        # Beide Bilder auf gleiche Höhe bringen
        h1, w1 = img1.shape
        h2, w2 = best["img2"].shape
        max_h = max(h1, h2)
        # ggf. Bilder in der Höhe auffüllen
        img1_pad = np.pad(img1, ((0, max_h - h1), (0, 0)), mode='constant', constant_values=255)
        img2_pad = np.pad(best["img2"], ((0, max_h - h2), (0, 0)), mode='constant', constant_values=255)
        sidebyside = np.hstack([img1_pad, img2_pad])
        cv2.imwrite(str(vis_dir / f"{t_name}_sidebyside.png"), sidebyside)
    else:
        # kein Match → nur Template links, rechts weiß
        h, w = img1.shape
        blank = np.full((h, w), 255, dtype=np.uint8)
        sidebyside = np.hstack([img1, blank])
        cv2.imwrite(str(vis_dir / f"{t_name}_sidebyside.png"), sidebyside)

# —————————————————————————————
# 5) CSV-Export
# —————————————————————————————
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print("✓ SIFT-Matching fertig!")
print("  Visuals in:", vis_dir)
print("  CSV:", csv_path)

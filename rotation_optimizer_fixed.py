#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation Optimizer - Bestimmt optimale Endrotation für Würfel
Vergleicht gematchte Würfelbilder mit Referenzbildern bei verschiedenen 90°-Rotationen
und berechnet die optimale Endrotation basierend auf der höchsten Ähnlichkeit.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import time
import os
import logging

print("START: rotation_optimizer.py wird geladen...")

# ========== KONFIGURATION ==========
# Ordnerpfade
coordinates_folder = r'\\192.168.0.1\Vision_Daten\DU_3\Koordinaten'
faces_folder = Path("faces")
target_tiles_folder = Path("target_tiles")
zwi_pos_file_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Koordinaten_template_matching\zwi_pos.txt")
pos_file_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Koordinaten_template_matching\pos.txt")
rotation_debug_folder = Path("rotation_debug")
final_rotations_file = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Koordinaten_template_matching\final_rotations.txt")

# Debug-Ordner erstellen
rotation_debug_folder.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rotation_optimizer.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ========== ÄHNLICHKEITS-METRIKEN ==========
def calculate_similarity_metrics(img1, img2):
    """
    Berechnet verschiedene Ähnlichkeitsmetriken zwischen zwei Bildern
    
    Args:
        img1: Erstes Bild (Referenz)
        img2: Zweites Bild (zu vergleichen)
        
    Returns:
        dict: Dictionary mit verschiedenen Ähnlichkeitsmetriken
    """
    try:
        # Bilder auf gleiche Größe bringen
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Kleinere Dimension als Zielgröße verwenden
        target_size = min(h1, w1, h2, w2)
        
        img1_resized = cv2.resize(img1, (target_size, target_size), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # SSIM (Structural Similarity Index)
        ssim_score = ssim(img1_resized, img2_resized, data_range=255)
        
        # MSE (Mean Squared Error) - niedriger ist besser
        mse_score = mean_squared_error(img1_resized, img2_resized)
        mse_normalized = 1.0 / (1.0 + mse_score / 1000.0)  # Normalisierung für bessere Vergleichbarkeit
        
        # Template Matching Score
        template_result = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
        template_score = np.max(template_result)
        
        # Histogram Correlation
        hist1 = cv2.calcHist([img1_resized], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_resized], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Kombinierter Score (gewichteter Durchschnitt)
        combined_score = (
            ssim_score * 0.4 +           # SSIM hat höchstes Gewicht
            template_score * 0.3 +       # Template Matching
            mse_normalized * 0.2 +       # MSE (normalisiert)
            hist_corr * 0.1              # Histogram Correlation
        )
        
        return {
            'ssim': ssim_score,
            'mse': mse_score,
            'mse_normalized': mse_normalized,
            'template_match': template_score,
            'hist_correlation': hist_corr,
            'combined_score': combined_score
        }
        
    except Exception as e:
        logging.error(f"Fehler bei Ähnlichkeitsberechnung: {e}")
        return {
            'ssim': 0.0,
            'mse': float('inf'),
            'mse_normalized': 0.0,
            'template_match': 0.0,
            'hist_correlation': 0.0,
            'combined_score': 0.0
        }

def rotate_image_90_degrees(image, rotation_steps):
    """
    Rotiert ein Bild um 90°-Schritte
    
    Args:
        image: Eingabebild
        rotation_steps: Anzahl der 90°-Schritte (0, 1, 2, 3)
        
    Returns:
        Rotiertes Bild
    """
    if rotation_steps == 0:
        return image.copy()
    elif rotation_steps == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_steps == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_steps == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("rotation_steps muss 0, 1, 2 oder 3 sein")

def load_image_robust(image_path):
    """
    Lädt ein Bild robust mit verschiedenen Methoden
    
    Args:
        image_path: Pfad zum Bild
        
    Returns:
        Geladenes Bild in Graustufen oder None bei Fehler
    """
    try:
        # Methode 1: Standard cv2.imread
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
            
        # Methode 2: Mit Numpy fromfile (für Sonderzeichen im Pfad)
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
            
        logging.error(f"Konnte Bild nicht laden: {image_path}")
        return None
        
    except Exception as e:
        logging.error(f"Fehler beim Laden von {image_path}: {e}")
        return None

def read_objekte_coordinates():
    """
    Liest die Koordinaten aus objekte.txt
    
    Returns:
        dict: Dictionary mit Objekt-ID als Key und Koordinaten als Value
    """
    objekte_path = Path(coordinates_folder) / "objekte.txt"
    coordinates = {}
    
    try:
        with open(objekte_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        if not lines:
            logging.error("objekte.txt ist leer")
            return coordinates
            
        # Header überspringen
        for i, line in enumerate(lines[1:], start=1):
            line = line.strip()
            if not line:
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                obj_id = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                rz = float(parts[6])  # RZ-Koordinate (Rotation)
                
                coordinates[obj_id] = {
                    'x': x,
                    'y': y,
                    'rz': rz
                }
                
        logging.info(f"Koordinaten für {len(coordinates)} Objekte aus objekte.txt gelesen")
        return coordinates
        
    except Exception as e:
        logging.error(f"Fehler beim Lesen von objekte.txt: {e}")
        return coordinates

def read_zwi_pos_file():
    """
    Liest die zwi_pos.txt Datei mit den Matching-Ergebnissen (Zwischenspeicher)
    
    Returns:
        dict: Dictionary mit Objekt-ID als Key und Ziel-Template als Value
    """
    matches = {}
    
    try:
        with open(zwi_pos_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        if not lines:
            logging.error("zwi_pos.txt ist leer")
            return matches
            
        # Header überspringen
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                obj_id = parts[0]
                ziel = parts[4]
                
                # Nur Objekte mit gültigen Zielen (nicht "wende")
                if ziel != "wende":
                    matches[obj_id] = ziel
                    
        logging.info(f"Matches für {len(matches)} Objekte aus zwi_pos.txt gelesen")
        return matches
        
    except Exception as e:
        logging.error(f"Fehler beim Lesen von zwi_pos.txt: {e}")
        return matches

def optimize_rotation_for_cube(cube_id, target_name, obj_coordinates):
    """
    Optimiert die Rotation für einen einzelnen Würfel
    
    Args:
        cube_id: ID des Würfels
        target_name: Name des gematchten Templates
        obj_coordinates: Koordinaten des Objekts
        
    Returns:
        dict: Optimierungsergebnisse
    """
    logging.info(f"Optimiere Rotation für Würfel {cube_id} mit Target {target_name}")
    
    # Pfade zu den Bildern
    face_path = faces_folder / f"{cube_id}.png"
    target_path = target_tiles_folder / f"{target_name}.png"
    
    # Bilder laden
    face_img = load_image_robust(face_path)
    target_img = load_image_robust(target_path)
    
    if face_img is None:
        logging.error(f"Konnte Würfelbild nicht laden: {face_path}")
        return None
        
    if target_img is None:
        logging.error(f"Konnte Referenzbild nicht laden: {target_path}")
        return None
    
    # Teste alle 90°-Rotationen
    rotation_results = []
    best_score = -1
    best_rotation = 0
    
    for rotation_steps in range(4):  # 0°, 90°, 180°, 270°
        rotation_degrees = rotation_steps * 90
        
        # Würfelbild rotieren
        rotated_face = rotate_image_90_degrees(face_img, rotation_steps)
        
        # Ähnlichkeit berechnen
        similarity = calculate_similarity_metrics(target_img, rotated_face)
        
        rotation_results.append({
            'rotation_degrees': rotation_degrees,
            'rotation_steps': rotation_steps,
            'similarity': similarity
        })
        
        # Debug-Bild speichern
        debug_comparison = create_debug_comparison(target_img, rotated_face, rotation_degrees, similarity)
        debug_path = rotation_debug_folder / f"cube_{cube_id}_{target_name}_rot_{rotation_degrees}.png"
        cv2.imwrite(str(debug_path), debug_comparison)
        
        # Besten Score tracken
        if similarity['combined_score'] > best_score:
            best_score = similarity['combined_score']
            best_rotation = rotation_degrees
            
        logging.info(f"  Rotation {rotation_degrees}°: Combined Score = {similarity['combined_score']:.4f}")
    
    # Ursprüngliche RZ-Koordinate aus objekte.txt
    original_rz = obj_coordinates.get('rz', 0.0)
    
    # Finale Rotation berechnen
    # Die optimale Rotation wird zur ursprünglichen RZ-Koordinate addiert
    final_rotation = (original_rz + best_rotation) % 360
    
    result = {
        'cube_id': cube_id,
        'target_name': target_name,
        'original_rz': original_rz,
        'optimal_rotation_offset': best_rotation,
        'final_rotation': final_rotation,
        'best_score': best_score,
        'all_rotations': rotation_results
    }
    
    logging.info(f"  Beste Rotation: {best_rotation}° (Score: {best_score:.4f})")
    logging.info(f"  Original RZ: {original_rz}°, Finale Rotation: {final_rotation}°")
    
    return result

def create_debug_comparison(target_img, face_img, rotation_degrees, similarity):
    """
    Erstellt ein Debug-Bild mit Vergleich zwischen Target und rotiertem Face
    
    Args:
        target_img: Referenzbild
        face_img: Würfelbild (rotiert)
        rotation_degrees: Rotationswinkel
        similarity: Ähnlichkeits-Metriken
        
    Returns:
        Kombiniertes Debug-Bild
    """
    # Bilder auf gleiche Größe bringen
    h1, w1 = target_img.shape[:2]
    h2, w2 = face_img.shape[:2]
    target_size = max(h1, w1, h2, w2)
    
    target_resized = cv2.resize(target_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    face_resized = cv2.resize(face_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Bilder nebeneinander legen
    combined = np.hstack([target_resized, face_resized])
    
    # Text mit Informationen hinzufügen
    text_info = [
        f"Rotation: {rotation_degrees}°",
        f"SSIM: {similarity['ssim']:.3f}",
        f"Template: {similarity['template_match']:.3f}",
        f"Combined: {similarity['combined_score']:.3f}"
    ]
    
    y_offset = 30
    for i, text in enumerate(text_info):
        cv2.putText(combined, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    
    return combined

def save_final_pos_file(optimization_results, obj_coordinates):
    """
    Erstellt die finale pos.txt Datei mit den optimierten Rotationen
    
    Args:
        optimization_results: Liste mit Optimierungsergebnissen
        obj_coordinates: Dictionary mit allen Objektkoordinaten
    """
    try:
        with open(pos_file_path, 'w', encoding='utf-8') as file:
            file.write("id,x,y,rotation,ziel\n")
            
            # Dictionary für schnellen Zugriff auf Optimierungsergebnisse erstellen
            optimization_dict = {}
            for result in optimization_results:
                if result is not None:
                    optimization_dict[result['cube_id']] = result
            
            # Alle Objekte aus objekte.txt durchgehen
            for obj_id, coords in obj_coordinates.items():
                if obj_id in optimization_dict:
                    # Objekt wurde gematcht und optimiert
                    result = optimization_dict[obj_id]
                    rotation = result['final_rotation']
                    ziel = result['target_name']
                else:
                    # Objekt wurde nicht gematcht -> "wende"
                    rotation = coords['rz']  # Original RZ-Wert verwenden
                    ziel = "wende"
                
                line = f"{obj_id},{coords['x']},{coords['y']},{rotation:.3f},{ziel}\n"
                file.write(line)
                logging.info(f"pos.txt: {line.strip()}")
        
        logging.info(f"Finale pos.txt gespeichert: {pos_file_path}")
        
        # Kopie für PowerBI erstellen
        powerbi_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\PowerBI\pos.txt")
        powerbi_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(pos_file_path, powerbi_path)
        logging.info(f"Kopie für PowerBI erstellt: {powerbi_path}")
        
    except Exception as e:
        logging.error(f"Fehler beim Speichern der finalen pos.txt: {e}")

def save_final_rotations(optimization_results):
    """
    Speichert die finalen Rotationen in eine Datei
    
    Args:
        optimization_results: Liste mit Optimierungsergebnissen
    """
    try:
        with open(final_rotations_file, 'w', encoding='utf-8') as file:
            file.write("cube_id,target_name,original_rz,optimal_rotation_offset,final_rotation,score\n")
            
            for result in optimization_results:
                if result is not None:
                    line = (f"{result['cube_id']},"
                           f"{result['target_name']},"
                           f"{result['original_rz']:.3f},"
                           f"{result['optimal_rotation_offset']},"
                           f"{result['final_rotation']:.3f},"
                           f"{result['best_score']:.4f}\n")
                    file.write(line)
        
        logging.info(f"Finale Rotationen gespeichert in: {final_rotations_file}")
        
        # Kopie für PowerBI erstellen
        powerbi_path = Path(r"\\192.168.0.1\Vision_Daten\DU_3\PowerBI\final_rotations.txt")
        powerbi_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(final_rotations_file, powerbi_path)
        logging.info(f"Kopie für PowerBI erstellt: {powerbi_path}")
        
    except Exception as e:
        logging.error(f"Fehler beim Speichern der finalen Rotationen: {e}")

def create_summary_report(optimization_results):
    """
    Erstellt einen zusammenfassenden Bericht
    
    Args:
        optimization_results: Liste mit Optimierungsergebnissen
    """
    print("\n" + "="*80)
    print("ROTATION OPTIMIZATION SUMMARY")
    print("="*80)
    
    valid_results = [r for r in optimization_results if r is not None]
    
    if not valid_results:
        print("❌ Keine gültigen Optimierungsergebnisse gefunden!")
        return
    
    print(f"✅ Optimierung abgeschlossen für {len(valid_results)} Würfel")
    print(f"📁 Debug-Bilder gespeichert in: {rotation_debug_folder}")
    print(f"📄 Finale Rotationen gespeichert in: {final_rotations_file}")
    print(f"📄 Finale pos.txt erstellt: {pos_file_path}")
    
    # Statistiken
    scores = [r['best_score'] for r in valid_results]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    print(f"\n📊 SCORE-STATISTIKEN:")
    print(f"   Durchschnittlicher Score: {avg_score:.4f}")
    print(f"   Bester Score: {max_score:.4f}")
    print(f"   Schlechtester Score: {min_score:.4f}")
    
    # Top 3 und Bottom 3 anzeigen
    sorted_results = sorted(valid_results, key=lambda x: x['best_score'], reverse=True)
    
    print(f"\n🏆 TOP 3 MATCHES:")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"   {i}. Würfel {result['cube_id']} → {result['target_name']} "
              f"(Score: {result['best_score']:.4f}, Rotation: {result['optimal_rotation_offset']}°)")
    
    if len(sorted_results) > 3:
        print(f"\n⚠️  SCHWÄCHSTE MATCHES:")
        for i, result in enumerate(sorted_results[-3:], 1):
            print(f"   {i}. Würfel {result['cube_id']} → {result['target_name']} "
                  f"(Score: {result['best_score']:.4f}, Rotation: {result['optimal_rotation_offset']}°)")
    
    print("="*80)

def main():
    """
    Hauptfunktion für die Rotationsoptimierung
    """
    start_time = time.time()
    
    print("🔄 Starte Rotationsoptimierung...")
    
    # 1. Koordinaten aus objekte.txt lesen
    obj_coordinates = read_objekte_coordinates()
    if not obj_coordinates:
        logging.error("Keine Koordinaten gefunden. Beende Programm.")
        return
    
    # 2. Matches aus zwi_pos.txt lesen
    matches = read_zwi_pos_file()
    if not matches:
        logging.error("Keine Matches in zwi_pos.txt gefunden. Beende Programm.")
        return
    
    print(f"📋 Gefunden: {len(matches)} Würfel mit Matches")
    
    # 3. Für jeden gematchten Würfel die Rotation optimieren
    optimization_results = []
    
    for cube_id, target_name in matches.items():
        if cube_id in obj_coordinates:
            result = optimize_rotation_for_cube(cube_id, target_name, obj_coordinates[cube_id])
            optimization_results.append(result)
        else:
            logging.warning(f"Keine Koordinaten für Würfel {cube_id} gefunden")
    
    # 4. Ergebnisse speichern
    save_final_rotations(optimization_results)
    
    # 5. Finale pos.txt erstellen
    save_final_pos_file(optimization_results, obj_coordinates)
    
    # 6. Zusammenfassung erstellen
    create_summary_report(optimization_results)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n⏱️  Gesamtlaufzeit: {elapsed:.2f} Sekunden")
    print(f"🎯 Durchschnitt pro Würfel: {elapsed/len(matches):.2f} Sekunden")

if __name__ == "__main__":
    main()

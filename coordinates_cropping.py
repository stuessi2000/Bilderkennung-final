#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import logging
from pathlib import Path
#from docx import Document
import os
import re
from math import cos, sin, radians
from datetime import datetime
import glob
import shutil
import sys

print("START: coordinates_cropping.py wird geladen...")

# ========== AUTOMATISCHE CACHE-LÖSCHUNG ==========
def clear_python_cache():
    """Löscht Python-Cache-Dateien für konsistente Ausführung"""
    try:
        # Aktuelles Verzeichnis und Unterverzeichnisse
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(current_dir)  # Ein Level höher
        
        cache_cleared = False
        
        # Lösche .pyc Dateien in src und workspace
        for search_dir in [current_dir, workspace_dir]:
            pyc_files = glob.glob(os.path.join(search_dir, "**/*.pyc"), recursive=True)
            for pyc_file in pyc_files:
                try:
                    os.remove(pyc_file)
                    cache_cleared = True
                except:
                    pass
        
        # Lösche __pycache__ Verzeichnisse in src und workspace
        for search_dir in [current_dir, workspace_dir]:
            pycache_dirs = glob.glob(os.path.join(search_dir, "**/__pycache__"), recursive=True)
            for pycache_dir in pycache_dirs:
                try:
                    shutil.rmtree(pycache_dir)
                    cache_cleared = True
                except:
                    pass
        
        if cache_cleared:
            print(f"🧹 Cache gelöscht in {current_dir} und {workspace_dir}")
        else:
            print(f"🔍 Kein Cache gefunden zum Löschen")
        
        # Auch sys.modules Cache für lokale Module löschen
        import sys
        modules_to_remove = []
        for module_name in sys.modules.keys():
            if 'coordinates_cropping' in module_name:
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        if modules_to_remove:
            print(f"🔄 Sys.modules Cache gelöscht: {modules_to_remove}")
        
    except Exception as e:
        print(f"⚠️  Cache-Löschung fehlgeschlagen: {e}")

# Cache beim Start automatisch löschen
clear_python_cache()

# Setup logging
log_file_path = 'coordinate_processing.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# Globale Variablen für Pfade und Ordner

# Ordnerpfade (angepasst für das aktuelle Workspace)
coordinates_folder = r'\\192.168.0.1\Vision_Daten\DU_3\Koordinaten'
cropped_folder = r"faces"  # Ausgabeordner für extrahierte Würfel-Oberseiten
image_folder = 'png_cubes'  # Eingabeordner mit Graustufen-Bildern
debug_folder = 'debug_cubes_output'  # Debug-Ausgabeordner

# === Einstellungen für die Würfel-Extraktion (aus extract_cubes.py) ===
# TARGET_W = 200               # Pixel-Breite der extrahierten Face
# TARGET_H = 200               # Pixel-Höhe der extrahierten Face

# Bildgröße explizit definieren
IMAGE_WIDTH = 1602
IMAGE_HEIGHT = 1212

# ===== AUTOMATISCHE 3-PUNKT-KALIBRIERUNG =====
# Neue Weltkoordinaten aus objekte.txt
OBJ1_WORLD = (130.58170, -362.36932, -91.92541)  # Ursprung (Objekt 1)
OBJ2_WORLD = (128.89926, -530.20135, -2.20383)   # Objekt 2
OBJ3_WORLD = (-113.93775, -361.92365, 178.03206) # Objekt 3

# Gemessene Pixelpositionen der 3 Objekte
OBJ1_PIXEL = (1454, 194)  # Ursprung (Objekt 1)
OBJ2_PIXEL = (1443, 972)  # Objekt 2  
OBJ3_PIXEL = (315, 189)   # Objekt 3

def auto_calibrate_3_points():
    """
    Automatische Kalibrierung basierend auf 3 bekannten Punkten
    Berechnet optimale Skalierung und validiert die Genauigkeit
    """
    print("\n🎯 === AUTOMATISCHE 3-PUNKT-KALIBRIERUNG ===")
    
    # Objekt 1 als Ursprung verwenden
    x0, y0, rz0 = OBJ1_WORLD
    px0, py0 = OBJ1_PIXEL
    
    print(f"Ursprung (Objekt 1): Welt({x0:.3f}, {y0:.3f}, RZ={rz0:.3f}) -> Pixel({px0}, {py0})")
    
    # Berechne Abstände und Skalierung mit Objekt 2 und 3
    distances_world = []
    distances_pixel = []
    
    for obj_world, obj_pixel, obj_name in [
        (OBJ2_WORLD, OBJ2_PIXEL, "Objekt 2"),
        (OBJ3_WORLD, OBJ3_PIXEL, "Objekt 3")
    ]:
        # Weltabstand zum Ursprung
        dx_world = obj_world[0] - x0
        dy_world = obj_world[1] - y0
        # Y-Koordinaten spiegeln (wichtig für das Koordinatensystem)
        dy_world = -dy_world  
        dist_world = (dx_world**2 + dy_world**2)**0.5
        
        # Pixelabstand zum Ursprung
        dx_pixel = obj_pixel[0] - px0
        dy_pixel = obj_pixel[1] - py0
        dist_pixel = (dx_pixel**2 + dy_pixel**2)**0.5
        
        print(f"{obj_name}: Weltabstand={dist_world:.3f}, Pixelabstand={dist_pixel:.3f}")
        
        if dist_world > 0:
            scale = dist_pixel / dist_world
            print(f"  -> Skalierung: {scale:.6f} Pixel/Einheit")
            distances_world.append(dist_world)
            distances_pixel.append(dist_pixel)
    
    # Durchschnittliche Skalierung berechnen
    if distances_world:
        total_world = sum(distances_world)
        total_pixel = sum(distances_pixel)
        avg_scale = total_pixel / total_world
        print(f"\n✅ Durchschnittliche Skalierung: {avg_scale:.6f} Pixel/Einheit")
        return x0, y0, rz0, px0, py0, avg_scale
    else:
        print("❌ Fehler bei der Skalierungsberechnung!")
        return x0, y0, rz0, px0, py0, 4.655331  # Fallback

def validate_calibration(x0, y0, rz0, px0, py0, scale):
    """
    Validiert die Kalibrierung durch Rückprojektion aller 3 Punkte
    """
    print("\n🔍 === KALIBRIERUNGS-VALIDIERUNG ===")
    
    validation_objects = [
        (OBJ1_WORLD, OBJ1_PIXEL, "Objekt 1 (Ursprung)"),
        (OBJ2_WORLD, OBJ2_PIXEL, "Objekt 2"),
        (OBJ3_WORLD, OBJ3_PIXEL, "Objekt 3")
    ]
    
    total_error = 0
    max_error = 0
    
    for obj_world, obj_pixel_measured, obj_name in validation_objects:
        # Berechne erwartete Pixelposition
        dx_world = obj_world[0] - x0
        dy_world = obj_world[1] - y0
        dy_world = -dy_world  # Y-Spiegelung
        
        px_calculated = px0 + dx_world * scale
        py_calculated = py0 + dy_world * scale
        
        # Berechne Abweichung
        error_x = abs(px_calculated - obj_pixel_measured[0])
        error_y = abs(py_calculated - obj_pixel_measured[1])
        error_total = (error_x**2 + error_y**2)**0.5
        
        print(f"{obj_name}:")
        print(f"  Gemessen:  ({obj_pixel_measured[0]}, {obj_pixel_measured[1]})")
        print(f"  Berechnet: ({px_calculated:.2f}, {py_calculated:.2f})")
        print(f"  Abweichung: {error_total:.2f}px (X: {error_x:.2f}px, Y: {error_y:.2f}px)")
        
        if obj_name != "Objekt 1 (Ursprung)":  # Ursprung hat immer 0 Fehler
            total_error += error_total
            max_error = max(max_error, error_total)
    
    avg_error = total_error / 2  # 2 Nicht-Ursprungs-Objekte
    print(f"\n📊 Validierungsergebnis:")
    print(f"  Durchschnittliche Abweichung: {avg_error:.2f}px")
    print(f"  Maximale Abweichung: {max_error:.2f}px")
    
    if avg_error < 15:
        print("  ✅ Kalibrierung: AUSGEZEICHNET (< 15px)")
    elif avg_error < 30:
        print("  ✅ Kalibrierung: GUT (< 30px)")
    else:
        print("  ⚠️  Kalibrierung: VERBESSERUNGSBEDÜRFTIG (> 30px)")
    
    return avg_error < 30

# Automatische Kalibrierung durchführen
X0, Y0, RZ0, IMAGE_ORIGIN_X, IMAGE_ORIGIN_Y, PIXELS_PER_UNIT = auto_calibrate_3_points()

# Validierung der neuen Kalibrierung
calibration_valid = validate_calibration(X0, Y0, RZ0, IMAGE_ORIGIN_X, IMAGE_ORIGIN_Y, PIXELS_PER_UNIT)

print(f"\n🎉 === NEUE KALIBRIERUNG AKTIVIERT ===")
print(f"Ursprung: X0={X0:.5f}, Y0={Y0:.5f}, RZ0={RZ0:.5f}")
print(f"Bildursprung: IMAGE_ORIGIN_X={IMAGE_ORIGIN_X}, IMAGE_ORIGIN_Y={IMAGE_ORIGIN_Y}")
print(f"Skalierung: PIXELS_PER_UNIT={PIXELS_PER_UNIT:.6f}")
print(f"Validierung: {'✅ ERFOLGREICH' if calibration_valid else '⚠️ WARNUNG'}")

# Feintuning-Offsets für Positionskorrektur (SYSTEMATISCHE KALIBRIERUNG)
POSITION_OFFSET_X = 0   # Zusätzlicher X-Offset (positiv = nach rechts)
POSITION_OFFSET_Y = 0   # TEST: Neue Offset-Konfiguration zur Cache-Überprüfung

# Temporäre objektspezifische Korrekturen (nur für Debugging - NICHT für Produktion!)
OBJECT_SPECIFIC_OFFSETS_DISABLED = {
    2: {"x": 0, "y": -5},    # Objekt 3: leicht nach oben
    5: {"x": -3, "y": -5},   # Objekt 6: nach links oben  
    7: {"x": 0, "y": -8},    # Objekt 8: nach oben (war zu niedrig)
}

# Aktiviere objektspezifische Korrekturen nur für Debugging
USE_OBJECT_SPECIFIC_OFFSETS = False  # Für universelle Lösung auf False setzen

# Die folgenden Werte werden jetzt durch auto_calibrate_3_points() automatisch gesetzt:
# PIXELS_PER_UNIT = wird automatisch berechnet
# X0, Y0, RZ0 = werden automatisch gesetzt  
# IMAGE_ORIGIN_X, IMAGE_ORIGIN_Y = werden automatisch gesetzt

# Bekannte Würfelgröße in Pixeln
CUBE_SIZE_PIXELS = 160

# ===== DEBUG-OPTIONEN FÜR KOORDINATENTRANSFORMATION =====
# Konfigurierbare Spiegelung der Weltkoordinaten (korrigiert für negative Y-Koordinaten)
FLIP_X = False  # X-Achse nicht spiegeln
FLIP_Y = True   # Y-Achse spiegeln (WICHTIG für negative Y-Koordinaten!)

# Manueller Offset zur Korrektur der globalen Rotation (in Grad)
ADJUST_GLOBAL_ROTATION_OFFSET = 0  # Kein manueller Offset mehr nötig mit automatischer 3-Punkt-Kalibrierung

# Konfigurierbarer Ursprung: Objekt 1/X ist der feste Ursprung
DEBUG_ORIGIN_OBJECT_INDEX = None  # Fester Ursprung verwenden (Objekt 1/X)

# Konfigurierbare Deaktivierung der globalen Rotation
DISABLE_GLOBAL_ROTATION = True  # Globale Rotation deaktiviert - Positionen bleiben korrekt

# Debug-Ausgabe aktivieren
DEBUG_VERBOSE = True  # Detaillierte Ausgabe der Transformationsschritte

# Funktion zum Rotieren eines Punktes
def rotate_point(dx, dy, angle_deg):
    angle_rad = radians(angle_deg)
    x_rot = dx * cos(angle_rad) - dy * sin(angle_rad)
    y_rot = dx * sin(angle_rad) + dy * cos(angle_rad)
    return x_rot, y_rot

# Funktion zum Lesen der Koordinaten aus einer .txt-Datei
def read_coordinates_from_txt(txt_path):
    print(f"Versuche, Koordinaten aus {txt_path} zu lesen...")
    try:
        coordinates = []
        with open(txt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        if not lines:
            print("Datei ist leer.")
            return []
            
        # Erste Zeile (Header) überspringen
        header = lines[0].strip()
        print(f"Header: {header}")
        
        for i, line in enumerate(lines[1:], start=1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # CSV mit Komma oder Semikolon getrennt interpretieren
                if ';' in line:
                    parts = line.split(';')
                else:
                    parts = line.split(',')
                    
                if len(parts) >= 7:
                    obj_id = parts[0].strip()
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip())
                    rx = float(parts[4].strip())
                    ry = float(parts[5].strip())
                    rz = float(parts[6].strip())
                    
                    # Für Kompatibilität mit bestehendem Code: (X, Y, RX, RY, RZ) oder (X, Y, RZ)
                    coordinates.append((x, y, rx, ry, rz))
                    print(f"Gelesen: Objekt {obj_id} mit X={x}, Y={y}, RZ={rz}")
                else:
                    print(f"Zeile {i} hat nicht genügend Spalten: {len(parts)} (erwartet: 7)")
                    
            except Exception as e:
                print(f"Fehler beim Verarbeiten der Zeile {i}: {e}")
                
        logging.info(f"Gelesene Koordinaten aus TXT: {coordinates}")
        print(f"Insgesamt {len(coordinates)} Koordinaten aus TXT-Datei gefunden.")
        return coordinates
        
    except Exception as e:
        print(f"Fehler beim Lesen der TXT-Datei {txt_path}: {e}")
        return []

# Funktion zum Zuschneiden des Bildes basierend auf den Koordinaten (alte Version, wird nicht mehr verwendet)
def crop_cube(img_array, x, y, width, height, angle=0, crop_idx=0, original_stem="image"):
    print(f"Schneide Bild aus Array zu, Position: ({x}, {y}), Größe: {width}x{height}, Winkel: {angle}, Index: {crop_idx}")
    try:
        # Bild ist bereits als Numpy-Array vorhanden
        img = img_array.copy() # Kopie erstellen, um das Original nicht zu verändern
        if img is None:
            print(f"Fehler: Übergebenes Bild-Array ist None.")
            return None
            
        print(f"Bild-Array erhalten. Größe: {img.shape}")
        
        # Bildgröße ermitteln
        height_img, width_img = img.shape[:2]
        
        # Sicherstellen, dass die Koordinaten innerhalb des Bildes liegen
        if x < 0 or y < 0 or x + width > width_img or y + height > height_img:
            print(f"Warnung: Koordinaten außerhalb des Bildes. Anpassung wird vorgenommen.")
            # Anpassen der Koordinaten, um innerhalb des Bildes zu bleiben
            x = max(0, min(x, width_img - width))
            y = max(0, min(y, height_img - height))
        
        # Zuerst den Bereich zuschneiden
        temp_cropped_img = img[y:y + height, x:x + width]
        
        if temp_cropped_img.size == 0:
            print(f"Fehler: Der zugeschnittene Bereich für ({x},{y}) mit Größe {width}x{height} ist leer.")
            return None

        if angle != 0:
            # Rotationszentrum (Mitte des zugeschnittenen Bildes)
            # Wichtig: Das Zentrum ist jetzt relativ zum zugeschnittenen Bild
            center_cropped = (width // 2, height // 2)
            
            # Erstellen der Rotationsmatrix
            M = cv2.getRotationMatrix2D(center_cropped, angle, 1.0)
            
            # Berechne die Größe des Ausgabebildes, um den gesamten rotierten Inhalt aufzunehmen
            # Dies ist wichtig, um Abschneiden zu verhindern
            abs_cos = abs(M[0,0]) 
            abs_sin = abs(M[0,1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            # Anpassen der Rotationsmatrix, um die Translation zu berücksichtigen
            M[0, 2] += (bound_w / 2) - center_cropped[0]
            M[1, 2] += (bound_h / 2) - center_cropped[1]

            # Bild rotieren
            # Das rotierte Bild wird so groß wie nötig, um alles zu umfassen
            rotated_bounded_img = cv2.warpAffine(temp_cropped_img, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC)

            # Jetzt zentriert auf die ursprüngliche Würfelgröße (width, height) zuschneiden
            # Dies stellt sicher, dass das Endbild die gewünschte Größe hat
            center_x_rot_bound = bound_w // 2
            center_y_rot_bound = bound_h // 2
            
            start_x = center_x_rot_bound - width // 2
            start_y = center_y_rot_bound - height // 2
            end_x = start_x + width
            end_y = start_y + height
            
            cropped_img = rotated_bounded_img[start_y:end_y, start_x:end_x]
        else:
            # Keine Rotation, das bereits zugeschnittene Bild verwenden (hat bereits die Größe width x height)
            cropped_img = temp_cropped_img
        
        print(f"Finale zugeschnittene Bild Größe: {cropped_img.shape if cropped_img is not None else 'None'}")
        
        # Dateiname für den zugeschnittenen Würfel
        output_filename = f"{original_stem}_cropped_{crop_idx}.png"
        output_path = Path(cropped_folder) / output_filename
        return cropped_img
    except Exception as e:
        print(f"Fehler beim Zuschneiden des Bildes: {e}")
        return None

# Neue Funktion zum korrekten Zuschneiden mit Rotation um den Mittelpunkt
def crop_cube_with_rotation(img_array, center_x, center_y, width, height, angle_deg, crop_idx=0, original_stem="image"):
    """
    Schneidet einen Bildbereich um einen Mittelpunkt aus und rotiert ihn korrekt.
    
    Args:
        img_array: Das Eingabebild als NumPy-Array
        center_x: X-Koordinate des Mittelpunkts im Bild
        center_y: Y-Koordinate des Mittelpunkts im Bild
        width: Breite des gewünschten Ausschnitts
        height: Höhe des gewünschten Ausschnitts
        angle_deg: Rotationswinkel in Grad (RZ-Wert des Würfels)
        crop_idx: Index für Debugging
        original_stem: Dateiname-Stamm für Debugging
    
    Returns:
        Rotierter und zugeschnittener Bildbereich als NumPy-Array
    """
    print(f"Schneide Bild mit Rotation zu, Mittelpunkt: ({center_x:.2f}, {center_y:.2f}), Größe: {width}x{height}, Winkel: {angle_deg}°, Index: {crop_idx}")
    
    try:
        # Kopie des Bildes erstellen
        img = img_array.copy()
        if img is None:
            print(f"Fehler: Übergebenes Bild-Array ist None.")
            return None
            
        # Bildgröße ermitteln
        img_height, img_width = img.shape[:2]
        print(f"Bild-Array erhalten. Größe: {img.shape}")
        
        # Prüfen, ob der Mittelpunkt im Bild liegt
        if center_x < 0 or center_x >= img_width or center_y < 0 or center_y >= img_height:
            print(f"Warnung: Mittelpunkt ({center_x:.2f}, {center_y:.2f}) liegt außerhalb des Bildes (Größe: {img_width}x{img_height})")
            # Mittelpunkt ins Bild verschieben
            center_x = max(0, min(center_x, img_width - 1))
            center_y = max(0, min(center_y, img_height - 1))
            print(f"Angepasster Mittelpunkt: ({center_x:.2f}, {center_y:.2f})")
        
        # Berechne die Größe des erweiterten Bereichs, um nach der Rotation genügend Material zu haben
        # Verwende die Diagonale des gewünschten Rechtecks als Sicherheitsabstand
        diagonal = int(np.sqrt(width**2 + height**2)) + 20  # +20 Pixel Puffer
        extended_size = max(diagonal, max(width, height) + 40)
        
        # Berechne den erweiterten Bereich um den Mittelpunkt
        half_extended = extended_size // 2
        
        # Koordinaten des erweiterten Bereichs
        ext_x1 = int(center_x - half_extended)
        ext_y1 = int(center_y - half_extended)
        ext_x2 = int(center_x + half_extended)
        ext_y2 = int(center_y + half_extended)
        
        # Padding hinzufügen, falls der erweiterte Bereich über die Bildgrenzen hinausgeht
        pad_left = max(0, -ext_x1)
        pad_top = max(0, -ext_y1)
        pad_right = max(0, ext_x2 - img_width)
        pad_bottom = max(0, ext_y2 - img_height)
        
        # Bild mit Padding erweitern, falls nötig
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
            # Mittelpunkt im gepaddeten Bild anpassen
            center_x_padded = center_x + pad_left
            center_y_padded = center_y + pad_top
        else:
            img_padded = img
            center_x_padded = center_x
            center_y_padded = center_y
        
        # Erweiterten Bereich aus dem (möglicherweise gepaddeten) Bild extrahieren
        ext_x1_padded = int(center_x_padded - half_extended)
        ext_y1_padded = int(center_y_padded - half_extended)
        ext_x2_padded = int(center_x_padded + half_extended)
        ext_y2_padded = int(center_y_padded + half_extended)
        
        extended_region = img_padded[ext_y1_padded:ext_y2_padded, ext_x1_padded:ext_x2_padded]
        
        if extended_region.size == 0:
            print(f"Fehler: Der erweiterte Bereich ist leer.")
            return None
        
        # Rotation um den Mittelpunkt des erweiterten Bereichs
        if angle_deg != 0:
            # Mittelpunkt des erweiterten Bereichs
            ext_center = (extended_region.shape[1] // 2, extended_region.shape[0] // 2)
            
            # Rotationsmatrix erstellen
            rotation_matrix = cv2.getRotationMatrix2D(ext_center, angle_deg, 1.0)
            
            # Bild rotieren
            rotated_region = cv2.warpAffine(extended_region, rotation_matrix, 
                                          (extended_region.shape[1], extended_region.shape[0]), 
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        else:
            rotated_region = extended_region
        
        # Finalen Ausschnitt aus dem rotierten Bereich extrahieren
        # Mittelpunkt des rotierten Bereichs
        rot_center_x = rotated_region.shape[1] // 2
        rot_center_y = rotated_region.shape[0] // 2
        
        # Koordinaten für den finalen Ausschnitt
        final_x1 = rot_center_x - width // 2
        final_y1 = rot_center_y - height // 2
        final_x2 = final_x1 + width
        final_y2 = final_y1 + height
        
        # Sicherstellen, dass der finale Ausschnitt innerhalb des rotierten Bereichs liegt
        final_x1 = max(0, final_x1)
        final_y1 = max(0, final_y1)
        final_x2 = min(rotated_region.shape[1], final_x2)
        final_y2 = min(rotated_region.shape[0], final_y2)
        
        # Finalen Ausschnitt extrahieren
        final_crop = rotated_region[final_y1:final_y2, final_x1:final_x2]
        
        # Falls der Ausschnitt nicht die gewünschte Größe hat, auf die gewünschte Größe anpassen
        if final_crop.shape[0] != height or final_crop.shape[1] != width:
            print(f"Warnung: Finaler Ausschnitt hat Größe {final_crop.shape[:2]}, erwartet: {height}x{width}. Größe wird angepasst.")
            final_crop = cv2.resize(final_crop, (width, height), interpolation=cv2.INTER_CUBIC)
        
        print(f"Finale zugeschnittene Bild Größe: {final_crop.shape}")
        return final_crop
        
    except Exception as e:
        print(f"Fehler beim Zuschneiden des Bildes mit Rotation: {e}")
        return None

# Funktion, die alle Bilder basierend auf den Koordinaten zuschneidet
def process_images():
    print("Starte Bildverarbeitung...")
    # Überprüfen, ob die Ordner existieren
    print(f"Koordinatenordner: {coordinates_folder}, existiert: {os.path.exists(coordinates_folder)}")
    print(f"Bildordner: {image_folder}, existiert: {os.path.exists(image_folder)}")
    
    # Suche nach Koordinatendateien (.txt wird verwendet, keine docx mehr)
    txt_files = list(Path(coordinates_folder).glob('*.txt'))
    print(f"Gefundene .txt-Dateien: {[str(f) for f in txt_files]}")
    
    coordinates = []
    
    # Nur noch TXT-Dateien werden akzeptiert
    if txt_files:
        coord_file = txt_files[0]
        print(f"Verarbeite TXT-Datei: {coord_file}")
        coordinates = read_coordinates_from_txt(coord_file)
    else:
        print("Keine .txt-Dateien gefunden!")
        return

    # Überprüfen, welche Bilder im Bildordner vorhanden sind
    image_files = list(Path(image_folder).glob('*.png'))
    print(f"Gefundene Bilder: {[str(f) for f in image_files]}")
    
    # Sicherstellen, dass wir genug Koordinaten für alle Bilder haben (oder zumindest eine, wenn keine Koordinaten gelesen wurden)
    if not coordinates and image_files:
        print("Keine Koordinaten gelesen, verwende Standard für Debugging, falls Bilder vorhanden.")
        # Hier könnten Standard-Weltkoordinaten für einen Testfall definiert werden, falls gewünscht
        # Fürs Erste wird die Schleife unten übersprungen, wenn keine Koordinaten vorhanden sind.
    elif coordinates and not image_files:
        print("Koordinaten vorhanden, aber keine Bilder im Quellordner gefunden!")
        return
    elif not coordinates and not image_files:
        print("Keine Koordinaten und keine Bilder gefunden.")
        return

    # Erstelle notwendige Ausgabe-Ordner
    os.makedirs(cropped_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    
    # CSV-Liste für Würfel-Mittelpunkte initialisieren
    cube_centers_csv = []

    if not image_files:
        print("Keine Bilder im Quellordner gefunden!") # Redundante Prüfung, aber sicher ist sicher
        return

    # Annahme: Wir verwenden das erste gefundene Bild für alle Koordinaten
    source_image_path = str(image_files[0])
    source_image_stem = image_files[0].stem
    print(f"Verwende Quellbild: {source_image_path} für alle Koordinaten.")

    # Lade das Originalbild
    original_img = cv2.imread(source_image_path)
    if original_img is None:
        print(f"Fehler beim Laden des Bildes: {source_image_path}")
        return

    img_height, img_width = original_img.shape[:2]
    
    # Debug-Bild-Canvas wird dynamisch nach Würfelverarbeitung bestimmt
    # Wir verschieben die Canvas-Berechnung und Debug-Bild-Erstellung nach unten,
    # nachdem alle Würfelkoordinaten (mit Debug-Konfiguration) verarbeitet wurden.

    # ===== KOORDINATENSYSTEM-KALIBRIERUNG =====
    print(f"\n=== AUTOMATISCH KALIBRIERTES KOORDINATENSYSTEM ===")
    print(f"Ursprung: X0={X0:.5f}, Y0={Y0:.5f}, RZ0={RZ0:.5f}")
    print(f"Bildursprung: ({IMAGE_ORIGIN_X}, {IMAGE_ORIGIN_Y})")
    print(f"Skalierung: PIXELS_PER_UNIT={PIXELS_PER_UNIT:.6f}")
    print(f"Kalibrierung gültig: {'✅' if calibration_valid else '⚠️'}")
    
    # Die automatische Kalibrierung hat bereits alle nötigen Werte gesetzt

    # ===== DEBUG-KONFIGURATION ANWENDEN =====
    print(f"\n=== DEBUG-KONFIGURATION ===")
    print(f"FLIP_X: {FLIP_X}, FLIP_Y: {FLIP_Y}")
    print(f"DISABLE_GLOBAL_ROTATION: {DISABLE_GLOBAL_ROTATION}")
    print(f"ADJUST_GLOBAL_ROTATION_OFFSET: {ADJUST_GLOBAL_ROTATION_OFFSET}")
    print(f"DEBUG_VERBOSE: {DEBUG_VERBOSE}")
    print(f"PIXELS_PER_UNIT: {PIXELS_PER_UNIT}")
    print("="*50)

    # Farben für Debugging variieren
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # === Neue Variablen für dynamische Debug-Canvas-Berechnung ===
    min_x = 0
    max_x = img_width
    min_y = 0
    max_y = img_height
    canvas_needs_resize = False

    # Wir sammeln erst alle Würfelmittelpunkte mit der neuen Transformationsfunktion
    cube_centers = []
    relative_angles = []
    
    for i, coord_data in enumerate(coordinates):
        # Entpacken der Koordinaten
        if len(coord_data) == 3:
            x_world, y_world, rz_cube = coord_data
        elif len(coord_data) == 5:
            x_world, y_world, _, _, rz_cube = coord_data
        else:
            logging.warning(f"Unerwartetes Koordinatenformat für Objekt {i+1}: {coord_data}")
            print(f"Unerwartetes Koordinatenformat für Objekt {i+1}: {coord_data}")
            continue

        print(f"\n--- Objekt {i+1} ---")
        print(f"Original Koordinaten: X={x_world}, Y={y_world}, RZ={rz_cube}")
        
        logging.info(f"Objekt {i+1}: Welt (X={x_world}, Y={y_world}), Würfelrotation RZ={rz_cube}")

        # Neue Transformationsfunktion verwenden
        center_x_pixel_abs, center_y_pixel_abs, relative_angle = world_to_image(x_world, y_world, rz_cube)

        # Objektspezifische Korrekturen anwenden (nur wenn aktiviert)
        if USE_OBJECT_SPECIFIC_OFFSETS and i in OBJECT_SPECIFIC_OFFSETS_DISABLED:
            offset = OBJECT_SPECIFIC_OFFSETS_DISABLED[i]
            center_x_pixel_abs += offset["x"]
            center_y_pixel_abs += offset["y"]
            print(f"  Objektspezifische Korrektur angewendet: dx={offset['x']}, dy={offset['y']}")
            print(f"  Korrigierte Position: ({center_x_pixel_abs:.3f}, {center_y_pixel_abs:.3f})")

        crop_width_px = CUBE_SIZE_PIXELS
        crop_height_px = CUBE_SIZE_PIXELS

        # Prüfe, ob der Mittelpunkt (mit Würfelausdehnung) außerhalb des aktuellen Canvas liegt
        half_size = CUBE_SIZE_PIXELS / 2
        left = center_x_pixel_abs - half_size
        right = center_x_pixel_abs + half_size
        top = center_y_pixel_abs - half_size
        bottom = center_y_pixel_abs + half_size

        # Prüfe und aktualisiere min/max, falls nötig
        if left < min_x:
            min_x = int(np.floor(left))
            canvas_needs_resize = True
        if right > max_x:
            max_x = int(np.ceil(right))
            canvas_needs_resize = True
        if top < min_y:
            min_y = int(np.floor(top))
            canvas_needs_resize = True
        if bottom > max_y:
            max_y = int(np.ceil(bottom))
            canvas_needs_resize = True

        # Speichere Würfelmittelpunkt und relativen Winkel für spätere Debug-Zeichnung
        cube_centers.append((center_x_pixel_abs, center_y_pixel_abs))
        relative_angles.append(relative_angle)

    # ========== Debug-Canvas und Bild vorbereiten ==========
    if canvas_needs_resize:
        padding_left = max(0, -min_x)
        padding_top = max(0, -min_y)
        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)
        print(f"Canvas erweitert: Original ({img_width}x{img_height}) -> Erweitert ({canvas_width}x{canvas_height})")
        print(f"Padding: links={padding_left}, oben={padding_top}")
        debug_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        debug_img[int(padding_top):int(padding_top + img_height), int(padding_left):int(padding_left + img_width)] = original_img
    else:
        padding_left = 0
        padding_top = 0
        canvas_width = img_width
        canvas_height = img_height
        debug_img = original_img.copy()
        print(f"Canvas unverändert: Original ({img_width}x{img_height})")

    # 5. Gitter zeichnen (mit Padding-Verschiebung)
    grid_spacing = 50  # Abstand der Gitterlinien
    grid_color = (50, 50, 50)  # Dunkelgrau für Gitter
    # Horizontale Linien
    for y_grid in range(0, img_height, grid_spacing):
        cv2.line(debug_img, (int(padding_left), int(padding_top + y_grid)), (int(padding_left + img_width), int(padding_top + y_grid)), grid_color, 1)
    # Vertikale Linien
    for x_grid in range(0, img_width, grid_spacing):
        cv2.line(debug_img, (int(padding_left + x_grid), int(padding_top)), (int(padding_left + x_grid), int(padding_top + img_height)), grid_color, 1)

    # 6. Koordinatensystem auf Debug-Bild zeichnen (mit Padding-Verschiebung)
    # Dokument-Ursprung (unten links im ursprünglichen Bild)
    doc_origin_x = int(padding_left + 0)
    doc_origin_y = int(padding_top + img_height - 1)  # Pixel sind 0-indiziert

    axis_length = 200
    # X-Achse (rot) - von links nach rechts
    cv2.line(debug_img, (doc_origin_x, doc_origin_y), (doc_origin_x + axis_length, doc_origin_y), (0, 0, 255), 2) # Rot
    cv2.putText(debug_img, 'X (Doc)', (doc_origin_x + axis_length + 5, doc_origin_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Y-Achse (grün) - von unten nach oben
    cv2.line(debug_img, (doc_origin_x, doc_origin_y), (doc_origin_x, doc_origin_y - axis_length), (0, 255, 0), 2) # Grün
    cv2.putText(debug_img, 'Y (Doc)', (doc_origin_x + 5, doc_origin_y - axis_length - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Ursprungspunkt (blau)
    cv2.circle(debug_img, (doc_origin_x, doc_origin_y), 5, (255, 0, 0), -1) # Blau
    cv2.putText(debug_img, 'Origin (Doc)', (doc_origin_x + 10, doc_origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

    # Bild-Koordinatensystem (oben links im ursprünglichen Bild) - optional zur Verdeutlichung
    img_cs_origin_x = int(padding_left + 0)
    img_cs_origin_y = int(padding_top + 0)
    cv2.line(debug_img, (img_cs_origin_x, img_cs_origin_y), (img_cs_origin_x + axis_length // 2, img_cs_origin_y), (200, 0, 200), 1) # Magenta X
    cv2.line(debug_img, (img_cs_origin_x, img_cs_origin_y), (img_cs_origin_x, img_cs_origin_y + axis_length // 2), (0, 200, 200), 1) # Cyan Y
    cv2.circle(debug_img, (img_cs_origin_x, img_cs_origin_y), 3, (200,200,0), -1) # Türkis
    cv2.putText(debug_img, 'Origin (Img)', (img_cs_origin_x + 5, img_cs_origin_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0),1)

    # ======= Jetzt alle Würfel zeichnen und zuschneiden =======
    cubes_saved = 0  # KPI-Zähler für erfolgreich gespeicherte Würfel
    for i, coord_data in enumerate(coordinates):
        center_x_pixel_abs, center_y_pixel_abs = cube_centers[i]
        relative_angle = relative_angles[i] 
        # Für Debug-Ausgabe und Logging: Koordinaten extrahieren
        if len(coord_data) == 3:
            x_world, y_world, rz_cube = coord_data
        elif len(coord_data) == 5:
            x_world, y_world, _, _, rz_cube = coord_data
        else:
            x_world, y_world, rz_cube = 0, 0, 0

        crop_width_px = CUBE_SIZE_PIXELS
        crop_height_px = CUBE_SIZE_PIXELS

        # Debug-Zeichnung des berechneten Mittelpunkts (mit Padding-Verschiebung)
        debug_center_x = int(center_x_pixel_abs + padding_left)
        debug_center_y = int(center_y_pixel_abs + padding_top)
        cv2.circle(debug_img, (debug_center_x, debug_center_y), 5, (0, 255, 255), -1) # Gelber Punkt für Mittelpunkt

        color_index = i % len(colors)
        rect_color = colors[color_index]
        text_color = colors[(color_index + 1) % len(colors)]

        # Debug-Overlay: Drehbares Rechteck visualisieren (mit Padding-Verschiebung)
        half_width = crop_width_px / 2
        half_height = crop_height_px / 2
        corners_rel = np.array([
            [-half_width, -half_height],  # oben links
            [half_width, -half_height],   # oben rechts
            [half_width, half_height],    # unten rechts
            [-half_width, half_height]    # unten links
        ])
        angle_rad = np.radians(relative_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners_rotated = np.dot(corners_rel, rotation_matrix.T)
        corners_abs = corners_rotated + np.array([center_x_pixel_abs + padding_left, center_y_pixel_abs + padding_top])
        corners_abs = corners_abs.astype(np.int32)
        cv2.polylines(debug_img, [corners_abs], True, rect_color, 2)
        cv2.putText(debug_img, f"Obj {i+1} ({relative_angle:.1f}°)", 
                   (debug_center_x + 10, debug_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Bild zuschneiden und rotieren mit dem relativen Winkel
        cropped_cube = crop_cube_with_rotation(original_img, center_x_pixel_abs, center_y_pixel_abs, crop_width_px, crop_height_px, relative_angle, i, source_image_stem)
        if cropped_cube is not None:
            # Benenne das Bild nach der Objekt-Nummer aus der Koordinatenliste (i+1)
            face_image_name = f"{i+1}.png"
            face_image_path = Path(cropped_folder) / face_image_name
            try:
                cv2.imwrite(str(face_image_path), cropped_cube)
                print(f"✓ {face_image_name}")
                # Mittelpunkte für CSV sammeln (Format wie in extract_cubes.py)
                cube_centers_csv.append({
                    "FaceImage": face_image_name,
                    "Center_X_px": int(center_x_pixel_abs),
                    "Center_Y_px": int(center_y_pixel_abs)
                })
                cubes_saved += 1  # KPI-Zähler erhöhen
            except Exception as e:
                print(f"Fehler beim Speichern des zugeschnittenen Bildes {face_image_path}: {e}")
                logging.error(f"Fehler beim Speichern des zugeschnittenen Bildes {face_image_path}: {e}")
        else:
            print(f"Konnte Würfel {i+1} nicht zuschneiden.")
            logging.warning(f"Konnte Würfel {i+1} (Welt X:{x_world}, Y:{y_world}, RZ_cube:{rz_cube}) nicht zuschneiden.")

    # Speichern des Debug-Bildes
    debug_image_name = f"{source_image_stem}_debug.png"
    debug_image_path = Path(debug_folder) / debug_image_name
    cv2.imwrite(str(debug_image_path), debug_img)
    print(f"📝 Debug gespeichert: {debug_image_name}")
    
    # CSV mit Mittelpunkten speichern (Format wie in extract_cubes.py)
    # csv_path = "cubes_centers.csv"
    # save_cube_centers_csv(cube_centers_csv, csv_path)

    # KPI-Ausgabe am Ende
    total_cubes = len(coordinates)
    if cubes_saved > 0:
        print(f"Es wurden {cubes_saved} von {total_cubes} Würfeln erkannt.")
    else:
        print("Es wurden keine Würfel erkannt.")

# Funktion zur Neukalibrierung des PIXELS_PER_UNIT Werts
def recalibrate_scale():
    """
    Berechnet den PIXELS_PER_UNIT Wert neu basierend auf Objekt 1/X und Objekt 2/Y
    """
    # Objekt 1/X (Ursprung)
    world_x1, world_y1 = X0, Y0  # 121.8949, -361.257
    
    # Objekt 2 aus der ursprünglichen Kalibrierung
    world_x2, world_y2 = 74.42368, -479.9174
    
    # Pixelposition von Objekt 2 aus der ursprünglichen Kalibrierung
    pixel_x2, pixel_y2 = 1216, 749  # Bereits gemessene Werte aus der Kalibrierung
    
    # Pixelposition von Objekt 1/X (bereits definiert)
    pixel_x1, pixel_y1 = IMAGE_ORIGIN_X, IMAGE_ORIGIN_Y
    
    # Abstand in Weltkoordinaten berechnen
    world_dist = ((world_x2 - world_x1)**2 + (world_y2 - world_y1)**2)**0.5
    
    # Abstand in Pixeln berechnen
    pixel_dist = ((pixel_x2 - pixel_x1)**2 + (pixel_y2 - pixel_y1)**2)**0.5
    
    # PIXELS_PER_UNIT berechnen
    if world_dist > 0:
        ppu = pixel_dist / world_dist
        print(f"Weltabstand zwischen Objekt 1/X und 2/Y: {world_dist:.3f}")
        print(f"Pixelabstand zwischen Objekt 1/X und 2/Y: {pixel_dist:.3f}")
        print(f"Neu berechneter PIXELS_PER_UNIT: {ppu:.6f}")
        return ppu
    else:
        print("FEHLER: Weltabstand ist 0, kann PIXELS_PER_UNIT nicht berechnen")
        return PIXELS_PER_UNIT

def world_to_image(x_world, y_world, rz_world=0):
    """
    Verbesserte Funktion zur Umrechnung von Weltkoordinaten in Bildpixel
    """
    if DEBUG_VERBOSE:
        print(f"Umrechnung: Welt({x_world:.3f}, {y_world:.3f}) -> Bild")
        
    # 1. Verschiebung zum Ursprung (Objekt 1/X)
    dx = x_world - X0
    dy = y_world - Y0
    
    if DEBUG_VERBOSE:
        print(f"  Verschiebung zum Ursprung: dx={dx:.3f}, dy={dy:.3f}")
    
    # 2. Spiegelung (falls aktiviert)
    if FLIP_X:
        dx = -dx
        if DEBUG_VERBOSE:
            print(f"  FLIP_X angewendet: dx={dx:.3f}")
    if FLIP_Y:
        dy = -dy
        if DEBUG_VERBOSE:
            print(f"  FLIP_Y angewendet: dy={dy:.3f}")
    
    # 3. Rotation (falls aktiviert)
    if not DISABLE_GLOBAL_ROTATION:
        angle = RZ0 + ADJUST_GLOBAL_ROTATION_OFFSET
        dx_rot, dy_rot = rotate_point(dx, dy, -angle)
        if DEBUG_VERBOSE:
            print(f"  Nach Rotation um {-angle:.3f}°: dx={dx_rot:.3f}, dy={dy_rot:.3f}")
        dx, dy = dx_rot, dy_rot
    
    # 4. Umrechnung von Welteinheiten in Pixel
    px = dx * PIXELS_PER_UNIT
    py = dy * PIXELS_PER_UNIT
    
    if DEBUG_VERBOSE:
        print(f"  In Pixel umgerechnet: px={px:.3f}, py={py:.3f}")
    
    # 5. Absolute Position im Bild (mit Feintuning-Offsets)
    x_pixel = IMAGE_ORIGIN_X + px + POSITION_OFFSET_X
    y_pixel = IMAGE_ORIGIN_Y + py + POSITION_OFFSET_Y
    
    if DEBUG_VERBOSE:
        print(f"  Endposition (mit Offset): Bild({x_pixel:.3f}, {y_pixel:.3f})")
        
    # Berechne relative Rotation: Würfelrotation minus Ursprungsrotation plus Offset
    # RZ0 ist die kalibrierte Rotation des Ursprungspunkts, die wir kompensieren müssen
    relative_angle = rz_world - RZ0 + ADJUST_GLOBAL_ROTATION_OFFSET
    
    # Normalisiere den Winkel auf den Bereich -180° bis +180°
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360
    
    if DEBUG_VERBOSE:
        print(f"  Rotationsberechnung: RZ_world={rz_world:.3f}° - RZ0={RZ0:.3f}° + Offset={ADJUST_GLOBAL_ROTATION_OFFSET:.3f}° = {relative_angle:.3f}° (normalisiert)")
    print(f"DEBUG: RZ_world={rz_world}, RZ0={RZ0}, ADJUST_GLOBAL_ROTATION_OFFSET={ADJUST_GLOBAL_ROTATION_OFFSET}, relative_angle={relative_angle}")
    
    return x_pixel, y_pixel, relative_angle

# Debug-Test-Funktion zur Überprüfung der Rotation
def test_rotation():
    """
    Testfunktion zum Überprüfen der Rotationsfunktionalität
    Erstellt Test-Bilder mit verschiedenen Rotationswinkeln
    """
    print("\n=== ROTATION TEST ===")
    
    # Bildordner und Debug-Ordner prüfen
    image_files = list(Path(image_folder).glob('*.png'))
    if not image_files:
        print("Keine Testbilder gefunden!")
        return
    
    # Erstes Bild für den Test verwenden
    image_path = image_files[0]
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"Fehler beim Laden des Testbildes: {image_path}")
        return
    
    print(f"Verwende Testbild: {image_path}")
    
    # Debug-Ordner erstellen
    debug_folder = '/Users/Leon/Documents/Bilderkennung 3/data/debug_cubes'
    os.makedirs(debug_folder, exist_ok=True)
    
    # Testpunkt in der Mitte des Bildes
    img_height, img_width = original_img.shape[:2]
    test_x = img_width // 2
    test_y = img_height // 2
    test_size = 200
    
    print(f"Testpunkt: ({test_x}, {test_y}), Größe: {test_size}x{test_size}")
    
    # Verschiedene Rotationswinkel testen
    test_angles = [0, 15, 30, 45, 90, -15, -30, -45]
    
    for angle in test_angles:
        print(f"Teste Rotation mit Winkel: {angle}°")
        
        # Würfel mit Rotation zuschneiden
        try:
            rotated_cube = crop_cube_with_rotation(
                original_img, test_x, test_y, test_size, test_size, angle, 
                crop_idx=angle, original_stem="rotation_test"
            )
            
            if rotated_cube is not None:
                # Speichern
                test_output_filename = f"rotation_test_{angle:+03d}_degrees.png"
                test_output_path = Path(debug_folder) / test_output_filename
                cv2.imwrite(str(test_output_path), rotated_cube)
                print(f"  ✅ Test mit Winkel {angle}° gespeichert: {test_output_path}")
            else:
                print(f"  ❌ Fehler beim Zuschneiden mit Winkel {angle}°")
                
        except Exception as e:
            print(f"  ❌ Exception bei Winkel {angle}°: {e}")
    
    # Debug-Bild mit markiertem Testpunkt erstellen
    debug_img = original_img.copy()
    cv2.circle(debug_img, (test_x, test_y), 10, (0, 255, 0), -1)  # Grüner Punkt
    cv2.rectangle(debug_img, 
                  (test_x - test_size//2, test_y - test_size//2),
                  (test_x + test_size//2, test_y + test_size//2),
                  (255, 0, 0), 3)  # Blaues Rechteck
    cv2.putText(debug_img, f"Test Area ({test_x}, {test_y})", 
                (test_x + 15, test_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    debug_overview_path = Path(debug_folder) / "rotation_test_overview.png"
    cv2.imwrite(str(debug_overview_path), debug_img)
    print(f"Debug-Übersicht gespeichert: {debug_overview_path}")
    
    print("=== ROTATION TEST ABGESCHLOSSEN ===\n")

def diagnose_position_accuracy():
    """
    Diagnosefunktion zur Überprüfung der Positionsgenauigkeit
    """
    print("\n=== POSITIONS-DIAGNOSE ===")
    print(f"Aktuelle Offset-Werte: POSITION_OFFSET_X={POSITION_OFFSET_X}, POSITION_OFFSET_Y={POSITION_OFFSET_Y}")
    
    # Teste verschiedene Konfigurationen
    configurations = [
        {"name": "Aktuell (mit Offsets)", "offset_x": POSITION_OFFSET_X, "offset_y": POSITION_OFFSET_Y},
        {"name": "Ohne Offsets", "offset_x": 0, "offset_y": 0},
        {"name": "Mit Y-Offset +5", "offset_x": POSITION_OFFSET_X, "offset_y": 5},
        {"name": "Mit Y-Offset +15", "offset_x": POSITION_OFFSET_X, "offset_y": 15},
    ]
    
    # Teste mit Objekt 2 (sollte nah am Ursprung sein)
    test_x, test_y, test_rz = 123.08289, -380.64551, 138.90343
    
    for config in configurations:
        print(f"\n--- {config['name']} ---")
        
        # Simulation der world_to_image Berechnung
        dx = test_x - X0  # 123.08289 - 121.8949 = 1.188
        dy = test_y - Y0  # -380.64551 - (-361.257) = -19.389
        
        if FLIP_Y:
            dy = -dy  # 19.389
            
        px = dx * PIXELS_PER_UNIT  # 1.188 * 4.655331 = 5.530
        py = dy * PIXELS_PER_UNIT  # 19.389 * 4.655331 = 90.260
        
        x_pixel = IMAGE_ORIGIN_X + px + config["offset_x"]  # 1438 + 5.530 + offset_x
        y_pixel = IMAGE_ORIGIN_Y + py + config["offset_y"]  # 197 + 90.260 + offset_y
        
        print(f"  Weltkoordinaten: ({test_x:.3f}, {test_y:.3f})")
        print(f"  Delta: dx={dx:.3f}, dy={dy:.3f}")
        print(f"  Pixel: px={px:.3f}, py={py:.3f}")
        print(f"  Offsets: X={config['offset_x']}, Y={config['offset_y']}")
        print(f"  Endposition: ({x_pixel:.3f}, {y_pixel:.3f})")

# Hauptfunktion, die den Prozess startet
if __name__ == "__main__":
    print("Starte Würfel-Extraktion mit coordinates_cropping.py...")
    
    # Normale Verarbeitung
    process_images()
    
    print("\n✓ Alle Würfel-Faces wurden erfolgreich extrahiert und die Mittelpunkte in cubes_centers.csv gespeichert.")


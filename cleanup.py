#!/usr/bin/env python3
import os
from pathlib import Path

def main():
    # Diese Ordner (und alles drin) sollen verschont bleiben:
    keep_dirs = {"hig_raw_zielbild", "hig_raw_cubes"}

    # Basisverzeichnis ist der Ordner, in dem cleanup.py liegt
    base_dir = Path(__file__).parent.resolve()
    print(f"Aufräumen in: {base_dir}\n")

    # Durchlaufe rekursiv alle Dateien unter base_dir
    for datei in base_dir.rglob("*"):
        if not datei.is_file():
            continue

        # Löschen nur, wenn Endung .png oder .csv (Case-Insensitive)
        if datei.suffix.lower() not in {".png", ".csv"}:
            continue

        # Falls einer der keep_dirs-Namen im relativen Pfad steckt, skippen wir:
        rel_parts = set(datei.relative_to(base_dir).parts)
        if keep_dirs & rel_parts:
            # z.B. "hig_raw_zielbild" oder "hig_raw_cubes" ist im Pfad
            continue

        # Ansonsten: Datei löschen
        try:
            datei.unlink()
            print(f"✔ Gelöscht: {datei}")
        except Exception as e:
            print(f"✘ Fehler beim Löschen von {datei}: {e}")

    print("\nAufräum-Prozess abgeschlossen.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_script(script_name: str, base_dir: Path):
    """
    Versucht, das Script mit Namen `script_name` im Verzeichnis `base_dir` auszuführen.
    Bricht ab, falls das Skript nicht existiert oder mit einem Fehlercode zurückkehrt.
    """
    # 1) Erstelle den absoluten Pfad zum Unter-Skript
    script_path = (base_dir / script_name).resolve()

    # 2) Prüfen, ob die Datei überhaupt existiert
    if not script_path.is_file():
        print(f"[FEHLER] Konnte '{script_name}' nicht finden:\n    {script_path}")
        sys.exit(1)

    # 3) Starte den Unterprozess mit sys.executable (die gleiche Python-Installation)
    print(f"\n>>> Starte Script: {script_path.name}")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True
        )
        print(f"[OK] '{script_path.name}' erfolgreich beendet (Exit-Code={result.returncode}).")
    except subprocess.CalledProcessError as e:
        print(f"[FEHLER] '{script_path.name}' endete mit Fehlercode {e.returncode}.")
        sys.exit(e.returncode)


def main():
    # 1) Basis-Verzeichnis: das Verzeichnis, in dem diese main.py liegt
    base_dir = Path(__file__).parent.resolve()
    print(f"Arbeitsverzeichnis: {base_dir}")

    # 2) Liste der Skriptnamen (in der gewünschten Reihenfolge):
    scripts = [
        "hig_raw_zielbild._in_png.py",   # 1. hig_raw_zielbild._in_png.py
        "extract_zielbild.py",          # 2. extract_zielbild.py
        "hig_raw_cubes._in_png.py",      # 3. hig_raw_cubes._in_png.py
        "coordinates_cropping.py",      # 4. Ersatz für extract_cubes.py
        "template_matching.py"          # 5. template_matching.py
    ]

    # 3) Jeden Eintrag mit run_script aufrufen
    for script_name in scripts:
        run_script(script_name, base_dir)

    print("\nAlle Skripte wurden erfolgreich durchlaufen.")


if __name__ == "__main__":
    main()

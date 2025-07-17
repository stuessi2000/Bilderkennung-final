import csv
import sys
from pathlib import Path

# === Konfigurationspfade ===
BASE_DIR = Path(r"\\192.168.0.1\Vision_Daten\DU_3")
FOLDERS = {
    "Koordinaten": BASE_DIR / "Koordinaten",
    "Koordinaten_TM": BASE_DIR / "Koordinaten_template_matching",
    "Wuerfelbilder": BASE_DIR / "Würfelbilder",
}
SOURCE_TXT = FOLDERS["Koordinaten"] / "objekte.txt"

# === Hilfsfunktionen ===

def txt_to_csv_print(src: Path):
    """
    Liest die TXT, wandelt sie in CSV-Zeilen um
    und schreibt die CSV direkt auf stdout.
    """
    with src.open("r", encoding="utf-8") as f:
        writer = csv.writer(sys.stdout)
        for line in f:
            parts = line.strip().split(",")
            if parts and parts[0]:
                writer.writerow(parts)


def clear_file(src: Path):
    """Leert die Quelldatei."""
    src.open("w", encoding="utf-8").close()


def clear_txt_and_delete_others(coord_dir: Path):
    """
    Leert alle .txt im Koordinaten-Ordner und löscht alle anderen Dateien.
    """
    for p in coord_dir.iterdir():
        if p.is_file():
            if p.suffix.lower() == ".txt":
                clear_file(p)
                print(f"✔ Datei geleert: {p}")
            else:
                p.unlink()
                print(f"✖ Datei gelöscht: {p}")


def clear_folder(folder: Path):
    """Löscht alle Dateien in einem Ordner, aber nicht den Ordner selbst."""
    if folder.exists() and folder.is_dir():
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()
                print(f"🗑️ Datei gelöscht: {f}")


def clear_wuerfelbilder_subfolders(base_dir: Path):
    """
    Leert alle Dateien in den Unterordnern von Würfelbilder (z.B. png_cubes, faces, debug_cubes_output).
    """
    w_dir = base_dir
    for sub in w_dir.iterdir():
        if sub.is_dir():
            files = list(sub.iterdir())
            if not files:
                print(f"ℹ Keine Dateien in {sub}, überspringe.")
                continue
            for f in files:
                if f.is_file():
                    f.unlink()
                    print(f"🗑️ Datei gelöscht: {f}")
                elif f.is_dir():
                    # optional: rekursiv löschen von Unterordnern
                    for inner in f.rglob('*'):
                        if inner.is_file():
                            inner.unlink()
                            print(f"🗑️ Datei gelöscht: {inner}")
                    f.rmdir()
                    print(f"📂 Leerer Unterordner entfernt: {f}")


def cleanup():
    """
    Führt die Aufräumarbeiten durch:
      1) .txt im Koordinaten-Ordner leeren, restliche Dateien löschen
      2) faces und png_cubes komplett leeren
      3) .hig-Dateien im Würfelbilder-Ordner löschen
      4) Alle Unterordner in Würfelbilder leeren, aber nicht löschen
    """
    # 1) Koordinaten löschen
    clear_txt_and_delete_others(FOLDERS["Koordinaten"])
    # 2) faces und png_cubes leeren
    clear_folder(Path("faces"))
    clear_folder(Path("png_cubes"))
    # 3) .hig im Wuerfelbilder-Ordner löschen
    base = FOLDERS["Wuerfelbilder"]
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() == ".hig":
            p.unlink()
            print(f"🗑️ HIG gelöscht: {p}")
    # 4) Unterordner leeren
    clear_wuerfelbilder_subfolders(FOLDERS["Wuerfelbilder"])

# Hinweis: Dieses Skript führt keinen automatischen Cleanup aus,
# wenn es direkt gestartet wird. Rufe stattdessen die Funktion cleanup()
# explizit aus einem anderen Skript (z.B. main.py) auf, wenn du
# die Aufräumarbeiten durchführen möchtest.: Dieses Skript führt keinen automatischen Cleanup aus,
# wenn es direkt gestartet wird. Rufe stattdessen die Funktion cleanup()
# explizit aus einem anderen Skript (z.B. main.py) auf, wenn du
# die Aufräumarbeiten durchführen möchtest.
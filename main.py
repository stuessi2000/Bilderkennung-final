#!/usr/bin/env python3
import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer

try:
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Das watchdog-Paket ist nicht installiert. Bitte führe 'pip install watchdog' aus.")
    sys.exit(1)
# —————————————————————————————
# 1) Pfade
# —————————————————————————————
ZIEL    = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Zielbild")
WUERFEL = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Würfelbilder")
KOORD   = Path(r"\\192.168.0.1\Vision_Daten\DU_3\Koordinaten")

# —————————————————————————————
# 2) Helfer zum Starten externer Skripte
# —————————————————————————————
def run(script_name: str):
    print(f">>> Starte {script_name}")
    subprocess.run([sys.executable, script_name], check=True)

# —————————————————————————————
# 3) Importiere deine Cleanup-Funktion
# —————————————————————————————
# from objekte_export_cleanup import cleanup
import cleanup

# —————————————————————————————
# 4) Watchdog-Handler mit Phasen-Logik
# —————————————————————————————
class WorkflowHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.phase = 1
        print(">> Watchdog gestartet – Phase 1 (Zielbild) aktiv")

    def on_created(self, event):
        path = Path(event.src_path)

        # — Phase 1: Erst auf neue .hig im Zielbild-Ordner warten
        if self.phase == 1 and path.parent == ZIEL and path.suffix.lower() == ".hig":
            run("hig_raw_zielbild._in_png.py")
            run("extract_zielbild.py")
            print("→ Wechsel zu Phase 2 (Würfel + Koordinaten)")
            self.phase = 2
            return

        # — Phase 2: Dann auf .hig in Würfel-Ordner UND objekte.txt warten
        if self.phase == 2:
            hig_ok = any(WUERFEL.glob("*.hig"))
            txt_ok = (KOORD / "objekte.txt").exists()
            if hig_ok and txt_ok:
                run("hig_raw_cubes._in_png.py")
                run("coordinates_cropping.py")
                run("sift-template.py")
                run("rotation_optimizer.py")

                # 30-Sekunden-Puffer, bevor gecleant wird
                time.sleep(60)

                # Cleanup aufrufen
                print(">>> Starte Cleanup")
                cleanup.cleanup()
                print("→ Zyklus abgeschlossen – bleibe in Phase 2 und warte auf neue Dateien")

# —————————————————————————————
# 5) Observer starten
# —————————————————————————————
if __name__ == "__main__":
    handler = WorkflowHandler()
    observer = Observer()
    for folder in (ZIEL, WUERFEL, KOORD):
        observer.schedule(handler, str(folder), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
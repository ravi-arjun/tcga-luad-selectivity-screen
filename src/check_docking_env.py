import os
import subprocess
import sys
import urllib.request

# 1. Install pure-wheel Python cheminformatics & structural biology packages (NO 'vina' via pip!)
PIP_PACKAGES = {
    "meeko": "meeko",
    "gemmi": "gemmi",
    "rdkit": "rdkit",
    "Bio": "biopython",
}

missing = []
for import_name, pip_name in PIP_PACKAGES.items():
    try:
        __import__(import_name)
        print(f"[OK] {import_name} is already installed.")
    except ImportError:
        print(f"[MISSING] {import_name} -> queueing '{pip_name}'")
        missing.append(pip_name)

if missing:
    print(f"\nInstalling Python packages: {missing}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# 2. Download official pre-compiled AutoDock Vina 1.2.7 Windows binary
os.makedirs("bin", exist_ok=True)
vina_exe = os.path.join("bin", "vina.exe")
vina_url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.exe"

if not os.path.exists(vina_exe):
    print(f"\nDownloading official AutoDock Vina 1.2.7 Windows executable to '{vina_exe}'...")
    urllib.request.urlretrieve(vina_url, vina_exe)

# 3. Verify Vina executable responds cleanly
version_out = subprocess.check_output([vina_exe, "--version"], text=True).strip()
print(f"[OK] AutoDock Vina Engine Ready: {version_out}")
print("\nStage 4 Docking Environment is 100% READY!")
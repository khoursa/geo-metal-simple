from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import rasterio
import numpy as np
import os
import math
import requests
import zipfile

# =========================
#   CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier où les .tif seront extraits
DEM_FOLDER = os.path.join(BASE_DIR, "dem")

# 🔴 METS ICI TON LIEN DRIVE DIRECT (dem.zip)
DEM_ZIP_URL = (
    "https://drive.google.com/uc?export=download&id=1Y2oOpHZz5D1o6SodGkiIiCKLHlRRI2bF"
)

# =========================
#   MODELES API
# =========================

class ScanRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 100.0

class Point(BaseModel):
    lat: float
    lon: float
    score: float

class ScanResponse(BaseModel):
    best_point: Point
    candidates: List[Point]
    metal_found: bool


# =========================
#   APP FASTAPI
# =========================

app = FastAPI(title="Geo-Metal Detector DEM", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
#   TÉLÉCHARGEMENT DEM.ZIP
# =========================

def ensure_dem_unzipped():
    """
    Vérifie si des .tif existent dans DEM_FOLDER.
    Si non : télécharge dem.zip depuis Google Drive, puis unzip dans DEM_FOLDER.
    """
    # Si dossier DEM existe déjà et contient des .tif -> on ne fait rien
    if os.path.isdir(DEM_FOLDER):
        tif_files = [f for f in os.listdir(DEM_FOLDER) if f.lower().endswith(".tif")]
        if tif_files:
            print(f"[DEM] {len(tif_files)} fichiers TIF déjà présents dans {DEM_FOLDER}")
            return

    os.makedirs(DEM_FOLDER, exist_ok=True)

    zip_path = os.path.join(BASE_DIR, "dem.zip")

    # Télécharger dem.zip si pas déjà présent
    if not os.path.isfile(zip_path):
        print(f"[DEM] Téléchargement de dem.zip depuis Drive...")
        try:
            resp = requests.get(DEM_ZIP_URL, stream=True, timeout=300)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("[DEM] dem.zip téléchargé avec succès.")
        except Exception as e:
            print(f"[DEM] ERREUR téléchargement dem.zip : {e}")
            return

    # Décompression
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DEM_FOLDER)
        print(f"[DEM] dem.zip extrait dans {DEM_FOLDER}")
    except Exception as e:
        print(f"[DEM] ERREUR unzip dem.zip : {e}")
        return


# =========================
#   OUTILS DEM + GEO
# =========================

def load_dem_tile(lat: float, lon: float):
    """
    S'assure d'abord que les TIF sont présents (download + unzip si besoin),
    puis cherche un .tif qui couvre lat/lon.
    """
    ensure_dem_unzipped()

    if not os.path.isdir(DEM_FOLDER):
        print(f"[DEM] Dossier DEM inexistant après unzip: {DEM_FOLDER}")
        return None

    for fname in os.listdir(DEM_FOLDER):
        if not fname.lower().endswith(".tif"):
            continue
        path = os.path.join(DEM_FOLDER, fname)
        try:
            with rasterio.open(path) as src:
                left, bottom, right, top = src.bounds
                # On suppose que le TIF est en WGS84 (lon/lat)
                if (left <= lon <= right) and (bottom <= lat <= top):
                    print(f"[DEM] Tuile trouvée: {path}")
                    return path
        except Exception as e:
            print(f"[DEM] Erreur ouverture {path}: {e}")
            continue

    print("[DEM] Aucune tuile ne couvre ce point.")
    return None


def slope_from_window(values: np.ndarray) -> float:
    """
    Pente locale normalisée (0–1) à partir d'une fenêtre 3x3.
    Sécurisée pour les bords.
    """
    if values is None or np.isnan(values).any():
        return 0.0
    if values.shape[0] < 3 or values.shape[1] < 3:
        return 0.0

    dzdx = (values[1, 2] - values[1, 0]) / 2.0
    dzdy = (values[2, 1] - values[0, 1]) / 2.0
    slope = math.sqrt(dzdx**2 + dzdy**2)
    return min(slope, 50.0) / 50.0


def roughness(values: np.ndarray) -> float:
    """
    Rugosité locale normalisée (0–1) à partir de l'écart-type.
    """
    if values is None or np.isnan(values).any():
        return 0.0
    return float(min(np.std(values) / 20.0, 1.0))


def geo_score(lat: float, lon: float) -> float:
    """
    Score géologique 0–100 basé sur la pente + rugosité autour du point.
    Utilise le DEM réel.
    """
    tif_path = load_dem_tile(lat, lon)
    if tif_path is None:
        # Si on n'a aucune tuile, on renvoie un score moyen
        return 50.0

    try:
        with rasterio.open(tif_path) as src:
            row, col = src.index(lon, lat)
            w = rasterio.windows.Window(col - 1, row - 1, 3, 3)
            z = src.read(1, window=w)

            if src.nodata is not None:
                z = np.where(z == src.nodata, np.nan, z)
    except Exception as e:
        print(f"[DEM] Erreur lecture DEM pour ({lat}, {lon}) : {e}")
        return 50.0

    s = slope_from_window(z)
    r = roughness(z)
    score_0_1 = 0.6 * s + 0.4 * r
    return float(score_0_1 * 100.0)


def offset_lat(m: float) -> float:
    return m / 111320.0

def offset_lon(m: float, lat: float) -> float:
    return m / (111320.0 * math.cos(math.radians(lat)))


# =========================
#   ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"status": "ok", "message": "Geo-Metal DEM avec téléchargement Drive prêt."}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    """
    Scanne un disque de rayon r autour du clic, calcule un score géologique
    basé sur le DEM réel et renvoie :
      - best_point
      - candidates (top 10)
      - metal_found si score >= 60
    """
    lat0 = req.lat
    lon0 = req.lon
    r = float(max(10.0, req.radius_m))

    print(f"[SCAN] lat={lat0}, lon={lon0}, rayon={r} m")

    candidates: List[Point] = []

    GRID_STEP_M = max(r / 6.0, 20.0)
    off_range = np.arange(-r, r + GRID_STEP_M, GRID_STEP_M)

    for off_x in off_range:
        for off_y in off_range:
            dist = math.hypot(off_x, off_y)
            if dist > r:
                continue

            lat = lat0 + offset_lat(off_y)
            lon = lon0 + offset_lon(off_x, lat0)

            s = geo_score(lat, lon)
            candidates.append(Point(lat=lat, lon=lon, score=s))

    if not candidates:
        print("[SCAN] Aucun candidat, fallback centre.")
        c_score = geo_score(lat0, lon0)
        best = Point(lat=lat0, lon=lon0, score=c_score)
        return ScanResponse(
            best_point=best,
            candidates=[],
            metal_found=(c_score >= 60.0)
        )

    best = max(candidates, key=lambda p: p.score)
    candidates_sorted = sorted(candidates, key=lambda p: p.score, reverse=True)
    top_candidates = candidates_sorted[:10]

    metal_found = best.score >= 60.0

    print(f"[SCAN] best_score={best.score:.1f}, metal_found={metal_found}")

    return ScanResponse(
        best_point=best,
        candidates=top_candidates,
        metal_found=metal_found
    )

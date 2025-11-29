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

# Dossier où les TIF seront extraits
DEM_FOLDER = os.path.join(BASE_DIR, "dem")

# 🔴 Lien direct Google Drive vers dem.zip (à adapter si tu changes d’ID)
DEM_ZIP_URL = (
    "https://drive.google.com/uc?export=download&id=1Y2oOpHZz5D1o6SodGkiIiCKLHlRRI2bF"
)

# =========================
#   MODELES
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

app = FastAPI(title="Geo-Metal DEM v3", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#   TÉLÉCHARGEMENT / UNZIP
# =========================

def ensure_dem_unzipped():
    """
    Vérifie s'il y a déjà au moins un .tif dans DEM_FOLDER (ou sous-dossiers).
    Sinon télécharge dem.zip depuis Drive et l'extrait dans DEM_FOLDER.
    """
    if os.path.isdir(DEM_FOLDER):
        for root, _, files in os.walk(DEM_FOLDER):
            if any(f.lower().endswith(".tif") for f in files):
                print(f"[DEM] TIF déjà présents dans {DEM_FOLDER}")
                return

    os.makedirs(DEM_FOLDER, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, "dem.zip")

    if not os.path.isfile(zip_path):
        print("[DEM] Téléchargement de dem.zip depuis Drive...")
        try:
            resp = requests.get(DEM_ZIP_URL, stream=True, timeout=300)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            print("[DEM] dem.zip téléchargé.")
        except Exception as e:
            print(f"[DEM] ERREUR téléchargement dem.zip : {e}")
            return

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DEM_FOLDER)
        print(f"[DEM] dem.zip extrait dans {DEM_FOLDER}.")
    except Exception as e:
        print(f"[DEM] ERREUR unzip dem.zip : {e}")
        return


# =========================
#   UTIL DEM + SCORE
# =========================

def load_dem_tile(lat: float, lon: float):
    """
    S'assure que les TIF existent.
    Cherche un .tif qui couvre lat/lon (via les bounds).
    Si aucune tuile ne match → prend simplement le premier .tif trouvé.
    """
    ensure_dem_unzipped()

    if not os.path.isdir(DEM_FOLDER):
        print("[DEM] Dossier DEM inexistant après unzip.")
        return None

    tif_paths = []
    for root, _, files in os.walk(DEM_FOLDER):
        for f in files:
            if f.lower().endswith(".tif"):
                tif_paths.append(os.path.join(root, f))

    if not tif_paths:
        print("[DEM] Aucun fichier .tif trouvé dans DEM_FOLDER.")
        return None

    # D'abord : essayer de trouver une tuile dont les bounds couvrent le point
    for path in tif_paths:
        try:
            with rasterio.open(path) as src:
                left, bottom, right, top = src.bounds
                if (left <= lon <= right) and (bottom <= lat <= top):
                    print(f"[DEM] Tuile trouvée par bounds: {path}")
                    return path
        except Exception as e:
            print(f"[DEM] Erreur ouverture {path}: {e}")
            continue

    # Sinon → fallback : on prend le premier TIF
    fallback_path = tif_paths[0]
    print(f"[DEM] Aucune tuile ne couvre le point, fallback sur {fallback_path}")
    return fallback_path


def slope_from_window(values: np.ndarray) -> float:
    """Pente locale normalisée (0–1) à partir d'une fenêtre 3x3."""
    if values is None or np.isnan(values).any():
        return 0.0
    if values.shape[0] < 3 or values.shape[1] < 3:
        return 0.0

    dzdx = (values[1, 2] - values[1, 0]) / 2.0
    dzdy = (values[2, 1] - values[0, 1]) / 2.0
    slope = math.sqrt(dzdx**2 + dzdy**2)
    return min(slope, 50.0) / 50.0


def roughness(values: np.ndarray) -> float:
    """Rugosité locale normalisée (0–1)."""
    if values is None or np.isnan(values).any():
        return 0.0
    return float(min(np.std(values) / 20.0, 1.0))


def geo_score(lat: float, lon: float) -> float:
    """
    Score géologique 0–100 basé sur DEM.
    Si impossible de lire correctement, retourne 50.0 (neutre).
    """
    tif_path = load_dem_tile(lat, lon)
    if tif_path is None:
        return 50.0

    try:
        with rasterio.open(tif_path) as src:
            row, col = src.index(lon, lat)
            # clamp pour rester à l'intérieur (fenêtre 3x3)
            row = max(1, min(row, src.height - 2))
            col = max(1, min(col, src.width - 2))

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
    return {"status": "ok", "message": "Geo-Metal DEM v3 prêt."}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    """
    Scanne un disque de rayon r autour du clic.
    Calcule un score 0–100 à partir du DEM réel.
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


# =========================
#   DEBUG DEM
# =========================

@app.get("/debug_dem")
def debug_dem():
    """
    Endpoint de diagnostic :
    - vérifie que dem.zip a bien été décompressé
    - liste les .tif trouvés
    - essaye d’ouvrir le premier TIF avec rasterio
    - calcule un geo_score test sur (34, -5)
    """
    info = {}

    # 1) TIF présents ?
    ensure_dem_unzipped()

    tif_paths = []
    if os.path.isdir(DEM_FOLDER):
        for root, _, files in os.walk(DEM_FOLDER):
            for f in files:
                if f.lower().endswith(".tif"):
                    tif_paths.append(os.path.join(root, f))

    info["dem_folder"] = DEM_FOLDER
    info["tif_count"] = len(tif_paths)
    info["tif_paths"] = tif_paths[:5]  # on montre max 5 chemins

    # 2) Essayer d’ouvrir le premier TIF
    if tif_paths:
        first_tif = tif_paths[0]
        try:
            with rasterio.open(first_tif) as src:
                info["first_tif"] = {
                    "path": first_tif,
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "bounds": tuple(src.bounds),
                    "nodata": src.nodata,
                }
        except Exception as e:
            info["first_tif_error"] = str(e)
    else:
        info["first_tif"] = None

    # 3) Essayer un geo_score de test
    try:
        test_score = geo_score(34.0, -5.0)
        info["test_score_34_-5"] = test_score
    except Exception as e:
        info["test_score_error"] = str(e)

    return info

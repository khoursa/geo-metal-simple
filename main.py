from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import math
import random

import rasterio
from rasterio.windows import Window
import numpy as np

# ====== CONFIG : dossier DEM pour Render ======
# main.py
# dem/
#   maroc_01.tif
#   maroc_02.tif
#   ...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEM_FOLDER = os.path.join(BASE_DIR, "dem")   # <<< METS TES TIF ICI

# ====== MODELES ======
class ScanRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 100.0   # rayon par défaut 100 m

class Point(BaseModel):
    lat: float
    lon: float
    score: float

class ScanResponse(BaseModel):
    best_point: Point
    candidates: List[Point]
    metal_found: bool

# ====== App FastAPI ====== #
app = FastAPI(title="Geo-Metal Detector", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Petit endpoint pour vérifier facilement
@app.get("/")
def root():
    return {"status": "ok", "service": "geo-metal-detector"}

# ====== INDEX DES TIFS AU DEMARRAGE ======
class DemInfo:
    def __init__(self, path, bounds):
        self.path = path
        self.bounds = bounds  # rasterio.coords.BoundingBox

DEM_FILES: List[DemInfo] = []

def load_dem_index():
    if not os.path.isdir(DEM_FOLDER):
        print(f"[WARN] Dossier DEM introuvable : {DEM_FOLDER}")
        return

    for name in os.listdir(DEM_FOLDER):
        if not name.lower().endswith(".tif"):
            continue
        path = os.path.join(DEM_FOLDER, name)
        try:
            with rasterio.open(path) as ds:
                DEM_FILES.append(DemInfo(path=path, bounds=ds.bounds))
                print(f"[DEM] Chargé: {path} – bounds={ds.bounds}")
        except Exception as e:
            print(f"[DEM] Erreur ouverture {path}: {e}")

load_dem_index()

def find_dem_for_point(lat: float, lon: float) -> DemInfo | None:
    """Trouve le TIF qui contient le point (lon, lat)."""
    for info in DEM_FILES:
        b = info.bounds
        if (b.left <= lon <= b.right) and (b.bottom <= lat <= b.top):
            return info
    return None

# ====== FONCTION SERIEUSE : score à partir du DEM ======
def compute_score_from_dem(lat: float, lon: float, radius_m: float) -> float:
    """
    Lit le DEM autour du point et calcule un score 0–1 basé sur :
    - relief local (min / max)
    - rugosité (écart-type)
    - pente moyenne (gradient)
    """
    dem_info = find_dem_for_point(lat, lon)
    if dem_info is None:
        # hors zone DEM → score 0
        return 0.0

    with rasterio.open(dem_info.path) as ds:
        try:
            row, col = ds.index(lon, lat)  # rasterio: (lon, lat)
        except Exception:
            return 0.0

        # On prend une petite fenêtre autour du point (7x7 px)
        half = 3
        row_min = max(row - half, 0)
        row_max = min(row + half, ds.height - 1)
        col_min = max(col - half, 0)
        col_max = min(col + half, ds.width - 1)

        win = Window(col_min, row_min,
                     col_max - col_min + 1,
                     row_max - row_min + 1)

        data = ds.read(1, window=win, masked=True)

        if data.mask.all():
            return 0.0

        # on travaille en float32
        patch = data.filled(np.nan).astype("float32")

    # ---- métriques terrain ----
    elev_min = float(np.nanmin(patch))
    elev_max = float(np.nanmax(patch))
    elev_std = float(np.nanstd(patch))
    relief = elev_max - elev_min

    # pente moyenne (norme du gradient)
    gy, gx = np.gradient(patch)
    slope_map = np.sqrt(gx ** 2 + gy ** 2)
    slope_mean = float(np.nanmean(slope_map))

    # ---- normalisation 0–1 (heuristique) ----
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    relief_n = clamp01(relief / 50.0)      # 0–50 m de relief local
    rough_n = clamp01(elev_std / 20.0)     # 0–20 m d'écart-type
    slope_n  = clamp01(slope_mean / 10.0)  # pente moyenne saturée

    base = 0.4 * rough_n + 0.4 * relief_n + 0.2 * slope_n
    # petit bruit pour éviter des scores trop rigides
    noisy = base * 0.95 + random.uniform(-0.03, 0.03)

    return round(clamp01(noisy), 3)

# ====== Scan API : utilise maintenant le DEM ====== #
@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest) -> ScanResponse:
    """
    Scan sérieux :
    - on prend le point cliqué (lat, lon)
    - on génère quelques candidats autour (disque ~100 m)
    - chaque candidat lit le DEM et calcule un score 0–1
    - on choisit le meilleur
    - metal_found si score >= 0.6
    """
    lat0 = req.lat
    lon0 = req.lon
    r = req.radius_m  # normalement 100 m

    candidates: List[Point] = []

    # offsets en degrés ~ 50–60 m (selon latitude) → même approche que ta démo
    offsets = [
        (0.0, 0.0),
        (0.0005, 0.0005),
        (-0.0005, 0.0005),
        (0.0005, -0.0005),
        (-0.0005, -0.0005),
    ]

    for dlat, dlon in offsets:
        lat = lat0 + dlat
        lon = lon0 + dlon
        s = compute_score_from_dem(lat, lon, r)
        candidates.append(Point(lat=lat, lon=lon, score=s))

    # sécurité : si aucun candidat → on met le point central avec score 0
    if not candidates:
        candidates.append(Point(lat=lat0, lon=lon0, score=0.0))

    # meilleur point
    best = max(candidates, key=lambda p: p.score)

    # seuil métal : par ex. 0.6
    metal_found = best.score >= 0.6

    # (option) on garde tous les candidats (Android affichera seulement le best)
    return ScanResponse(
        best_point=best,
        candidates=candidates,
        metal_found=metal_found,
    )

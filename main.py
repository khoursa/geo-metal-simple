from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import math
import random

import requests
import numpy as np
import rasterio
from rasterio.windows import Window

# =========================
#   CONFIG DEM
# =========================

# URL DIRECTE (celle que tu m'as donnée)
DEM_URL = "https://drive.google.com/uc?export=download&id=1Y2oOpHZz5D1o6SodGkiIiCKLHlRRI2bF"

# Chemin local sur Render (dans /project/src)
DEM_FOLDER = "dem"
DEM_LOCAL = os.path.join(DEM_FOLDER, "afrique_dem.tif")

# Cache du dataset Rasterio
_dem_ds = None


def ensure_dem_local():
    """
    Vérifie si le DEM existe en local.
    - Si NON : télécharge depuis Google Drive.
    - Si OUI : ne fait rien.
    """
    if os.path.exists(DEM_LOCAL):
        return

    os.makedirs(DEM_FOLDER, exist_ok=True)
    print(f"[DEM] Téléchargement du DEM depuis {DEM_URL} ...")

    with requests.get(DEM_URL, stream=True) as r:
        r.raise_for_status()
        with open(DEM_LOCAL, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[DEM] Fichier téléchargé → {DEM_LOCAL}")


def get_dem_dataset():
    """
    Ouvre le DEM avec Rasterio (une seule fois, puis cache en mémoire).
    """
    global _dem_ds
    if _dem_ds is None:
        ensure_dem_local()
        _dem_ds = rasterio.open(DEM_LOCAL)
        print("[DEM] Dataset ouvert, CRS:", _dem_ds.crs, "res:", _dem_ds.res)
    return _dem_ds


# =========================
#   MODELES API
# =========================

class ScanRequest(BaseModel):
    lat: float      # latitude en degrés
    lon: float      # longitude en degrés
    radius_m: float = 100.0  # rayon de scan en mètres


class Point(BaseModel):
    lat: float
    lon: float
    score: float


class ScanResponse(BaseModel):
    best_point: Point
    candidates: List[Point]
    metal_found: bool
    message: str


# =========================
#   LOGIQUE METAL / DEM
# =========================

def _sample_dem(lat: float, lon: float, radius_m: float) -> np.ndarray:
    """
    Lit une petite fenêtre du DEM autour du point (lat, lon)
    en fonction du rayon (en mètres).
    Retourne un tableau numpy 2D d'altitudes.
    """
    ds = get_dem_dataset()

    # ds.index attend (x, y) = (lon, lat)
    row, col = ds.index(lon, lat)

    # approx : 1° ≈ 111320 m
    res_x_deg, res_y_deg = ds.res
    mean_res_deg = (abs(res_x_deg) + abs(res_y_deg)) / 2.0
    deg_per_meter = 1.0 / 111320.0
    radius_deg = radius_m * deg_per_meter

    # rayon en pixels
    radius_px = max(
        2,
        int(radius_deg / mean_res_deg) + 1
    )

    # fenêtre Rasterio
    w = Window(
        col - radius_px,
        row - radius_px,
        2 * radius_px + 1,
        2 * radius_px + 1,
    )

    data = ds.read(
        1,
        window=w,
        boundless=True,
        fill_value=np.nan
    )

    return data


def compute_metal_score(lat: float, lon: float, radius_m: float) -> float:
    """
    Calcule un score de 0 à 1 à partir du relief :
    - plus le terrain est "cassé" (relief & rugosité),
      plus on augmente le score (hypothèse : présence de structures rocheuses/minéralisées).
    """
    dem_window = _sample_dem(lat, lon, radius_m)

    valid = np.isfinite(dem_window)
    if valid.sum() < 10:
        # Pas assez de données
        return 0.1

    vals = dem_window[valid]

    # paramètres simples
    elev_range = float(vals.max() - vals.min())     # différence max-min
    elev_std = float(vals.std())                    # rugosité

    # Normalisation maison (à ajuster selon ton DEM)
    # On suppose qu'un relief entre 5m et 100m est intéressant.
    range_score = max(0.0, min(1.0, elev_range / 100.0))
    rough_score = max(0.0, min(1.0, elev_std / 30.0))

    # Combinaison + petit bruit pour ne pas avoir des 0/1 parfaits
    base = 0.6 * range_score + 0.4 * rough_score
    noisy = base + random.uniform(-0.05, 0.05)

    # Clamp entre 0 et 1
    score = max(0.0, min(1.0, noisy))
    return round(score, 3)


def generate_candidates(req: ScanRequest) -> List[Point]:
    """
    Génère quelques candidats autour du clic :
    - le point exact
    - + 4 voisins (N, S, E, O) à ~ 30 m
    """
    deg_per_meter = 1.0 / 111320.0
    offset_deg = 30.0 * deg_per_meter

    offsets = [
        (0.0, 0.0),                  # centre
        (offset_deg, 0.0),           # nord
        (-offset_deg, 0.0),          # sud
        (0.0, offset_deg),           # est
        (0.0, -offset_deg),          # ouest
    ]

    candidates: List[Point] = []
    for dlat, dlon in offsets:
        lat = req.lat + dlat
        lon = req.lon + dlon
        s = compute_metal_score(lat, lon, req.radius_m)
        candidates.append(Point(lat=lat, lon=lon, score=s))

    return candidates


# =========================
#   APP FASTAPI
# =========================

app = FastAPI(title="Geo-Metal Simple", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "geo-metal-simple"}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest) -> ScanResponse:
    """
    Endpoint principal :
    - lit le DEM
    - calcule un score autour du clic et de quelques voisins
    - renvoie le meilleur point + message
    """
    print(f"[SCAN] lat={req.lat}, lon={req.lon}, radius_m={req.radius_m}")

    # Génère les candidats & cherche le meilleur
    candidates = generate_candidates(req)
    best = max(candidates, key=lambda p: p.score)

    # Seuil métal probable
    threshold = 0.6
    metal_found = best.score >= threshold

    if metal_found:
        message = (
            "Métal probable dans cette zone. "
            f"Score = {best.score:.2f} (seuil {threshold})."
        )
    else:
        message = (
            "Faible probabilité de métal dans cette zone. "
            f"Score = {best.score:.2f} (seuil {threshold})."
        )

    return ScanResponse(
        best_point=best,
        candidates=candidates,
        metal_found=metal_found,
        message=message,
    )

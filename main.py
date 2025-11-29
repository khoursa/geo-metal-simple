from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import math
import random

# ====== Modèles Pydantic ====== #
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
    message: str


# ====== App FastAPI ====== #
app = FastAPI(title="Geo Metal Simple", version="1.0")


@app.get("/")
def root():
    return {"message": "Geo Metal Detector API is working"}


# ====== Faux modèle de métal (DEMO sans DEM) ====== #
def compute_score(lat: float, lon: float, radius_m: float) -> float:
    """
    Petit modèle mathématique simple :
    - combine sin / cos de lat/lon
    - ajoute un effet du rayon
    - ajoute un peu de bruit aléatoire
    Retourne un score entre 0 et 1.
    """
    base = (math.sin(lat * 10.0) + math.cos(lon * 10.0)) / 2.0
    radius_factor = max(0.1, min(1.0, radius_m / 200.0))
    noisy = base * 0.7 + radius_factor * 0.3 + random.uniform(-0.05, 0.05)
    return max(0.0, min(1.0, noisy))


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest) -> ScanResponse:
    """
    Scan simple :
    - prend le point cliqué (lat, lon, radius_m)
    - génère quelques candidats autour
    - calcule un score pour chacun
    - choisit le meilleur
    - décide si 'metal_found' ou non selon un seuil
    """
    candidates: List[Point] = []

    # offsets en degrés (~50–60 m selon latitude)
    offsets = [
        (0.0, 0.0),
        (0.0005, 0.0005),
        (-0.0005, 0.0005),
        (0.0005, -0.0005),
        (-0.0005, -0.0005),
    ]

    for dlat, dlon in offsets:
        lat = req.lat + dlat
        lon = req.lon + dlon
        s = compute_score(lat, lon, req.radius_m)
        candidates.append(Point(lat=lat, lon=lon, score=round(s, 3)))

    # meilleur point
    best = max(candidates, key=lambda p: p.score)

    # seuil : 0.6 = métal probable
    metal_found = best.score >= 0.6
    if metal_found:
        message = "Métal probable détecté dans cette zone."
    else:
        message = "Faible probabilité de métal dans cette zone."

    return ScanResponse(
        best_point=best,
        candidates=candidates,
        metal_found=metal_found,
        message=message,
    )

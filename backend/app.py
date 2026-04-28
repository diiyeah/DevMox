"""AquaWatch backend with Earth Engine imagery analysis and bundled ML inference."""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import ee
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .ml_models import PrebuiltWaterQualityModels, label_from_score
except ImportError:
    from ml_models import PrebuiltWaterQualityModels, label_from_score

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aquawatch")

app = FastAPI(
    title="AquaWatch API",
    description="Water pollution detection with Sentinel-2 imagery, Google Earth Engine, and bundled ML models.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_COVER_THRESHOLD = 20
DEFAULT_BUFFER_METERS = 5000
MODEL_BUNDLE = PrebuiltWaterQualityModels()
COLOR_MAP = {"Safe": "#27ae60", "Moderate": "#f39c12", "Polluted": "#e74c3c"}
GEE_STATE = {"initialized": False, "project_id": os.getenv("GEE_PROJECT_ID", ""), "error": None}


def initialize_gee(force: bool = False) -> bool:
    if GEE_STATE["initialized"] and not force:
        return True

    key_json = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    project_id = os.getenv("GEE_PROJECT_ID", "")

    try:
        if key_json:
            key_data = json.loads(key_json)
            credentials = ee.ServiceAccountCredentials(
                email=key_data["client_email"],
                key_data=json.dumps(key_data),
            )
            ee.Initialize(credentials=credentials, project=project_id or None)
            logger.info("GEE initialized with service account credentials.")
        else:
            ee.Initialize(project=project_id or None)
            logger.info("GEE initialized with local/application-default credentials.")

        GEE_STATE["initialized"] = True
        GEE_STATE["project_id"] = project_id
        GEE_STATE["error"] = None
        return True
    except Exception as exc:
        GEE_STATE["initialized"] = False
        GEE_STATE["project_id"] = project_id
        GEE_STATE["error"] = str(exc)
        logger.warning("GEE initialization unavailable: %s", exc)
        return False


initialize_gee()


def require_gee() -> None:
    if initialize_gee():
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "Google Earth Engine is not initialized. Configure GEE credentials with "
            "GEE_SERVICE_ACCOUNT_KEY and GEE_PROJECT_ID, or authenticate locally."
        ),
    )


def build_aoi(lat: float, lng: float, buffer_m: int = DEFAULT_BUFFER_METERS) -> ee.Geometry:
    return ee.Geometry.Point([lng, lat]).buffer(buffer_m)


def mask_clouds_s2(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


def compute_ndwi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


def compute_ndti(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B4", "B3"]).rename("NDTI")


def compute_fai(image: ee.Image) -> ee.Image:
    nir = image.select("B8")
    red = image.select("B4")
    swir1 = image.select("B11")
    baseline = red.add(swir1.subtract(red).multiply((832.8 - 664.6) / (1613.7 - 664.6)))
    return nir.subtract(baseline).rename("FAI")


def classify_pollution_rule_based(ndwi_val: float, ndti_val: float, fai_val: float) -> Dict:
    score = 0.0
    factors: List[str] = []

    if ndwi_val < 0.1:
        score += 40
        factors.append("Low water clarity from NDWI.")
    elif ndwi_val < 0.3:
        score += 20
        factors.append("Moderate water clarity from NDWI.")

    if ndti_val > 0.1:
        score += 35
        factors.append("High turbidity from NDTI.")
    elif ndti_val > 0.0:
        score += 15
        factors.append("Moderate turbidity from NDTI.")

    if fai_val > 0.02:
        score += 25
        factors.append("Algal bloom signal from FAI.")
    elif fai_val > 0.005:
        score += 10
        factors.append("Possible algal activity from FAI.")

    label = label_from_score(score)
    return {
        "label": label,
        "score": round(min(score, 100.0), 2),
        "color": COLOR_MAP[label],
        "factors": factors,
        "source": "rules",
    }


def blend_classification(rule_based: Dict, ml_insights: Dict) -> Dict:
    hybrid_score = max(rule_based["score"], round(0.6 * rule_based["score"] + 0.4 * ml_insights["ensemble_score"], 2))
    hybrid_label = label_from_score(hybrid_score)

    factors = list(rule_based["factors"])
    if ml_insights["signals"]:
        factors.extend(ml_insights["signals"][:2])

    if ml_insights["confidence"] >= 0.7:
        factors.append(
            f"Bundled ML classifier voted {ml_insights['ensemble_label'].lower()} with {int(ml_insights['confidence'] * 100)}% confidence."
        )

    return {
        "label": hybrid_label,
        "score": round(min(hybrid_score, 100.0), 2),
        "color": COLOR_MAP[hybrid_label],
        "factors": list(dict.fromkeys(factors)),
        "source": "hybrid-rule-ml",
        "rule_label": rule_based["label"],
        "ml_label": ml_insights["ensemble_label"],
    }


def get_date_range(days_back: int = 60) -> Tuple[str, str]:
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def build_image_collection(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection(SENTINEL2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_THRESHOLD))
        .map(mask_clouds_s2)
    )


def slope_trend(values: List[float], improving_threshold: float = 0.005) -> str:
    if len(values) < 3:
        return "stable"

    x_values = list(range(len(values)))
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(values) / len(values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
    denominator = sum((x - x_mean) ** 2 for x in x_values) or 1.0
    slope = numerator / denominator
    if slope > improving_threshold:
        return "improving"
    if slope < -improving_threshold:
        return "degrading"
    return "stable"


def analyze_location(lat: float, lng: float, buffer: int, days_back: int) -> Dict:
    require_gee()

    aoi = build_aoi(lat, lng, buffer)
    start_date, end_date = get_date_range(days_back)
    collection = build_image_collection(aoi, start_date, end_date)

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No cloud-free Sentinel-2 images found for this location in the last {days_back} days. "
                "Try increasing days_back or choosing a different location."
            ),
        )

    composite = collection.median().clip(aoi)
    ndwi_img = compute_ndwi(composite)
    ndti_img = compute_ndti(composite)
    fai_img = compute_fai(composite)

    stats = (
        ndwi_img.addBands(ndti_img).addBands(fai_img)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=20,
            maxPixels=1e9,
        )
        .getInfo()
    )

    ndwi_val = float(stats.get("NDWI", 0) or 0)
    ndti_val = float(stats.get("NDTI", 0) or 0)
    fai_val = float(stats.get("FAI", 0) or 0)

    rule_based = classify_pollution_rule_based(ndwi_val, ndti_val, fai_val)
    ml_insights = MODEL_BUNDLE.predict(ndwi_val, ndti_val, fai_val)
    classification = blend_classification(rule_based, ml_insights)

    rgb_map = composite.getMapId({"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3, "gamma": 1.4})
    ndwi_map = ndwi_img.getMapId(
        {
            "bands": ["NDWI"],
            "min": -0.5,
            "max": 0.8,
            "palette": ["#8B4513", "#F5DEB3", "#87CEEB", "#1E90FF", "#00008B"],
        }
    )
    pollution_map = ndti_img.getMapId(
        {
            "bands": ["NDTI"],
            "min": -0.2,
            "max": 0.3,
            "palette": ["#27ae60", "#f39c12", "#e74c3c"],
        }
    )

    bounds = aoi.bounds().getInfo()["coordinates"][0]
    bbox = {"west": bounds[0][0], "south": bounds[0][1], "east": bounds[2][0], "north": bounds[2][1]}

    return {
        "location": {"lat": lat, "lng": lng},
        "aoi_buffer_m": buffer,
        "date_range": {"start": start_date, "end": end_date},
        "images_used": image_count,
        "indices": {
            "ndwi": round(ndwi_val, 4),
            "ndti": round(ndti_val, 4),
            "fai": round(fai_val, 6),
        },
        "classification": classification,
        "rule_based": rule_based,
        "ml_insights": ml_insights,
        "tile_urls": {
            "rgb": rgb_map["tile_fetcher"].url_format,
            "ndwi": ndwi_map["tile_fetcher"].url_format,
            "pollution": pollution_map["tile_fetcher"].url_format,
        },
        "bbox": bbox,
        "gee": {"initialized": GEE_STATE["initialized"], "project_id": GEE_STATE["project_id"] or None},
    }


@app.get("/")
def root() -> Dict:
    return {"service": "AquaWatch API", "status": "running", "version": "1.1.0"}


@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "gee": {
            "initialized": GEE_STATE["initialized"],
            "project_id": GEE_STATE["project_id"] or None,
            "error": GEE_STATE["error"],
        },
        "ml_models": MODEL_BUNDLE.health(),
    }


@app.get("/models")
def models() -> Dict:
    return {
        "gee": {"initialized": GEE_STATE["initialized"], "project_id": GEE_STATE["project_id"] or None},
        "ml_models": MODEL_BUNDLE.health(),
        "features": MODEL_BUNDLE.feature_names,
    }


@app.get("/analyze")
def analyze(
    lat: float = Query(..., description="Latitude of the point of interest"),
    lng: float = Query(..., description="Longitude of the point of interest"),
    buffer: int = Query(DEFAULT_BUFFER_METERS, description="AOI buffer radius in metres"),
    days_back: int = Query(60, description="Days of imagery to look back"),
) -> Dict:
    try:
        return analyze_location(lat, lng, buffer, days_back)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /analyze: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/timeseries")
def timeseries(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    buffer: int = Query(DEFAULT_BUFFER_METERS, description="AOI buffer radius in metres"),
    months: int = Query(12, description="Number of months of history to fetch"),
) -> Dict:
    try:
        require_gee()
        aoi = build_aoi(lat, lng, buffer)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=months * 30)
        collection = build_image_collection(aoi, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        image_count = collection.size().getInfo()
        if image_count == 0:
            raise HTTPException(status_code=404, detail="No cloud-free images found for this location and time range.")

        results = []
        current = start_date.replace(day=1)

        while current <= end_date:
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_collection = collection.filterDate(current.strftime("%Y-%m-%d"), next_month.strftime("%Y-%m-%d"))
            month_count = month_collection.size().getInfo()

            if month_count > 0:
                monthly = month_collection.median().clip(aoi)
                ndwi_img = compute_ndwi(monthly)
                ndti_img = compute_ndti(monthly)
                fai_img = compute_fai(monthly)
                stats = (
                    ndwi_img.addBands(ndti_img).addBands(fai_img)
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=aoi,
                        scale=20,
                        maxPixels=1e9,
                    )
                    .getInfo()
                )

                ndwi_val = float(stats.get("NDWI", 0) or 0)
                ndti_val = float(stats.get("NDTI", 0) or 0)
                fai_val = float(stats.get("FAI", 0) or 0)
                rule_based = classify_pollution_rule_based(ndwi_val, ndti_val, fai_val)
                ml_insights = MODEL_BUNDLE.predict(ndwi_val, ndti_val, fai_val)
                classification = blend_classification(rule_based, ml_insights)

                results.append(
                    {
                        "month": current.strftime("%Y-%m"),
                        "ndwi": round(ndwi_val, 4),
                        "ndti": round(ndti_val, 4),
                        "fai": round(fai_val, 6),
                        "classification": classification["label"],
                        "score": classification["score"],
                        "ml_score": ml_insights["ensemble_score"],
                        "ml_label": ml_insights["ensemble_label"],
                        "images": month_count,
                    }
                )

            current = next_month

        if not results:
            raise HTTPException(status_code=404, detail="Could not compute time-series. No valid water pixels found.")

        ndwi_values = [row["ndwi"] for row in results]
        ml_scores = [row["ml_score"] for row in results]
        return {
            "location": {"lat": lat, "lng": lng},
            "months": months,
            "data_points": len(results),
            "trend": slope_trend(ndwi_values),
            "ml_trend": slope_trend(ml_scores, improving_threshold=1.5),
            "series": results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /timeseries: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/alerts")
def alerts(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    buffer: int = Query(DEFAULT_BUFFER_METERS, description="AOI buffer radius in metres"),
) -> Dict:
    try:
        result = analyze_location(lat, lng, buffer, days_back=60)
        classification = result["classification"]
        ml_insights = result["ml_insights"]

        if classification["label"] == "Polluted":
            recommendations = [
                "Avoid recreational water contact until field validation is completed.",
                "Do not use the water source for drinking or irrigation.",
                "Notify local environmental authorities and trigger confirmatory sampling.",
                "Review nearby discharge points and upstream runoff conditions.",
            ]
        elif classification["label"] == "Moderate":
            recommendations = [
                "Exercise caution near this water body.",
                "Increase monitoring frequency over the next 2 to 4 weeks.",
                "Schedule targeted sampling to confirm turbidity and algae conditions.",
            ]
        else:
            recommendations = [
                "Water quality appears stable from the current satellite pass.",
                "Continue routine monitoring and seasonal trend tracking.",
            ]

        if ml_insights["anomaly"]["flagged"]:
            recommendations.append("Investigate the site for unusual spectral patterns not seen in the clean baseline.")

        return {
            "location": {"lat": lat, "lng": lng},
            "alert_level": classification["label"],
            "alert_color": classification["color"],
            "pollution_score": classification["score"],
            "factors": classification["factors"],
            "indices": result["indices"],
            "ml_insights": ml_insights,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /alerts: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

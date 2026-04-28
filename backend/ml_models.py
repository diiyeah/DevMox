import json
import math
from pathlib import Path
from typing import Dict, List


LABEL_SCORE_MAP = {"Safe": 10.0, "Moderate": 45.0, "Polluted": 80.0}


def label_from_score(score: float) -> str:
    if score >= 50:
        return "Polluted"
    if score >= 20:
        return "Moderate"
    return "Safe"


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class PrebuiltWaterQualityModels:
    def __init__(self, artifact_path: Path | None = None) -> None:
        self.artifact_path = artifact_path or Path(__file__).with_name("prebuilt_models.json")
        self._artifact = self._load_artifact()

    def _load_artifact(self) -> Dict:
        with self.artifact_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def version(self) -> str:
        return self._artifact["version"]

    @property
    def feature_names(self) -> List[str]:
        return list(self._artifact["features"])

    def health(self) -> Dict:
        return {
            "status": "loaded",
            "artifact": self.artifact_path.name,
            "version": self.version,
            "models": ["multiclass_classifier", "pollution_regressor", "anomaly_detector"],
        }

    def predict(self, ndwi: float, ndti: float, fai: float) -> Dict:
        features = {"ndwi": ndwi, "ndti": ndti, "fai": fai}
        classifier = self._predict_classifier(features)
        regressor_score = self._predict_regressor(features)
        anomaly = self._predict_anomaly(features)

        classifier_expected_score = sum(
            classifier["probabilities"][label] * LABEL_SCORE_MAP[label]
            for label in classifier["labels"]
        )
        ensemble_score = clip(
            0.45 * regressor_score + 0.35 * classifier_expected_score + 0.20 * anomaly["score"],
            0.0,
            100.0,
        )
        ensemble_label = label_from_score(ensemble_score)
        confidence = max(classifier["probabilities"].values())

        signals = []
        if anomaly["score"] >= 65:
            signals.append("Anomalous spectral signature compared with the bundled clean-water baseline.")
        if classifier["top_label"] != ensemble_label:
            signals.append(
                f"Classifier leaned {classifier['top_label'].lower()}, but the ensemble settled on {ensemble_label.lower()}."
            )
        if regressor_score >= 55:
            signals.append("Regression model estimated a materially elevated pollution burden.")
        elif regressor_score <= 20:
            signals.append("Regression model placed the water body inside the low-risk range.")

        return {
            "model_version": self.version,
            "features": {name: round(features[name], 6) for name in self.feature_names},
            "classifier": classifier,
            "regression_score": round(regressor_score, 2),
            "anomaly": anomaly,
            "ensemble_score": round(ensemble_score, 2),
            "ensemble_label": ensemble_label,
            "confidence": round(confidence, 4),
            "signals": signals,
        }

    def _predict_classifier(self, features: Dict[str, float]) -> Dict:
        classifier = self._artifact["classifier"]
        feature_vector = [features[name] for name in self.feature_names]
        logits = {}
        for label in classifier["labels"]:
            logits[label] = (
                sum(weight * value for weight, value in zip(classifier["weights"][label], feature_vector))
                + classifier["bias"][label]
            )

        max_logit = max(logits.values())
        exp_logits = {label: math.exp(value - max_logit) for label, value in logits.items()}
        total = sum(exp_logits.values()) or 1.0
        probabilities = {label: exp_logits[label] / total for label in classifier["labels"]}
        top_label = max(probabilities, key=probabilities.get)

        return {
            "labels": classifier["labels"],
            "logits": {label: round(value, 4) for label, value in logits.items()},
            "probabilities": {label: round(probabilities[label], 4) for label in classifier["labels"]},
            "top_label": top_label,
        }

    def _predict_regressor(self, features: Dict[str, float]) -> float:
        regressor = self._artifact["regressor"]
        score = regressor["bias"]
        for name, weight in regressor["weights"].items():
            score += weight * features[name]
        return clip(score, 0.0, 100.0)

    def _predict_anomaly(self, features: Dict[str, float]) -> Dict:
        detector = self._artifact["anomaly_detector"]
        squared_distance = 0.0
        for name in self.feature_names:
            centered = features[name] - detector["center"][name]
            scale = detector["scale"][name] or 1.0
            squared_distance += (centered / scale) ** 2

        normalized_distance = math.sqrt(squared_distance / len(self.feature_names))
        anomaly_probability = 1.0 / (1.0 + math.exp(-(normalized_distance - detector["threshold"]) * 1.8))
        anomaly_score = clip(anomaly_probability * 100.0, 0.0, 100.0)

        return {
            "normalized_distance": round(normalized_distance, 4),
            "threshold": detector["threshold"],
            "score": round(anomaly_score, 2),
            "flagged": anomaly_score >= 60.0,
        }

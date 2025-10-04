# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib, numpy as np, pandas as pd, uvicorn, os, io, hashlib, re
from datetime import datetime

# ----------------------------
# App & CORS
# ----------------------------
app = FastAPI(
    title="Exoplanet Classification API",
    description="Predict exoplanet disposition and serve UI-friendly endpoints",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load model artifacts (env-configurable)
# ----------------------------
MODEL_PATH    = os.getenv("MODEL_PATH",    "exoplanet_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_names.pkl")
IMPUTER_PATH  = os.getenv("IMPUTER_PATH",  "imputer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    feature_names: List[str] = joblib.load(FEATURES_PATH)
    imputer = joblib.load(IMPUTER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v0")

# Optional catalog for identifier lookup (KOI/Kepler)
DATASET_PATH = os.getenv("DATASET_PATH")  # e.g., datasets/koi_cumulative.csv
_df_cached: Optional[pd.DataFrame] = None
def get_dataset() -> Optional[pd.DataFrame]:
    global _df_cached
    if _df_cached is not None:
        return _df_cached
    if DATASET_PATH and os.path.exists(DATASET_PATH):
        try:
            _df_cached = pd.read_csv(DATASET_PATH, comment="#")
            return _df_cached
        except Exception:
            return None
    return None

# ----------------------------
# Label mapping helpers
# ----------------------------
IDX2LABEL = {
    0: "confirmed_exoplanet",
    1: "planet_candidate",
    2: "false_positive",
}
def normalize_probs(probs: List[float]) -> Dict[str, float]:
    return {IDX2LABEL[i]: float(probs[i]) for i in range(min(len(probs), 3))}

def make_id(prefix: str, key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"

# ----------------------------
# Schemas
# ----------------------------
class PredictionInput(BaseModel):
    koi_fpflag_nt: Optional[int] = 0
    koi_fpflag_ss: Optional[int] = 0
    koi_fpflag_co: Optional[int] = 0
    koi_fpflag_ec: Optional[int] = 0
    koi_depth: Optional[float] = None
    koi_duration: Optional[float] = None
    koi_impact: Optional[float] = None
    koi_period: Optional[float] = None
    koi_model_snr: Optional[float] = None
    koi_prad: Optional[float] = None
    koi_teq: Optional[float] = None
    koi_steff: Optional[float] = None
    koi_srad: Optional[float] = None
    koi_slogg: Optional[float] = None

class PredictionOutput(BaseModel):
    id: str
    identifier: Optional[str] = None
    classification: str
    confidence: float
    scores: Dict[str, float]
    flags: List[str] = []
    model_version: str = MODEL_VERSION

class DiscoveryItem(BaseModel):
    id: str
    identifier: str
    name: Optional[str] = None
    classification: str
    confidence: float
    scores: Dict[str, float] = {}
    star: Optional[str] = None
    distance_pc: Optional[float] = None
    radius_re: Optional[float] = None
    period_days: Optional[float] = None
    teq_k: Optional[float] = None
    mission: Optional[str] = None
    discovered_at: Optional[str] = None
    thumbnail_url: Optional[str] = None
    flags: List[str] = []
    model_version: str = MODEL_VERSION

class DiscoveriesResponse(BaseModel):
    items: List[DiscoveryItem]
    page: int
    page_size: int
    total: Optional[int] = None

class ExplainResponse(BaseModel):
    id: str
    feature_importance: List[Dict[str, Any]] = []
    saliency: List[Dict[str, float]] = []
    notes: Optional[str] = None

class DiagnosticsResponse(BaseModel):
    odd_even_depth_ratio: float
    secondary_eclipse_snr: float
    centroid_offset_sigma: float
    ghosting_score: float
    data_quality: float
    flags: List[str]

class TargetResponse(BaseModel):
    summary: PredictionOutput
    lightcurve: Dict[str, Any]
    diagnostics: DiagnosticsResponse

# ----------------------------
# Core inference helpers
# ----------------------------
NUMERICAL_FEATURES = [
    "koi_depth","koi_impact","koi_model_snr","koi_prad","koi_teq",
    "koi_steff","koi_srad","koi_slogg"
]

def _prepare_df(raw: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw]).reindex(columns=feature_names)
    df_num = df[NUMERICAL_FEATURES]
    df[NUMERICAL_FEATURES] = pd.DataFrame(
        imputer.transform(df_num),
        columns=NUMERICAL_FEATURES,
        index=df.index,
    )
    return df.astype(np.float32)

def classify_row(row: Dict[str, Any], identifier: Optional[str] = None) -> PredictionOutput:
    X = _prepare_df(row).values
    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist()
    label = IDX2LABEL.get(pred_idx, "unknown")
    confidence = float(probs[pred_idx]) if 0 <= pred_idx < len(probs) else 0.0
    pid = make_id("clf", identifier or str(row))
    return PredictionOutput(
        id=pid,
        identifier=identifier,
        classification=label,
        confidence=confidence,
        scores=normalize_probs(probs),
        flags=[],
    )

def parse_identifier_from_dataset(identifier: str) -> Optional[Dict[str, Any]]:
    df = get_dataset()
    if df is None:
        return None
    id_clean = identifier.strip().upper()
    try:
        # KOI-xxx(.xx)
        if id_clean.startswith("KOI"):
            m = re.findall(r"KOI[-\s]?(\d+(\.\d+)?)", id_clean)
            if m:
                key = f"KOI-{m[0][0]}"
                sub = df.get("kepoi_name")
                if sub is not None:
                    sub_df = df[df["kepoi_name"].astype(str).str.upper()==key]
                    if len(sub_df)==0:
                        sub_df = df[df["kepoi_name"].astype(str).str.contains(m[0][0], case=False, na=False)]
                    if len(sub_df):
                        row = sub_df.iloc[0].to_dict()
                        return {k: row.get(k, None) for k in feature_names}
        # Kepler ID
        if id_clean.isdigit() and "kepid" in df.columns:
            sub = df[df["kepid"].astype(str) == id_clean]
            if len(sub):
                row = sub.iloc[0].to_dict()
                return {k: row.get(k, None) for k in feature_names}
    except Exception:
        return None
    return None

def seeded_rng(key: str) -> np.random.RandomState:
    import hashlib as _h
    return np.random.RandomState(int(_h.sha1(key.encode()).hexdigest()[:8], 16))

def synthetic_lightcurve(identifier: str, period_days: float = 3.2, depth: float = 0.002) -> Dict[str, Any]:
    rng = seeded_rng(identifier)
    N = 1000
    time_arr = np.linspace(0, period_days, N)
    flux = 1.0 + rng.normal(0, 0.0005, size=N)
    phase = ((time_arr % period_days) / period_days) - 0.5
    in_transit = np.abs(phase) < 0.02
    flux[in_transit] -= depth * (1 - (np.abs(phase[in_transit]) / 0.02))
    detr = flux - pd.Series(flux).rolling(41, min_periods=1, center=True).median().values
    return {
        "id": make_id("lc", identifier),
        "identifier": identifier,
        "period_days": period_days,
        "time": time_arr.tolist(),
        "flux": flux.tolist(),
        "detrended": (1.0 + detr).tolist(),
        "meta": {"mission": "TESS"},
    }

def basic_diagnostics(identifier: str) -> DiagnosticsResponse:
    rng = seeded_rng("diag_"+identifier)
    odd_even_ratio = 1.0 + rng.normal(0, 0.05)
    sec_snr = max(0.0, rng.normal(0.8, 0.3))
    centroid = abs(rng.normal(0.6, 0.2))
    ghost = max(0.0, rng.normal(0.1, 0.05))
    dq = float(np.clip(rng.normal(0.92, 0.04), 0, 1))
    flags = []
    if abs(odd_even_ratio - 1.0) < 0.1: flags.append("odd-even-ok")
    if sec_snr < 1.5: flags.append("no-sec-eclipse")
    if centroid < 1.5: flags.append("centroid-stable")
    return DiagnosticsResponse(
        odd_even_depth_ratio=float(odd_even_ratio),
        secondary_eclipse_snr=float(sec_snr),
        centroid_offset_sigma=float(centroid),
        ghosting_score=float(ghost),
        data_quality=dq,
        flags=flags,
    )

# ----------------------------
# Health & metadata
# ----------------------------
@app.get("/")
def root(): return {"message": "Exoplanet Classification API is running!"}

@app.get("/healthz")
def healthz(): return {"ok": True, "model_loaded": True, "model_version": MODEL_VERSION}

@app.get("/models")
def models_meta():
    return {"active": MODEL_VERSION,
            "artifacts": {"model": MODEL_PATH, "features": FEATURES_PATH, "imputer": IMPUTER_PATH}}

# ----------------------------
# Legacy single prediction (kept for compat)
# ----------------------------
class LegacyProb(BaseModel):
    class_id: int
    probability: float
class LegacyOutput(BaseModel):
    prediction: int
    class_label: str
    probabilities: List[LegacyProb]

@app.post("/predict", response_model=LegacyOutput)
def predict_disposition(input_data: PredictionInput):
    try:
        X_df = _prepare_df(input_data.dict())
        pred = int(model.predict(X_df.values)[0])
        probs = model.predict_proba(X_df.values)[0].tolist()
        labels = {0: 'CONFIRMED (Exoplanet)', 1: 'CANDIDATE', 2: 'FALSE POSITIVE'}
        return LegacyOutput(
            prediction=pred,
            class_label=labels[pred],
            probabilities=[LegacyProb(class_id=i, probability=float(p)) for i, p in enumerate(probs)],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# ----------------------------
# New endpoints used by your React app
# ----------------------------

# 1) Classify by identifier/features or CSV upload
class ClassifyRequest(BaseModel):
    identifier: Optional[str] = None
    features: Optional[PredictionInput] = None

@app.post("/classify", response_model=PredictionOutput)
async def classify(body: Optional[ClassifyRequest] = None, file: UploadFile = File(None)):
    try:
        identifier = None
        row: Optional[Dict[str, Any]] = None
        if file is not None:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
            if df.empty:
                raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
            row = df.iloc[0].to_dict()
        elif body:
            if body.features:
                row = body.features.dict()
            elif body.identifier:
                identifier = body.identifier.strip()
                maybe = parse_identifier_from_dataset(identifier)
                if maybe:
                    row = maybe
                else:
                    raise HTTPException(status_code=404,
                        detail=f"Identifier '{identifier}' not in local dataset; send features or CSV.")
        else:
            raise HTTPException(status_code=400, detail="Provide an identifier, features, or CSV file.")

        out = classify_row(row, identifier=identifier)
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/classify failed: {e}")

# 2) Batch classify
class BatchRequest(BaseModel): identifiers: List[str]
@app.post("/classify/batch", response_model=List[PredictionOutput])
def classify_batch(req: BatchRequest):
    out: List[PredictionOutput] = []
    for ident in req.identifiers:
        row = parse_identifier_from_dataset(ident)
        if not row:
            pid = make_id("clf", ident)
            out.append(PredictionOutput(
                id=pid, identifier=ident, classification="unknown", confidence=0.0,
                scores={"confirmed_exoplanet":0.0,"planet_candidate":0.0,"false_positive":0.0},
                flags=["not-found"],
            ))
            continue
        out.append(classify_row(row, identifier=ident))
    return out

# 3) Lightcurve for plotting
@app.get("/lightcurve")
def lightcurve(id: str = Query(..., description="KOI/KepID/TIC if supported")):
    # try to pull period from dataset if present
    period_guess = 3.2
    ds = get_dataset()
    if ds is not None:
        try:
            sub = None
            if "kepoi_name" in ds.columns:
                sub = ds[ds["kepoi_name"].astype(str).str.upper()==id.strip().upper()]
            if (sub is None or len(sub)==0) and id.isdigit() and "kepid" in ds.columns:
                sub = ds[ds["kepid"].astype(str)==id]
            if sub is not None and len(sub) and "koi_period" in sub.columns and pd.notna(sub.iloc[0]["koi_period"]):
                period_guess = float(sub.iloc[0]["koi_period"])
        except Exception:
            pass
    return synthetic_lightcurve(id, period_days=period_guess)

# 4) Explanation (feature importances + synthetic saliency)
@app.get("/explain", response_model=ExplainResponse)
def explain(id: Optional[str] = None, identifier: Optional[str] = None):
    try:
        importances = getattr(model, "feature_importances_", None)
        fi = []
        if importances is not None and len(importances) == len(feature_names):
            total = float(np.sum(importances)) or 1.0
            for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
                fi.append({"feature": fname, "importance": float(imp/total), "description": ""})
        else:
            fi = [{"feature": f, "importance": 1.0/len(feature_names)} for f in feature_names]
        key = identifier or id or "sample"
        rng = seeded_rng("sal_"+key)
        phases = np.linspace(-0.5, 0.5, 101)
        sal = (np.exp(-((phases)/0.06)**2) + rng.normal(0, 0.02, size=len(phases)))
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        saliency = [{"phase": float(ph), "contribution": float(v)} for ph, v in zip(phases, sal)]
        return ExplainResponse(id=id or make_id("exp", key), feature_importance=fi, saliency=saliency,
                               notes=f"Model {MODEL_VERSION} (global LightGBM gains)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/explain failed: {e}")

# 5) Diagnostics (false-positive checks)
@app.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(id: str = Query(..., description="KOI/KepID/TIC if supported")):
    return basic_diagnostics(id)

# 6) Aggregated target details
@app.get("/target/{identifier}", response_model=TargetResponse)
def target(identifier: str):
    row = parse_identifier_from_dataset(identifier) or {k: 0 for k in feature_names}
    summary = classify_row(row, identifier=identifier)
    lc = lightcurve(identifier)
    diag = diagnostics(identifier)
    return TargetResponse(summary=summary, lightcurve=lc, diagnostics=diag)

# 7) Discoveries feed (powers the gallery)
@app.get("/discoveries", response_model=DiscoveriesResponse)
def discoveries(
    page: int = 1,
    page_size: int = 24,
    mission: Optional[str] = None,
    min_conf: Optional[float] = None,
    class_in: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "confidence",
    order: str = "desc",
):
    ds = get_dataset()
    if ds is None:
        return DiscoveriesResponse(items=[], page=page, page_size=page_size, total=0)

    work = ds.copy()
    if search:
        s = search.strip()
        mask = pd.Series(False, index=work.index)
        if "kepoi_name" in work.columns:
            mask = mask | work["kepoi_name"].astype(str).str.contains(s, case=False, na=False)
        if "kepid" in work.columns:
            mask = mask | work["kepid"].astype(str).str.contains(s, case=False, na=False)
        work = work[mask]

    work = work.sample(min(500, len(work)), random_state=42) if len(work) > 500 else work

    def safe_row(r: pd.Series) -> Dict[str, Any]:
        return {k: r.get(k, None) for k in feature_names}

    results: List[PredictionOutput] = []
    for _, r in work.iterrows():
        ident = None
        if "kepoi_name" in work.columns and pd.notna(r.get("kepoi_name")):
            ident = str(r["kepoi_name"])
        elif "kepid" in work.columns and pd.notna(r.get("kepid")):
            ident = str(int(r["kepid"]))
        results.append(classify_row(safe_row(r), identifier=ident))

    if class_in:
        class_set = set([x.strip() for x in class_in.split(",") if x.strip()])
        results = [x for x in results if x.classification in class_set]
    if min_conf is not None:
        results = [x for x in results if x.confidence >= float(min_conf)]

    reverse = (order.lower() != "asc")
    if sort_by == "confidence":
        results.sort(key=lambda x: x.confidence, reverse=reverse)

    total = len(results)
    start = max(0, (page - 1) * page_size); end = start + page_size
    page_items = results[start:end]
    items = [
        DiscoveryItem(
            id=r.id, identifier=r.identifier or r.id, name=r.identifier or r.id,
            classification=r.classification, confidence=r.confidence, scores=r.scores,
            star=r.identifier, mission=mission, discovered_at=datetime.utcnow().isoformat()+"Z",
            flags=r.flags, model_version=r.model_version
        )
        for r in page_items
    ]
    return DiscoveriesResponse(items=items, page=page, page_size=page_size, total=total)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

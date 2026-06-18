import io, time, base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from .predictor import FireDetector
from .db import init_db, log_request, get_stats, get_history

app      = FastAPI(title="Fire Detection API", version="1.0")
detector = FireDetector()

@app.on_event("startup")
async def startup():
    await init_db()

@app.get("/health")
async def health():
    return {"status": "ok", "model": detector.model_name, "device": detector.device}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Ожидается изображение")
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    t0 = time.perf_counter()
    detections, annotated = detector.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    alarm = any(d["class"] == "fire" for d in detections)
    await log_request(file.filename, detector.model_name, detections, latency_ms, alarm)

    return {
        "detections":          detections,
        "latency_ms":          round(latency_ms, 2),
        "alarm":               alarm,
        "annotated_image_b64": img_b64,
    }

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        img = Image.open(io.BytesIO(await f.read())).convert("RGB")
        t0  = time.perf_counter()
        dets, _ = detector.predict(img)
        lat  = (time.perf_counter() - t0) * 1000
        alarm = any(d["class"] == "fire" for d in dets)
        await log_request(f.filename, detector.model_name, dets, lat, alarm)
        out.append({"filename": f.filename, "detections": dets,
                    "latency_ms": round(lat, 2), "alarm": alarm})
    return out

@app.get("/metrics")
async def metrics():
    return await get_stats()

@app.get("/alert_history")
async def alert_history(limit: int = 50):
    return await get_history(limit)
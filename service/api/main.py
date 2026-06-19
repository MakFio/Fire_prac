import io, time, base64, os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from .predictor import FireDetector
from .db import init_db, log_request, get_stats, get_history

app      = FastAPI(title="Fire Detection API", version="1.1")
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


@app.post("/video_predict")
async def video_predict(file: UploadFile = File(...)):
    if file.content_type not in ("video/mp4", "video/mpeg", "application/octet-stream"):
        raise HTTPException(400, "Ожидается видеофайл MP4")

    video_bytes = await file.read()

    # Записываем входное видео во временный файл
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".mp4", "_out.mp4")

    try:
        import cv2, numpy as np

        cap      = cv2.VideoCapture(tmp_in_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

        all_detections = []
        frame_count    = 0
        t0             = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # BGR -> RGB -> PIL
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets, annotated_pil = detector.predict(pil_img)
            all_detections.extend(dets)
            # PIL -> BGR -> writer
            annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

        cap.release()
        writer.release()

        latency_ms  = (time.perf_counter() - t0) * 1000
        alarm       = any(d["class"] == "fire" for d in all_detections)
        unique_dets = _summarize_detections(all_detections)

        await log_request(
            file.filename, detector.model_name,
            unique_dets, latency_ms, alarm
        )

        with open(tmp_out_path, "rb") as vf:
            video_b64 = base64.b64encode(vf.read()).decode()

    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    return {
        "frame_count":      frame_count,
        "detections_total": len(all_detections),
        "detections":       unique_dets,
        "latency_ms":       round(latency_ms, 2),
        "fps_processed":    round(frame_count / (latency_ms / 1000), 1) if latency_ms > 0 else 0,
        "alarm":            alarm,
        "annotated_video_b64": video_b64,
    }


def _summarize_detections(dets: list) -> list:
    summary: dict[str, dict] = {}
    for d in dets:
        cls = d["class"]
        if cls not in summary:
            summary[cls] = {"class": cls, "count": 0, "max_confidence": 0.0}
        summary[cls]["count"] += 1
        summary[cls]["max_confidence"] = max(summary[cls]["max_confidence"], d["confidence"])
    return list(summary.values())


@app.get("/metrics")
async def metrics():
    return await get_stats()

@app.get("/alert_history")
async def alert_history(limit: int = 50):
    return await get_history(limit)

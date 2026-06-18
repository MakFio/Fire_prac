import io, requests
from PIL import Image

API = "http://localhost:8000"

def test_health():
    r = requests.get(f"{API}/health", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    print(f"[OK] /health  model={d['model']}  device={d['device']}")

def test_predict_blank():
    img = Image.new("RGB", (640, 640), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = requests.post(f"{API}/predict",
                      files={"file": ("blank.jpg", buf, "image/jpeg")},
                      timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert "detections" in d and "latency_ms" in d
    print(f"[OK] /predict  dets={len(d['detections'])}  latency={d['latency_ms']} мс")

def test_metrics():
    r = requests.get(f"{API}/metrics", timeout=5)
    assert r.status_code == 200
    assert "total_requests" in r.json()
    print(f"[OK] /metrics  requests={r.json()['total_requests']}")

if __name__ == "__main__":
    test_health()
    test_predict_blank()
    test_metrics()
    print("\nВсе smoke-тесты пройдены")
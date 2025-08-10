# server/main.py
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np

# Quantum Blur (Qiskit-based)
from quantumblur import quantumblur as qb

# ---- Config ----
W_DEFAULT = 64
H_DEFAULT = 64
SHOTS_DEFAULT = 128
ROTATION_DEFAULT = 0.35

app = FastAPI()

# Lock this down to your domains in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qata.live",
        "https://www.qata.live",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True, "endpoints": ["/blur (PNG)", "/blur_raw (raw bytes)"]}

@app.on_event("startup")
def warmup():
    """Warm the quantum path to avoid first-hit latency spikes."""
    x = np.zeros((H_DEFAULT, W_DEFAULT), dtype=np.float32)
    _ = qb.blur(x, rotation=0.3, shots=64)

# ---------- Helpers ----------

def to_png_bytes(arr: np.ndarray) -> bytes:
    """Normalize float array to 8-bit and return PNG bytes."""
    arr = np.asarray(arr, dtype=np.float32)
    # Normalize safely even if arr.ptp() == 0
    rng = float(arr.max() - arr.min())
    if rng < 1e-12:
        arr8 = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr8 = np.clip(255.0 * (arr - arr.min()) / (rng + 1e-12), 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr8, mode="L").save(buf, "PNG", optimize=True)
    return buf.getvalue()

# ---------- PNG endpoint (simple; flexible) ----------

@app.post("/blur")
async def blur_png(
    req: Request,
    shots: int = SHOTS_DEFAULT,
    rotation: float = ROTATION_DEFAULT,
):
    """
    Send a small PNG/JPEG frame (e.g., 64x64). Returns a grayscale PNG
    with Quantum Blur applied.
    """
    data = await req.body()
    if not data:
        return Response(status_code=400, content=b"empty body")

    try:
        img = Image.open(BytesIO(data)).convert("L")
    except Exception:
        return Response(status_code=400, content=b"bad image")

    x = np.asarray(img, dtype=np.float32) / 255.0
    y = qb.blur(x, rotation=rotation, shots=shots)
    return Response(content=to_png_bytes(y), media_type="image/png")

# ---------- RAW endpoint (ultra-low-latency) ----------

@app.post("/blur_raw")
async def blur_raw(
    req: Request,
    w: int = W_DEFAULT,
    h: int = H_DEFAULT,
    shots: int = SHOTS_DEFAULT,
    rotation: float = ROTATION_DEFAULT,
):
    """
    Send raw 8-bit grayscale bytes of size (w*h). Returns raw 8-bit
    grayscale bytes (same size) after Quantum Blur. Media type:
    application/octet-stream
    """
    data = await req.body()
    if not data:
        return Response(status_code=400, content=b"empty body")

    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != w * h:
        return Response(status_code=400, content=b"bad size")

    x = (arr.reshape((h, w)).astype(np.float32)) / 255.0
    y = qb.blur(x, rotation=rotation, shots=shots)
    out = np.clip(y * 255.0, 0, 255).astype(np.uint8).tobytes()
    return Response(content=out, media_type="application/octet-stream")

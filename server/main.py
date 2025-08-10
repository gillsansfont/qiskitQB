from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import time

# Quantum Blur (Python API)
# https://github.com/qiskit-community/QuantumBlur
from quantumblur import quantumblur as qb

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-build a small input heightmap (fast!)
W, H = 64, 64  # keep small for low latency
base = (np.random.rand(H, W) * 255).astype(np.uint8)

# Simple cache with timestamp to avoid recomputing too fast
_last = {"ts": 0.0, "png": None}

@app.get("/quantum-mask.png")
def quantum_mask():
    now = time.time()
    # Update at most ~10 Hz
    if _last["png"] is None or (now - _last["ts"]) > 0.1:
        # Run Quantum Blur on a tiny heightmap
        # Rotation parameter picks a circuit; feel free to expose as a query param
        blurred = qb.blur(base, rotation=0.35, shots=256)  # small shots = fast
        # Normalize result → 8-bit
        arr = np.clip(255 * (blurred - blurred.min()) / (blurred.ptp() + 1e-6), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")  # grayscale
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        _last["png"] = buf.getvalue()
        _last["ts"] = now

    return Response(content=_last["png"], media_type="image/png")

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np, time

# Quantum Blur
from quantumblur import quantumblur as qb

app = FastAPI()

# tighten CORS in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qata.live", "https://www.qata.live"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

W, H = 64, 64
base = (np.random.rand(H, W) * 255).astype(np.uint8)
cache = {"ts": 0.0, "png": None}

@app.get("/quantum-mask.png")
def quantum_mask():
    now = time.time()
    if cache["png"] is None or (now - cache["ts"]) > 0.1:
        blurred = qb.blur(base, rotation=0.35, shots=256)
        arr = np.clip(255 * (blurred - blurred.min()) / (blurred.ptp() + 1e-6), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "L")
        buf = BytesIO(); img.save(buf, "PNG", optimize=True)
        cache.update(ts=now, png=buf.getvalue())
    return Response(content=cache["png"], media_type="image/png")

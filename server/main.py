from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
from quantumblur import quantumblur as qb

app = FastAPI()

# Lock this down to your domains once live:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qata.live", "https://www.qata.live", "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_png_bytes(arr: np.ndarray) -> bytes:
    arr8 = np.clip(255*(arr - arr.min())/(arr.ptp()+1e-6), 0, 255).astype(np.uint8)
    out = BytesIO(); Image.fromarray(arr8, mode="L").save(out, "PNG", optimize=True)
    return out.getvalue()

@app.post("/blur")
async def blur(req: Request):
    """
    Receives a tiny PNG/JPEG frame (e.g., 64x64) from your Hydra canvas,
    runs Quantum Blur, returns a grayscale PNG you use as a modulator.
    """
    data = await req.body()
    img = Image.open(BytesIO(data)).convert("L")
    x = np.asarray(img, dtype=np.float32) / 255.0
    y = qb.blur(x, rotation=0.35, shots=256)
    return Response(content=to_png_bytes(y), media_type="image/png")

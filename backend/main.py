from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import math
import time

# We import the inference logic
# It will load the OpenCLIP model upon import.
import langsplat_inference

app = FastAPI(title="LangSplat Platform API")

# Setup CORS to allow the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text_query: str
    threshold: float
    frame_id: str

class QueryResponse(BaseModel):
    image_base64: str
    message: str

@app.post("/api/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    img_b64, msg = langsplat_inference.process_query(
        text_query=req.text_query,
        threshold=req.threshold,
        frame_id=req.frame_id
    )
    if img_b64 is None:
        raise HTTPException(status_code=400, detail=msg)
    
    return QueryResponse(image_base64=img_b64, message=msg)

class QueryGifRequest(BaseModel):
    text_query: str
    threshold: float
    category: str

@app.post("/api/query_gif", response_model=QueryResponse)
def run_query_gif(req: QueryGifRequest):
    import io
    from PIL import Image
    import base64
    
    if req.category == "lerf":
        frames_range = range(0, 30)
    elif req.category == "etri":
        frames_range = range(32, 42)
    else:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    frames = []
    for i in frames_range:
        frame_id = f"{i:05d}"
        try:
            img_b64, msg = langsplat_inference.process_query(
                text_query=req.text_query,
                threshold=req.threshold,
                frame_id=frame_id
            )
            if img_b64:
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                frames.append(img)
        except Exception as e:
            print(f"Skipping frame {frame_id} due to error: {e}")
            continue
    
    if not frames:
        raise HTTPException(status_code=400, detail="Failed to generate frames")
        
    out_io = io.BytesIO()
    frames[0].save(
        out_io,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    gif_b64 = base64.b64encode(out_io.getvalue()).decode('utf-8')
    return QueryResponse(image_base64=gif_b64, message="Generated GIF sequence")

@app.get("/api/training/metrics")
def get_training_metrics():
    # Mocking training metrics for the dashboard
    # Since we don't have a real training loop, we generate an interesting curve
    steps = 100
    metrics = []
    base_loss = 2.0
    for i in range(steps):
        # decaying loss with a floor so it never hits exactly 0
        loss = base_loss * math.exp(-i / 20.0) + (0.05 * math.sin(i)) + 0.015
        # increasing PSNR
        psnr = 15.0 + 15.0 * (1.0 - math.exp(-i / 30.0)) + (0.5 * math.cos(i))
        
        # mock GPU usage (%) around 85-95
        gpu_usage = 90.0 + 5.0 * math.sin(i * 0.5)
        
        # mock gaussian count (points) growing and saturating
        gaussian_count = int(50000 + 50000 * (1.0 - math.exp(-i / 40.0)))
        
        # mock iterations/sec
        iterations_per_sec = 12.0 + 2.0 * math.cos(i * 0.3)
        
        metrics.append({
            "step": i * 100,
            "loss": max(0.001, loss),
            "psnr": psnr,
            "gpu_usage": gpu_usage,
            "gaussian_count": gaussian_count,
            "iterations_per_sec": iterations_per_sec
        })
    return {"status": "training", "progress": 100, "metrics": metrics}

@app.get("/api/models")
def get_models():
    # Mocking a list of available models/checkpoints
    return {
        "models": [
            {"id": "langsplat-base-001", "name": "LangSplat Autoencoder", "status": "Ready"},
            {"id": "langsplat-ft-002", "name": "LangSplat Fine-Tuned", "status": "Training (80%)"}
        ]
    }

@app.get("/api/failures")
def get_failures():
    # Mocking failure cases gallery data
    return {
        "failures": [
            {
                "id": "f-105",
                "query": "table",
                "score": 0.50,
                "issue": "Failed to isolate the table. Relevancy threshold (0.5) did not mask out the plush toys and other objects on top of it.",
                "timestamp": "2026-07-10T12:45:00Z",
                "image_url": "/failure_table.png"
            },
            {
                "id": "f-101",
                "query": "A flying red elephant",
                "score": 0.12,
                "issue": "Model hallucinations. Relevancy score extremely low. Points clustered randomly.",
                "timestamp": "2026-07-10T10:15:00Z"
            },
            {
                "id": "f-102",
                "query": "Transparent glass cup on a white table",
                "score": 0.25,
                "issue": "Failed to render transparency. Glass appears opaque and grayish.",
                "timestamp": "2026-07-10T09:42:00Z"
            },
            {
                "id": "f-103",
                "query": "Intricate spider web",
                "score": 0.31,
                "issue": "Thin structures are completely lost in the point cloud resolution.",
                "timestamp": "2026-07-09T18:20:00Z"
            },
            {
                "id": "f-104",
                "query": "Mirror reflecting the room",
                "score": 0.19,
                "issue": "Specular reflections not handled correctly; mirror surface is a solid color.",
                "timestamp": "2026-07-09T15:05:00Z"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

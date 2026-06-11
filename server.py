"""
Ultra Fast Image Gen - web server.

FastAPI backend + static frontend (web/). Replaces the old Gradio app:

    python server.py            # http://127.0.0.1:7860

The heavy lifting (pipeline cache, job queue, storage management) lives in
engine.py; this file is just the HTTP surface.
"""

import os
import subprocess

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = FastAPI(title="Ultra Fast Image Gen", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=4, ge=1, le=50)
    guidance: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)
    device: str | None = None
    input_images: list[str] | None = None  # base64 data URLs (img2img models)
    lora_path: str | None = None
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    anima_preset: str | None = None
    auto_save: bool = False
    output_dir: str | None = None


@app.get("/api/status")
def status():
    return {
        "devices": engine.get_devices(),
        "current_model": engine.current_model(),
        "default_output_dir": engine.DEFAULT_OUTPUT_DIR,
    }


@app.get("/api/models")
def models():
    return engine.list_models()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model not in engine.MODELS:
        raise HTTPException(404, f"unknown model: {req.model}")
    job_id = engine.submit_job(req.model_dump())
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = engine.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    elapsed = None
    if job["started"]:
        elapsed = (job["finished"] or __import__("time").time()) - job["started"]
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "images": [{"url": i["url"], "seed": i["seed"], "saved": i.get("saved")} for i in job["images"]],
        "error": job["error"],
        "elapsed": elapsed,
        "queue_position": engine.queue_position(job_id),
    }


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    if engine.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    return {"cancelled": engine.cancel_job(job_id)}


@app.get("/api/files/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if "/" in name or ".." in name or ".." in job_id or "/" in job_id:
        raise HTTPException(400, "bad path")
    path = os.path.join(engine.JOB_ROOT, job_id, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/storage")
def storage():
    return engine.scan_storage()


class DeleteRequest(BaseModel):
    key: str


@app.post("/api/storage/delete")
def storage_delete(req: DeleteRequest):
    ok, message = engine.delete_storage(req.key)
    if not ok:
        raise HTTPException(400, message)
    return {"message": message}


class FolderRequest(BaseModel):
    dir: str | None = None


@app.post("/api/open_folder")
def open_folder(req: FolderRequest):
    folder = os.path.expanduser(req.dir or engine.DEFAULT_OUTPUT_DIR)
    os.makedirs(folder, exist_ok=True)
    subprocess.run(["open", folder], check=False)
    return {"opened": folder}


# Static frontend last so /api keeps precedence.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    print("Ultra Fast Image Gen — http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")

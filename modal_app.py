"""
FastAPI + Wan 2.1 T2V model downloader on Modal
"""

import os
import modal
import subprocess
import shutil
from fastapi import FastAPI, UploadFile, File 

# ------------------- FastAPI setup -------------------
web_app = FastAPI()

@web_app.get("/")
def hello():
    return {"message": "Hello from Modal FastAPI!"}

@web_app.get("/status")
def status():
    return {"status": "running on Modal!"}


# ------------------- Modal setup -------------------
# Define the Modal image with required packages
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("fastapi", "uvicorn", "huggingface_hub","requests","python-multipart" )
)


# Create Modal app
app = modal.App("app", image=image)

# Create a persistent volume to store model files
volume = modal.Volume.from_name("vace-models", create_if_missing=True)


# ------------------- Download model -------------------
@app.function(image=image, volumes={"/models": volume} , timeout=3600)
def download_wan_model():
    """
    Downloads Wan-AI/Wan2.1-T2V-14B model from Hugging Face into the /models directory.
    If already present, skips the download.
    """
    from huggingface_hub import snapshot_download
    import os

    target_dir = "/models/Wan2.1-T2V-14B"

    if os.path.exists(os.path.join(target_dir, "config.json")):
        print("✅ Model already exists in /models.")
        return

    os.makedirs(target_dir, exist_ok=True)

    print("⬇️ Downloading Wan2.1-T2V-14B from Hugging Face...")

    try:
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Model successfully downloaded to {target_dir}")
    except Exception as e:
        print(f"⚠️ Download failed: {e}")



# ------------------- Upload video endpoint -------------------
@web_app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts a video upload and saves it to the persistent Modal volume.
    """
    # Folder for uploaded videos
    save_dir = f"/models/training_data/videos"
    os.makedirs(save_dir, exist_ok=True)

    # Save uploaded video to volume
    video_path = os.path.join(save_dir, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📁 Saved video to {video_path}")
    # ✅ Automatically extract frames
    result = extract_frames(video_path)
    return {"message": "Video uploaded and extracted successfully", "path": video_path, "no_frames": len(result) if result else 0}



# ------------------- Helper: Extract frames -------------------
def extract_frames(video_path: str):
    """
    Extracts frames from a video using ffmpeg and deletes the video after extraction.
    """
    try:
        # Prepare output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"/models/training_data/frames/{video_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Run ffmpeg to extract frames (1 frame every 0.1s = 10fps)
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:v", "2",                  # quality
            "-vf", "fps=10",              # 10 frames per second
            f"{output_dir}/frame_%04d.jpg",
            "-v warning"
        ]
        subprocess.run(cmd, check=True)

        # Count frames extracted
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])

        # Delete original video
        os.remove(video_path)

        print(f"🖼️ Extracted {frame_count} frames to {output_dir} and deleted {video_path}")

        return {"frames_dir": output_dir, "frame_count": frame_count}

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {"error": str(e)}



# ------------------- Expose FastAPI -------------------
@app.function(volumes={"/models": volume})
@modal.asgi_app()
def fastapi_app():
    return web_app

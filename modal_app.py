"""
FastAPI + Wan 2.1 T2V model downloader on Modal
"""

import os
import modal
import subprocess
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException

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
    .apt_install("git", "ffmpeg")  # ffmpeg + git for ComfyUI clone
    .pip_install(
        "fastapi",
        "uvicorn",
        "huggingface_hub",
        "requests",
        "python-multipart",
        "Pillow",
    )
    # ✅ Install ComfyUI via comfy-cli (like example.py)
    .pip_install("comfy-cli")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
)

# Create Modal app
app = modal.App("app", image=image)

# Create a persistent volume to store model files
volume = modal.Volume.from_name("vace-models", create_if_missing=True)


# ------------------- Download model -------------------
@app.function(image=image, volumes={"/models": volume}, timeout=3600)
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


# ------------------- Download Wan 2.2 VACE-Fun-A14B model -------------------
@app.function(image=image, volumes={"/models": volume}, timeout=7200)
def download_wan22_vace_model():
    """
    Downloads the Wan2.2 VACE-Fun-A14B model from Hugging Face into the /models directory.
    """
    from huggingface_hub import snapshot_download
    import os

    target_dir = "/models/Wan2.2-VACE-Fun-A14B"

    # Skip download if already exists
    if os.path.exists(os.path.join(target_dir, "config.json")):
        print("✅ Wan2.2 VACE-Fun-A14B already exists in /models.")
        return

    os.makedirs(target_dir, exist_ok=True)
    print("⬇️ Downloading Wan2.2 VACE-Fun-A14B from Hugging Face...")

    try:
        snapshot_download(
            repo_id="alibaba-pai/Wan2.2-VACE-Fun-A14B",
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

    # Check extraction result
    if not result or "frames_dir" not in result:
        # Extraction failed — return HTTP error with details (if available)
        err = result.get("stderr") if isinstance(result, dict) else str(result)
        detail = {"error": result.get("error") if isinstance(result, dict) else "unknown", "stderr": err}
        print(f"⚠️ Frame extraction failed: {detail}")
        raise HTTPException(status_code=500, detail={"message": "Frame extraction failed", **detail})

    # ✅ Prepare LoRA dataset
    dataset_info = prepare_lora_dataset(result["frames_dir"])

    return {"message": "Video uploaded and extracted successfully", "path": video_path, "no_frames": result.get("frame_count", 0), "lora_dataset": dataset_info}



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
        # Pass loglevel as a separate argument and capture stderr for better error messages.
        cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-i", video_path,
            "-q:v", "2",                  # quality
            "-vf", "fps=10",              # 10 frames per second
            f"{output_dir}/frame_%04d.jpg",
        ]
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            # Include stderr in the returned error for easier debugging
            print(f"❌ FFmpeg failed (rc={proc.returncode}): {proc.stderr}")
            return {"error": "ffmpeg_failed", "rc": proc.returncode, "stderr": proc.stderr}

        # Count frames extracted
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])

        # Delete original video
        os.remove(video_path)

        print(f"🖼️ Extracted {frame_count} frames to {output_dir} and deleted {video_path}")

        return {"frames_dir": output_dir, "frame_count": frame_count}

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed (CalledProcessError): {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {"error": str(e)}



# ------------------- Prepare LoRA Dataset -------------------
def prepare_lora_dataset(frames_dir: str):
    """
    Converts extracted frames into a dataset structure for LoRA training.
    Creates /models/training_data/lora_dataset/images and /captions.
    """
    import shutil
    from PIL import Image
    from pathlib import Path

    dataset_dir = Path("/models/training_data/lora_dataset")
    images_dir = dataset_dir / "images"
    captions_dir = dataset_dir / "captions"

    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, frame_path in enumerate(sorted(Path(frames_dir).glob("*.jpg"))):
        img_id = f"{i:06d}"
        target_path = images_dir / f"{img_id}.jpg"

        try:
            # Resize to 512x512 (common for LoRA training)
            img = Image.open(frame_path)
            img = img.resize((512, 512))
            img.save(target_path, "JPEG")

            # Create simple caption text
            with open(captions_dir / f"{img_id}.txt", "w") as f:
                f.write("oxaction")

            count += 1
        except Exception as e:
            print(f"⚠️ Error processing {frame_path}: {e}")

    print(f"✅ Prepared {count} images for LoRA training at {dataset_dir}")
    return {"dataset_dir": str(dataset_dir), "images_count": count}


# ------------------- Launch ComfyUI -------------------
@app.function(
    image=image,
    gpu="L4",
    volumes={"/models": volume},
    timeout=7200,
    scaledown_window=600,  # Keep container alive for 10 min after last request
)
@modal.web_server(port=8188, startup_timeout=180)  # Give it 3 minutes to start
def launch_comfyui():
    """
    Launches ComfyUI with access to the Wan models stored in /models.
    Exposed publicly via Modal web endpoint.
    """
    import os
    import subprocess

    # Set environment variables for better performance (from example.py)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # ComfyUI is installed at /root/comfy/ComfyUI by comfy-cli
    comfy_models = "/root/comfy/ComfyUI/models"
    
    # Create symlinks for model directories to point to persistent volume
    model_types = ["checkpoints", "vae", "loras", "embeddings", "unet", "clip", "diffusion_models"]
    for model_type in model_types:
        src = "/models"
        dst = os.path.join(comfy_models, model_type)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                print(f"✓ Created symlink: {dst} -> {src}")
            except Exception as e:
                print(f"⚠️  Could not create symlink for {model_type}: {e}")
    
    # Also make models directly accessible
    if not os.path.exists(os.path.join(comfy_models, "Wan2.1-T2V-14B")):
        try:
            os.symlink("/models/Wan2.1-T2V-14B", os.path.join(comfy_models, "Wan2.1-T2V-14B"))
            print("✓ Linked Wan2.1 model")
        except Exception as e:
            print(f"⚠️  Wan2.1 link failed: {e}")
    
    if not os.path.exists(os.path.join(comfy_models, "Wan2.2-VACE-Fun-A14B")):
        try:
            os.symlink("/models/Wan2.2-VACE-Fun-A14B", os.path.join(comfy_models, "Wan2.2-VACE-Fun-A14B"))
            print("✓ Linked Wan2.2 model")
        except Exception as e:
            print(f"⚠️  Wan2.2 link failed: {e}")
    
    print(f"🚀 Starting ComfyUI on 0.0.0.0:8188")
    print(f"📁 Models location: {comfy_models}")
    print(f"📁 Volume mounted at: /models")
    
    # Launch ComfyUI using comfy-cli (same pattern as example.py)
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8188", shell=True)


# ------------------- Expose FastAPI -------------------
@app.function(volumes={"/models": volume})
@modal.asgi_app()
def fastapi_app():
    return web_app
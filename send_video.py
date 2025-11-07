import requests

url = "https://baigmuhammadumer2004--app-fastapi-app.modal.run/upload_video"
file_path = "video.mp4"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "video/mp4")}
    response = requests.post(url, files=files)

print("✅ Response:", response.status_code)
print(response.json())

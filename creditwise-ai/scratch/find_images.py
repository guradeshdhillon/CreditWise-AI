import os
from pathlib import Path

app_data = Path(r"C:\Users\gurad\.gemini\antigravity-ide")
images = []
for ext in [".png", ".jpg", ".jpeg", ".webp"]:
    for p in app_data.rglob(f"*{ext}"):
        images.append((p, p.stat().st_mtime))

images.sort(key=lambda x: x[1], reverse=True)
for img, mtime in images[:10]:
    print(f"{img} - {mtime}")

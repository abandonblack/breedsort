from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms

from app.model import build_model
from app.train import DATASET_CATALOG

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FEEDBACK_FILE = BASE_DIR / "feedback.json"
MODEL_PATH = BASE_DIR / "artifacts" / "breednet.pth"

UPLOAD_DIR.mkdir(exist_ok=True)
if not FEEDBACK_FILE.exists():
    FEEDBACK_FILE.write_text("[]", encoding="utf-8")

app = FastAPI(title="猫狗品种智能识别")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: torch.nn.Module | None = None
CLASS_NAMES: list[str] = []
IMAGE_SIZE = 224
ARCH = "resbreednet"


def load_model() -> None:
    global MODEL, CLASS_NAMES, IMAGE_SIZE, ARCH
    if not MODEL_PATH.exists():
        MODEL = None
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    CLASS_NAMES = checkpoint["classes"]
    IMAGE_SIZE = checkpoint.get("image_size", 224)
    ARCH = checkpoint.get("arch", "seresnet34")

    model = build_model(num_classes=len(CLASS_NAMES), arch=ARCH).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    MODEL = model


def infer_image(image_path: Path) -> dict[str, Any]:
    if MODEL is None:
        return {
            "error": (
                "模型权重不存在，请先运行训练脚本："
                "python -m app.train --data-dir data/oxford_iiit_pet"
            )
        }

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with Image.open(image_path).convert("RGB") as image:
        x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_k = min(3, len(CLASS_NAMES))
    scores, indices = torch.topk(probs, k=top_k)
    top3 = [
        {"label": CLASS_NAMES[i.item()], "score": float(s.item())}
        for s, i in zip(scores, indices)
    ]

    best = top3[0]
    return {
        "arch": ARCH,
        "label": best["label"],
        "confidence": round(best["score"] * 100, 2),
        "description": f"{best['label']}：来自自训练 {ARCH} 模型识别结果。",
        "top3": top3,
    }


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/datasets")
def list_datasets():
    return {"items": DATASET_CATALOG}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    suffix = Path(image.filename or "").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    file_path = UPLOAD_DIR / filename

    content = await image.read()
    file_path.write_bytes(content)

    result = infer_image(file_path)
    result["image_url"] = f"/uploads/{filename}"

    if "error" in result:
        return JSONResponse(result, status_code=503)
    return JSONResponse(result)


@app.post("/api/feedback")
def submit_feedback(
    nickname: str = Form(...),
    message: str = Form(...),
    rating: int = Form(...),
):
    try:
        feedback_list = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        feedback_list = []

    feedback = {
        "nickname": nickname,
        "message": message,
        "rating": rating,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    feedback_list.append(feedback)
    FEEDBACK_FILE.write_text(
        json.dumps(feedback_list[-100:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"message": "反馈提交成功，感谢你的建议！"}


@app.get("/api/feedback")
def list_feedback():
    try:
        feedback_list = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        feedback_list = []
    return {"items": list(reversed(feedback_list[-10:]))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

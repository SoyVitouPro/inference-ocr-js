import io
import json
import os
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import onnxruntime as ort


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ocr_encoder.onnx")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab_char.json")

IMG_H = 32
IMG_W = 640

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


def _load_vocab(vocab_path: str) -> Dict[str, Any]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    unk = "<unk>"
    if unk not in vocab_raw:
        max_id = max(vocab_raw.values()) if vocab_raw else -1
        vocab_raw[unk] = max_id + 1

    items = sorted(vocab_raw.items(), key=lambda x: x[1])

    id_to_token = {}
    token_to_id = {}
    for new_id, (tok, _) in enumerate(items):
        token_to_id[tok] = new_id
        id_to_token[new_id] = tok

    blank_id = 0
    pad_id = 1
    ctc_offset = 2
    vocab_size = len(items)

    return {
        "id_to_token": id_to_token,
        "token_to_id": token_to_id,
        "blank_id": blank_id,
        "pad_id": pad_id,
        "ctc_offset": ctc_offset,
        "vocab_size": vocab_size,
        "unk": unk,
    }


VOCAB = _load_vocab(VOCAB_PATH)

SESSION = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    iw, ih = img.size

    scale = IMG_H / max(1, ih)
    nw = max(1, int(round(iw * scale)))

    img1 = img.resize((nw, IMG_H), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))

    if nw == IMG_W:
        canvas.paste(img1, (0, 0))
    elif nw < IMG_W:
        left = (IMG_W - nw) // 2
        canvas.paste(img1, (left, 0))
    else:
        img2 = img1.resize((IMG_W, IMG_H), resample=Image.BICUBIC)
        canvas.paste(img2, (0, 0))

    arr = np.asarray(canvas).astype(np.float32)  # H,W,3
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    gray01 = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    norm = (gray01 - 0.5) / 0.5

    x = norm.reshape(1, 1, IMG_H, IMG_W).astype(np.float32)
    return x


def _argmax_ctc(ctc_logits: np.ndarray) -> np.ndarray:
    # ctc_logits: [1, T, K]
    return np.argmax(ctc_logits[0], axis=-1).astype(np.int32)  # [T]


def _decode_ctc(ids: np.ndarray) -> str:
    blank_id = VOCAB["blank_id"]
    pad_id = VOCAB["pad_id"]
    ctc_offset = VOCAB["ctc_offset"]
    vocab_size = VOCAB["vocab_size"]
    id_to_token = VOCAB["id_to_token"]
    unk = VOCAB["unk"]

    out = []
    prev = None
    for x in ids.tolist():
        if x == blank_id or x == pad_id:
            prev = x
            continue
        if prev == x:
            continue
        prev = x

        y = x - ctc_offset
        if 0 <= y < vocab_size:
            t = id_to_token.get(y, unk)
            if t != unk:
                out.append(t)

    return "".join(out)


@app.get("/")
def home():
    return {"ok": True, "service": "ikhodeocr", "input": [1, 1, IMG_H, IMG_W]}


@app.post("/ocr/batch/printed/")
async def ocr_batch_printed(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for f in files:
        content = await f.read()
        if not content:
            results.append({"text": "", "error": "empty file"})
            continue

        x = _preprocess(content)

        try:
            outputs = SESSION.run(None, {"imgs": x})
        except Exception as e:
            results.append({"text": "", "error": str(e)})
            continue

        # prefer named output if available
        out_names = [o.name for o in SESSION.get_outputs()]
        out_map = {name: outputs[i] for i, name in enumerate(out_names)}

        ctc = out_map.get("ctc_logits", outputs[0] if outputs else None)
        if ctc is None:
            results.append({"text": "", "error": "ctc_logits not found"})
            continue

        pred_ids = _argmax_ctc(ctc)
        text = _decode_ctc(pred_ids)

        results.append({
            "filename": f.filename,
            "text": text,
            "ctc_shape": list(ctc.shape),
        })

    return JSONResponse(content=results)

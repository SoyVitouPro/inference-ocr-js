// khmerocr.js
// Browser OCR (CTC greedy) for ocr_encoder.onnx + vocab_char.json
// Requires onnxruntime-web loaded in index.html (global `ort`)

const IMG_H = 32;
const IMG_W = 640;

export class KhmerOCR {
  constructor({ modelUrl = "/static/ocr_encoder.onnx", vocabUrl = "/static/vocab_char.json" } = {}) {
    this.modelUrl = modelUrl;
    this.vocabUrl = vocabUrl;
    this.session = null;
    this.tok = null;
  }

  async init() {
    if (!window.ort) throw new Error("onnxruntime-web not found. Please load it before khmerocr.js");

    ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
    ort.env.wasm.simd = true;

    this.tok = await this._loadTokenizer(this.vocabUrl);
    this.session = await ort.InferenceSession.create(this.modelUrl, { executionProviders: ["wasm"] });
  }

  async inferFromFile(file) {
    const img = await this._fileToImage(file);
    return await this.inferFromImage(img);
  }

  async inferFromImage(imgEl) {
    if (!this.session || !this.tok) await this.init();

    const x = await preprocessToCHW(imgEl, IMG_H, IMG_W);
    const input = new ort.Tensor("float32", x, [1, 1, IMG_H, IMG_W]);

    const outputs = await this.session.run({ imgs: input });

    const ctc = outputs.ctc_logits || outputs["ctc_logits"];
    const mem = outputs.mem_proj || outputs["mem_proj"];
    if (!ctc) throw new Error("ctc_logits not found in ONNX outputs. Check output names.");

    const dims = ctc.dims;
    if (dims.length !== 3 || dims[0] !== 1) throw new Error("Unexpected ctc_logits shape: " + dims.join("x"));

    const T = dims[1];
    const K = dims[2];

    const predIds = argmax2D(ctc.data, T, K);
    const text = this.tok.decodeCTC(predIds);

    return { text, ctcShape: dims, memShape: mem?.dims || null };
  }

  async _loadTokenizer(vocabUrl, unkToken = "<unk>") {
    const vocabRaw = await (await fetch(vocabUrl)).json();

    if (!(unkToken in vocabRaw)) {
      const maxId = Object.values(vocabRaw).reduce((a, b) => Math.max(a, b), -1);
      vocabRaw[unkToken] = maxId + 1;
    }

    const items = Object.entries(vocabRaw).sort((a, b) => a[1] - b[1]);

    const idToToken = new Map();
    const tokenToId = new Map();
    items.forEach(([tok], newId) => {
      tokenToId.set(tok, newId);
      idToToken.set(newId, tok);
    });

    const blankId = 0;
    const padId = 1;
    const ctcOffset = 2;
    const vocabSize = items.length;

    function decodeCTC(ids) {
      const out = [];
      let prev = null;

      for (let i = 0; i < ids.length; i++) {
        const x = ids[i] | 0;

        if (x === blankId || x === padId) {
          prev = x;
          continue;
        }
        if (prev === x) continue;
        prev = x;

        const y = x - ctcOffset;
        if (y >= 0 && y < vocabSize) {
          const t = idToToken.get(y) ?? unkToken;
          if (t !== unkToken) out.push(t);
        }
      }
      return out.join("");
    }

    return { decodeCTC, blankId, padId, ctcOffset, vocabSize, tokenToId, idToToken };
  }

  _fileToImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = (e) => reject(e);
      img.src = url;
    });
  }
}

export async function preprocessToCHW(imgEl, H = 32, W = 640) {
  const iw = imgEl.naturalWidth;
  const ih = imgEl.naturalHeight;
  const scale = H / Math.max(1, ih);
  const nw = Math.max(1, Math.round(iw * scale));

  const canvas1 = document.createElement("canvas");
  canvas1.width = nw;
  canvas1.height = H;
  const ctx1 = canvas1.getContext("2d", { willReadFrequently: true });
  ctx1.imageSmoothingEnabled = true;
  ctx1.imageSmoothingQuality = "high";
  ctx1.drawImage(imgEl, 0, 0, nw, H);

  const canvas2 = document.createElement("canvas");
  canvas2.width = W;
  canvas2.height = H;
  const ctx2 = canvas2.getContext("2d", { willReadFrequently: true });
  ctx2.imageSmoothingEnabled = true;
  ctx2.imageSmoothingQuality = "high";
  ctx2.fillStyle = "white";
  ctx2.fillRect(0, 0, W, H);

  if (nw === W) {
    ctx2.drawImage(canvas1, 0, 0);
  } else if (nw < W) {
    const padTotal = W - nw;
    const left = Math.floor(padTotal / 2);
    ctx2.drawImage(canvas1, left, 0);
  } else {
    ctx2.drawImage(canvas1, 0, 0, W, H);
  }

  const imgData = ctx2.getImageData(0, 0, W, H).data; // RGBA
  const out = new Float32Array(1 * 1 * H * W);

  let p = 0;
  for (let i = 0; i < H * W; i++) {
    const r = imgData[i * 4 + 0];
    const g = imgData[i * 4 + 1];
    const b = imgData[i * 4 + 2];

    const gray01 = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
    const norm = (gray01 - 0.5) / 0.5;
    out[p++] = norm;
  }

  return out;
}

function argmax2D(logits, T, K) {
  const out = new Int32Array(T);
  for (let t = 0; t < T; t++) {
    let bestI = 0;
    let bestV = logits[t * K + 0];
    for (let k = 1; k < K; k++) {
      const v = logits[t * K + k];
      if (v > bestV) {
        bestV = v;
        bestI = k;
      }
    }
    out[t] = bestI;
  }
  return out;
}

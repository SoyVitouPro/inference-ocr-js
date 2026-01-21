# Fast Khmer OCR API

A high-performance, lightweight OCR (Optical Character Recognition) service specifically optimized for the **Khmer language**. This project uses **FastAPI** for the web interface and **ONNX Runtime** for efficient model inference, ensuring fast response times and low resource consumption.



## 🚀 Features

* **Optimized for Khmer:** Specialized encoder-decoder architecture for Khmer script.
* **ONNX Acceleration:** Uses `.onnx` format for high-speed CPU inference.
* **Batch Processing:** Supports multiple image uploads in a single request.
* **Robust Preprocessing:** Automatically handles image resizing, padding, and grayscale normalization ($32 \times 640$ input).
* **CTC Decoding:** Implements Connectionist Temporal Classification (CTC) for accurate sequence prediction.

---

## 🛠️ Tech Stack

* **Backend:** FastAPI (Python)
* **Inference Engine:** ONNX Runtime
* **Image Processing:** Pillow (PIL), NumPy
* **Server:** Uvicorn

---

## 📋 Prerequisites

Before running the project, ensure you have the following files in the root directory:
1.  `ocr_encoder.onnx`: The trained model file.
2.  `vocab_char.json`: The character vocabulary mapping.

---

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <project-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn onnxruntime numpy Pillow
    ```

3.  **Run the application:**
    You can use the provided `Makefile`:
    ```bash
    make dev
    ```
    *Alternatively, run directly via Uvicorn:*
    ```bash
    python main.py
    ```

The server will start at `http://0.0.0.0:2405`.

---

## 🔌 API Endpoints

### 1. Health Check
* **URL:** `/`
* **Method:** `GET`
* **Response:**
    ```json
    { "ok": true, "service": "ikhodeocr", "input": [1, 1, 32, 640] }
    ```

### 2. Batch OCR Prediction
* **URL:** `/ocr/batch/printed/`
* **Method:** `POST`
* **Body:** `multipart/form-data` (Key: `files`, Value: Image files)
* **Description:** Upload one or more images of printed Khmer text to receive transcriptions.

---

## 📂 Project Structure

```text
├── server_ocr_api.py   # Core logic, preprocessing, and FastAPI routes
├── main.py             # Server entry point
├── ocr_encoder.onnx    # ONNX Model (Required)
├── vocab_char.json     # Vocabulary file (Required)
├── Makefile            # Shortcut commands
└── static/             # Static file storage

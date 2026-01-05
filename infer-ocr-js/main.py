import uvicorn

if __name__ == "__main__":
    uvicorn.run("server_ocr_api:app", host="0.0.0.0", port=2405, reload=True)

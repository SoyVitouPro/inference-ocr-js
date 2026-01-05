import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Serve everything in this folder under /static
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2405, reload=True)
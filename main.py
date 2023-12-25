from fastapi import FastAPI, File
from fastapi.responses import Response, HTMLResponse
from virtual_background import process_image
from typing import Annotated

VIEWS_PATH = "./application_root/views"

app = FastAPI()

@app.post("/file/process-files")
def process_files(
        image: Annotated[bytes, File()],
        background: Annotated[bytes, File()]
):
    result = process_image(image, background)
    return Response(content=result, media_type="image/png")

@app.get("/")
def index():
    with open(f"{VIEWS_PATH}/index.html", "r", encoding='utf-8') as f:
      content = f.read()
    return HTMLResponse(content, status_code=200)

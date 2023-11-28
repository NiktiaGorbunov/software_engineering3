from fastapi import FastAPI, File
from fastapi.responses import Response
from virtual_background import process_image
from typing import Annotated

app = FastAPI()


@app.post("/file/upload-bytes")
def upload_file_bytes(
    image: Annotated[bytes, File()],
    background: Annotated[bytes, File()]
):
  result = process_image(image, background)
  return Response(content=result, media_type="image/png")

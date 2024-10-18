import os
from typing import Iterator
from towhee import AutoPipes
from pydantic import BaseModel


class ImageData(BaseModel):
    file_name: str = ""
    file_path: str = ""
    embedding: list[float]


img_pipe = AutoPipes.pipeline("text_image_embedding")


def embed_img(path) -> list[float]:
    return img_pipe(path).get()[0]


def load_imgs(dir_path: str) -> Iterator[ImageData]:
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.startswith("."):
                continue
            file_path = os.path.abspath(os.path.join(root, f))
            embedding = embed_img(file_path)
            yield ImageData(
                file_name=f,
                file_path=file_path,
                embedding=embedding,
            )

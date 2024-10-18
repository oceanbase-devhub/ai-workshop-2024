import os
from tqdm import tqdm
from typing import Iterator
from .embeddings import ImageData, embed_img, load_imgs

from pyobvector import (
    VECTOR,
    ObVecClient,
)

from sqlalchemy import Column, Integer, JSON, String
from sqlalchemy import func

cols = [
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("file_name", String(512)),
    Column("file_path", String(2048)),
    Column("embedding", VECTOR(512)),
]

output_fields = [
    "id",
    "file_name",
    "file_path",
    # "embedding",
]


class OBImageStore:
    def __init__(
        self,
        *,
        user: str = "",
        uri: str = "",
        db_name: str = "",
        password: str = "",
        table_name: str = "image_store",
        **kwargs,
    ):
        self.table_name = table_name
        self.client = ObVecClient(
            user=user,
            uri=uri,
            db_name=db_name,
            password=password,
        )

    def load_image_dir(self, dir_path: str, batch_size: int = 32) -> Iterator:
        if not self.client.check_table_exists(self.table_name):
            self.client.create_table(self.table_name, columns=cols)
            self.client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")
            self.client.perform_raw_text_sql("SET GLOBAL ob_query_timeout=100000000")
            self.client.create_index(
                self.table_name,
                is_vec_index=True,
                index_name="img_embedding_idx",
                column_names=["embedding"],
                vidx_params="distance=l2, type=hnsw, lib=vsag",
            )
        batch = []
        total = 0
        for _, _, files in os.walk(dir_path):
            total += len(files)
        for img in tqdm(load_imgs(dir_path), total=total):
            batch.append(img.model_dump())
            yield
            if len(batch) == batch_size:
                self.client.insert(self.table_name, batch)
                batch = []
        if len(batch) > 0:
            self.client.insert(self.table_name, batch)

    def search(self, image_path: str, limit: int = 10) -> list[dict[str, any]]:
        target_embedding = embed_img(image_path)

        res = self.client.ann_search(
            self.table_name,
            vec_data=target_embedding,
            vec_column_name="embedding",
            topk=limit,
            distance_func=func.l2_distance,
            output_column_names=output_fields,
            with_dist=True,
        )
        return [
            {
                "id": r[0],
                "file_name": r[1],
                "file_path": r[2],
                "distance": r[3],
            }
            for r in res
        ]

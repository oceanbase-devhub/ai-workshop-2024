####
# OBImageStore with Milvus like client
####

from tqdm import tqdm
from .embeddings import ImageData, embed_img, load_imgs

from pyobvector import (
    MilvusLikeClient,
    FieldSchema,
    DataType,
    CollectionSchema,
    VecIndexType,
)

output_fields = [
    "id",
    "file_name",
    # "embedding",
]

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
        max_length=128,
    ),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
]

index_params = MilvusLikeClient.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_name="img_embedding_idx",
    index_type=VecIndexType.HNSW,
    distance="l2",
    m=16,
    ef_construction=256,
)
schema = CollectionSchema(fields)


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
        self.client = MilvusLikeClient(
            user=user,
            uri=uri,
            db_name=db_name,
            password=password,
        )

        if not self.client.has_collection(table_name):
            self.client.create_collection(table_name, schema=schema)
            print(f"Table {table_name} created")
            self.client.create_index(table_name, index_params)
            print(f"Index created")

    def insert_images(self, image_paths: list[str]):
        for path in tqdm(image_paths):
            embedding = embed_img(path)
            self.client.insert(
                self.table_name,
                [ImageData(file_name=path, embedding=embedding).model_dump()],
            )

    def load_image_dir(self, dir_path: str, batch_size: int = 32):
        batch = []
        for img in tqdm(load_imgs(dir_path)):
            batch.append(img.model_dump())
            if len(batch) == batch_size:
                self.client.insert(self.table_name, batch)
                batch = []
        if len(batch) > 0:
            self.client.insert(self.table_name, batch)

    def search(self, image_path: str, limit: int = 10) -> list[dict[str, any]]:
        target_embedding = embed_img(image_path)

        return self.client.search(
            self.table_name,
            data=target_embedding,
            limit=limit,
            anns_field="embedding",
            output_fields=output_fields,
            with_dist=True,
        )

import os
import dotenv

dotenv.load_dotenv()
from cv.image_store import OBImageStore
from connection import connection_args

if __name__ == "__main__":
    table_name = os.getenv("IMG_TABLE_NAME", "image_search")
    img_base = os.getenv("IMG_BASE", None)
    if img_base is None:
        print("Please set IMG_BASE env variable.")
        exit(1)
    if not os.path.exists(img_base):
        print(f"Image base directory {img_base} does not exist.")
        exit(1)

    store = OBImageStore(
        uri=f"{connection_args['host']}:{connection_args['port']}",
        **connection_args,
        table_name=table_name,
    )
    for _ in store.load_image_dir(img_base):
        # Consume the iterator
        pass

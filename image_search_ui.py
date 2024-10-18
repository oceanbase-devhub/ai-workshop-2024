import os
import time
import dotenv

dotenv.load_dotenv()

import streamlit as st
from cv.image_store import OBImageStore
from connection import connection_args


table_name = os.getenv("IMG_TABLE_NAME", "image_search")
tmp_path = "doc_repos/temp.jpg"


st.set_page_config(
    layout="wide",
    page_title="Image Search",
    page_icon="demo/ob-icon.png",
)
st.title("🔍 Image Search")
st.caption("🚀 Search similar images using OceanBase and Streamlit")

with st.sidebar:
    st.title("🔧 Settings")
    st.logo("demo/logo.png")
    st.caption("🚀 Configure the application")
    st.subheader("Search Images Settings")
    top_k = st.slider("Top K", 1, 30, 10, help="Number of similar images to return.")
    show_distance = st.checkbox("Show Distance", value=True)
    show_file_path = st.checkbox("Show File Path", value=True)

    table_name = st.text_input("Table Name", table_name)

    st.subheader("Load Images Settings")
    image_base = st.text_input(
        "Image Base",
        os.getenv("IMG_BASE", None),
        help="Base directory of images to load.",
        placeholder="Absolute path to image directory.",
    )
    click_load = st.button("Load Images")
    st.caption("📌 Load images from the image base directory.")


store = OBImageStore(
    uri=f"{connection_args['host']}:{connection_args['port']}",
    **connection_args,
    table_name=table_name,
)

table_exist = store.client.check_table_exists(table_name)
if not table_name:
    st.error("Please set the Table Name.")
    st.stop()
elif click_load:
    if image_base is None:
        st.error("Please set the Image Base.")
    elif not os.path.exists(image_base):
        st.error(f"Image base directory {image_base} does not exist.")
    else:
        total = 0
        for _, _, files in os.walk(image_base):
            total += len(files)
        finished = 0
        bar = st.progress(0, text="Loading images...")
        for _ in store.load_image_dir(image_base):
            finished += 1
            bar.progress(
                finished / total,
                text=f"Loading images... (finished {finished} / {total})",
            )
        st.toast("Images loaded successfully.", icon="🎉")
        st.balloons()
        time.sleep(2.5)

        st.rerun()
elif table_exist:
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        col1.subheader("Uploaded Image")
        col1.caption("📌 Image you uploaded")
        col1.image(uploaded_file, use_column_width=True)
        
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        col2.subheader("Similar Images")
        results = store.search(tmp_path, limit=top_k)
        with col2:
            if len(results) == 0:
                st.warning("No similar images found.")
            else:
                tabs = st.tabs([f"Image {i+1}" for i in range(len(results))])
                for res, tab in zip(results, tabs):
                    with tab:
                        if show_distance:
                            st.write(f"📏 Distance: {res['distance']:.8f}")
                        if show_file_path:
                            st.write("📂 File Path:", os.path.join(res["file_path"]))
                        st.image(res["file_path"])
else:
    st.warning("Table does not exist. Please load images first.")

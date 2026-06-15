from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.app_pages import PAGES


st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.sidebar.title("Breast Cancer ML")
page_name = st.sidebar.radio("Chọn trang", list(PAGES.keys()))

PAGES[page_name]()

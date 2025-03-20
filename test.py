# test_streamlit_log.py
import streamlit as st
from config import logger

logger.debug("Streamlit app started.")

st.title("Test Logging")

st.write("Hello, logging!")

logger.info("Streamlit app finished.")
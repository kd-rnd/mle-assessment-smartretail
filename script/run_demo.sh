uvicorn api:app --reload --host 0.0.0.0 --port 8000 &
streamlit run  streamlit.py --server.port 8501 --server.fileWatcherType none
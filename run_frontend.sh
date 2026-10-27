#!/usr/bin/env bash
cd "$(dirname "$0")" && python -m streamlit run frontend/app.py --server.port 8501

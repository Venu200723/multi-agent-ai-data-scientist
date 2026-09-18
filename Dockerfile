# Multi-Agent AI Data Scientist — Streamlit demo image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY multi_agent_ds ./multi_agent_ds
COPY data ./data
COPY examples ./examples
COPY artifacts ./artifacts
COPY tests ./tests

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "multi_agent_ds/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

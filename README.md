# Electronix AI Assignment

## Overview
This project is an end-to-end microservice for binary sentiment analysis, featuring a Python backend (FastAPI + Hugging Face), a React frontend, and a CLI fine-tuning script. All components are containerized with Docker Compose.

---

## Setup & Run Instructions

### Prerequisites
- Docker & Docker Compose installed
- Python 3.9+ (if running without Docker)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library

### Running with Docker (Recommended)
```bash
docker-compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

### Running Locally (Without Docker)
1. Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2. Install backend dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```
3. Start the backend:
    ```bash
    uvicorn backend.app.main:app --reload
    ```
4. Start the frontend:
    ```bash
    cd frontend
    npm install
    npm start
    ```

---

## Hugging Face Transformers
This project uses the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library for model loading, inference, and fine-tuning. Transformers provides state-of-the-art pre-trained models for NLP tasks. In this project, it is used to:
- Load a sentiment analysis model in the backend
- Run inference (predict sentiment)
- Fine-tune the model on your own data using `finetune.py`

See the [Transformers documentation](https://huggingface.co/docs/transformers/index) for more information and advanced usage.

### Fine-tuning
To fine-tune the model on your own data:
```bash
python finetune.py --data data.jsonl --epochs 3 --lr 3e-5
```
This saves updated weights to `./model/`.

---

## Design Decisions
- **FastAPI** for simple, async REST API.
- **Hugging Face Transformers** for model loading & inference.
- **React** for a minimal, modern frontend.
- **Docker Compose** for easy local development and reproducibility.

---

## API Docs
- **POST /predict**
  - Request: `{ "text": "your sentence here" }`
  - Response: `{ "label": "positive"|"negative", "score": float }`

---

## Example Data
```
{"text": "I love this product!", "label": "positive"}
{"text": "This is terrible.", "label": "negative"}
```

---

## Approx. Fine-tune Time
- CPU: ~5-10 min for 100 samples, 3 epochs
- GPU (if available): ~1-2 min

---

## Deliverables
- Codebase (backend, frontend, finetune script)
- Docker files & docker-compose
- README & API docs
- Example data

---

## Author
Team Electronix AI
#

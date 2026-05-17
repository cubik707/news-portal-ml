# news-portal-ml

Python ML microservice for personalized news recommendations.

## Model

Content-based filtering using TF-IDF vectorization and cosine similarity, augmented with a temporal decay factor:

```
Score(a, u) = α · CosSim(v(a), P(u)) + (1 − α) · exp(−λ · Δt)
```

## Requirements

- Python 3.12+
- PostgreSQL (running instance)

## Running locally

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS**
  ```bash
  source venv/bin/activate
  ```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your database credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=news_portal
DB_USERNAME=postgres
DB_PASSWORD=your_password

# Optional — model tuning
ALPHA=0.7         # weight between content similarity and recency (0..1)
LAMBDA_DECAY=0.05 # recency decay rate

ML_PORT=8000
```

### 4. Start the server

```bash
uvicorn main:app --reload --port 8000
```

The service will be available at `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

---

## Running with Docker

```bash
docker build -t news-portal-ml .
docker run --env-file .env -p 8000:8000 news-portal-ml
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Healthcheck |
| GET | `/recommend/{user_id}?limit=10` | Personalized article IDs |

### Example

```bash
# health check
curl http://localhost:8000/health

# get recommendations for user "42", top 5
curl "http://localhost:8000/recommend/42?limit=5"
```

Response:

```json
{
  "user_id": "42",
  "news_ids": ["101", "205", "88", "312", "77"],
  "model": "tfidf-cosine-v1"
}
```

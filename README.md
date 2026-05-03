# news-portal-ml

Python ML microservice for personalized news recommendations.

## Model

Content-based filtering using TF-IDF vectorization and cosine similarity, augmented with a temporal decay factor:

```
Score(a, u) = α · CosSim(v(a), P(u)) + (1 − α) · exp(−λ · Δt)
```

## Setup

```bash
cp .env.example .env
# fill in DB credentials

pip install -r requirements.txt
uvicorn main:app --reload
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Healthcheck |
| GET | `/recommend/{user_id}?limit=10` | Personalized article IDs |

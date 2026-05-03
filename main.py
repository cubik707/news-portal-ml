import logging

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import recommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="News Portal ML Service", version="1.0.0")


class RecommendResponse(BaseModel):
    user_id: str
    news_ids: list[str]
    model: str = "tfidf-cosine-v1"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def get_recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50),
):
    try:
        news_ids = recommender.recommend(user_id, limit=limit)
        return RecommendResponse(user_id=user_id, news_ids=news_ids)
    except Exception as exc:
        logger.exception("Failed to compute recommendations for user %s", user_id)
        raise HTTPException(status_code=500, detail="Recommendation engine error") from exc

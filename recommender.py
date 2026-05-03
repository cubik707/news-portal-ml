from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import database


# Interaction weights
WEIGHT_VIEW = 1
WEIGHT_COMMENT = 2
WEIGHT_LIKE = 3
WEIGHT_SUBSCRIPTION = 5


def _days_since(published_at: datetime) -> float:
    if published_at is None:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    delta = datetime.now(tz=timezone.utc) - published_at
    return max(delta.total_seconds() / 86400, 0.0)


def _build_doc(article: dict[str, Any]) -> str:
    parts = [
        article.get("title", ""),
        article.get("content", ""),
        article.get("category_name", ""),
        article.get("tags", ""),
    ]
    return " ".join(p for p in parts if p)


def recommend(user_id: str, limit: int = 10) -> list[str]:
    articles = database.get_published_news()
    if not articles:
        return []

    interactions = database.get_user_interactions(user_id)

    # ── Build TF-IDF matrix ────────────────────────────────────────────────────
    corpus = [_build_doc(a) for a in articles]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)  # (N_articles, N_features)
    article_ids = [str(a["id"]) for a in articles]
    id_to_index = {aid: i for i, aid in enumerate(article_ids)}

    # ── Build user profile ─────────────────────────────────────────────────────
    weighted_sum = np.zeros(tfidf_matrix.shape[1])
    total_weight = 0.0

    def _add(news_id: str, weight: int) -> None:
        nonlocal total_weight
        idx = id_to_index.get(str(news_id))
        if idx is not None:
            weighted_sum[:] += weight * tfidf_matrix[idx].toarray()[0]
            total_weight += weight

    for nid in interactions["views"]:
        _add(nid, WEIGHT_VIEW)
    for nid in interactions["comments"]:
        _add(nid, WEIGHT_COMMENT)
    for nid in interactions["likes"]:
        _add(nid, WEIGHT_LIKE)

    # Subscriptions: add all articles from subscribed categories
    if interactions["subscriptions"]:
        sub_news_ids = database.get_news_ids_by_categories(interactions["subscriptions"])
        for nid in sub_news_ids:
            _add(nid, WEIGHT_SUBSCRIPTION)

    user_profile: np.ndarray | None = None
    if total_weight > 0:
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            user_profile = weighted_sum / norm

    # ── Compute scores ─────────────────────────────────────────────────────────
    if user_profile is not None:
        similarities = cosine_similarity(
            user_profile.reshape(1, -1), tfidf_matrix
        )[0]
    else:
        similarities = np.zeros(len(articles))

    recency = np.array([
        np.exp(-config.LAMBDA_DECAY * _days_since(a["published_at"]))
        for a in articles
    ])

    scores = config.ALPHA * similarities + (1 - config.ALPHA) * recency

    # ── Filter already-viewed articles ────────────────────────────────────────
    viewed_ids = {str(nid) for nid in interactions["views"]}

    ranked = sorted(
        [
            (article_ids[i], float(scores[i]))
            for i in range(len(articles))
            if article_ids[i] not in viewed_ids
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return [news_id for news_id, _ in ranked[:limit]]

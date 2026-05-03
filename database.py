from typing import Any
import psycopg2
import psycopg2.extras
import config


def _connect():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def get_published_news() -> list[dict[str, Any]]:
    """Return all published articles with category name and aggregated tag names."""
    sql = """
        SELECT
            n.id,
            n.title,
            n.content,
            c.name AS category_name,
            COALESCE(STRING_AGG(t.name, ' '), '') AS tags,
            n.published_at
        FROM news n
        JOIN news_categories c ON c.id = n.category_id
        LEFT JOIN news_tags nt ON nt.news_id = n.id
        LEFT JOIN tags t ON t.id = nt.tag_id
        WHERE n.status = 'published'
        GROUP BY n.id, c.name, n.published_at
        ORDER BY n.published_at DESC
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]


def get_user_interactions(user_id: str) -> dict[str, list]:
    """Return all interaction signals for a given user."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT news_id FROM news_views WHERE user_id = %s", (user_id,)
            )
            views = [row["news_id"] for row in cur.fetchall()]

            cur.execute(
                "SELECT news_id FROM likes WHERE user_id = %s", (user_id,)
            )
            likes = [row["news_id"] for row in cur.fetchall()]

            cur.execute(
                """
                SELECT DISTINCT news_id FROM comments
                WHERE author_id = %s
                """,
                (user_id,),
            )
            comments = [row["news_id"] for row in cur.fetchall()]

            cur.execute(
                "SELECT category_id FROM user_subscriptions WHERE user_id = %s",
                (user_id,),
            )
            subscriptions = [str(row["category_id"]) for row in cur.fetchall()]

    return {
        "views": views,
        "likes": likes,
        "comments": comments,
        "subscriptions": subscriptions,
    }


def get_news_ids_by_categories(category_ids: list[str]) -> list[str]:
    """Return IDs of published articles belonging to given categories."""
    if not category_ids:
        return []
    sql = """
        SELECT id FROM news
        WHERE status = 'published'
          AND category_id = ANY(%s)
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (category_ids,))
            return [str(row[0]) for row in cur.fetchall()]

"""
src/preprocessing/mastodon_collector.py

Verzamelt berichten van Mastodon via de publieke API op basis van hashtags.
Sla je MASTODON_ACCESS_TOKEN op in een .env bestand.

Gebruik:
    python src/preprocessing/mastodon_collector.py
"""

import os
import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from mastodon import Mastodon
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs

load_dotenv()


def clean_html(html_text: str) -> str:
    """Verwijder HTML-tags uit Mastodon-berichten."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ").strip()


def extract_post_features(status: dict) -> dict:
    """Haal relevante kenmerken op uit een Mastodon-status object."""
    content = clean_html(status.get("content", ""))
    account = status.get("account", {})

    return {
        "id": status.get("id"),
        "content": content,
        "content_length": len(content),
        "created_at": status.get("created_at"),
        "language": status.get("language", "unknown"),
        "url": status.get("url"),
        "replies_count": status.get("replies_count", 0),
        "reblogs_count": status.get("reblogs_count", 0),
        "favourites_count": status.get("favourites_count", 0),
        "account_username": account.get("username", ""),
        "account_followers": account.get("followers_count", 0),
        "account_following": account.get("following_count", 0),
        "account_statuses": account.get("statuses_count", 0),
        "account_created_at": account.get("created_at"),
        "account_bot": account.get("bot", False),
        # Bronbetrouwbaarheid (berekend later op basis van accountkenmerken)
        "source_trust_score": None,
        # Label moet handmatig worden toegevoegd via factcheckers
        "label": None,
        "hashtag_used": None,  # wordt ingevuld tijdens verzameling
    }


def estimate_source_trust(row: pd.Series) -> float:
    """
    Schat de bronbetrouwbaarheidsscore (1-10) op basis van accountkenmerken.
    Dit is een heuristiek — verfijn dit op basis van je eigen criteria.

    Factoren:
    - Aantal volgers (meer = betrouwbaarder, tot een plafond)
    - Verhouding volgers/volgend (hoge verhouding = meer autoriteit)
    - Aantal berichten (actief account = betrouwbaarder)
    - Account ouderdom (oudere accounts = iets betrouwbaarder)
    - Of het een bot is (bots krijgen lagere score)
    """
    score = 5.0  # neutraal startpunt

    # Bot-penalty
    if row.get("account_bot", False):
        score -= 2.0

    # Volgersscore (logaritmisch geschaald)
    followers = max(row.get("account_followers", 0), 1)
    import math
    follower_bonus = min(math.log10(followers) * 0.8, 3.0)
    score += follower_bonus

    # Volgers/volgend verhouding
    following = max(row.get("account_following", 1), 1)
    ratio = followers / following
    if ratio > 2:
        score += 0.5
    elif ratio < 0.1:
        score -= 1.0

    # Account activiteit
    statuses = row.get("account_statuses", 0)
    if statuses > 100:
        score += 0.5
    elif statuses < 5:
        score -= 1.0

    return max(1.0, min(10.0, round(score, 2)))


def collect_mastodon_posts(config: dict) -> pd.DataFrame:
    """Verzamel berichten van Mastodon via hashtag-zoekopdrachten."""
    access_token = os.getenv("MASTODON_ACCESS_TOKEN")
    instance_url = config["mastodon"]["instance_url"]

    if not access_token:
        raise ValueError(
            "MASTODON_ACCESS_TOKEN niet gevonden in omgevingsvariabelen. "
            "Sla je token op in .env"
        )

    mastodon = Mastodon(
        access_token=access_token,
        api_base_url=instance_url,
    )

    hashtags = config["mastodon"]["hashtags"]
    max_posts = config["mastodon"]["max_posts"]
    min_length = config["mastodon"]["min_post_length"]
    all_posts = []
    seen_ids = set()

    logger.info(f"Start verzameling van Mastodon-berichten van {instance_url}")
    logger.info(f"Hashtags: {hashtags}")
    logger.info(f"Doel: {max_posts} berichten")

    posts_per_hashtag = max(max_posts // len(hashtags), 50)

    for hashtag in hashtags:
        logger.info(f"  Hashtag #{hashtag} verzamelen...")
        collected = 0
        max_id = None

        while collected < posts_per_hashtag:
            try:
                kwargs = {"limit": 40}
                if max_id:
                    kwargs["max_id"] = max_id

                statuses = mastodon.timeline_hashtag(hashtag, **kwargs)

                if not statuses:
                    logger.info(f"    Geen berichten meer voor #{hashtag}")
                    break

                for status in statuses:
                    post = extract_post_features(status)

                    # Filter: geen duplicaten, minimale lengte, geen lege berichten
                    if (
                        post["id"] not in seen_ids
                        and len(post["content"]) >= min_length
                        and post["content"].strip()
                    ):
                        post["hashtag_used"] = hashtag
                        all_posts.append(post)
                        seen_ids.add(post["id"])
                        collected += 1

                max_id = statuses[-1]["id"]
                time.sleep(1)  # Respecteer de rate limits

            except Exception as e:
                logger.error(f"    Fout bij ophalen van #{hashtag}: {e}")
                time.sleep(5)
                break

        logger.info(f"    {collected} berichten verzameld voor #{hashtag}")

    df = pd.DataFrame(all_posts)
    logger.info(f"Totaal unieke berichten verzameld: {len(df)}")
    return df


def add_trust_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg geschatte bronbetrouwbaarheidsscores toe."""
    df["source_trust_score"] = df.apply(estimate_source_trust, axis=1)
    return df


def save_for_labeling(df: pd.DataFrame, output_path: str) -> None:
    """
    Sla berichten op als CSV voor handmatige labeling.

    De kolom 'label' moet handmatig worden ingevuld:
    - 1 = waar (geverifieerd door factcheckers)
    - 0 = onwaar (weerlegd door factcheckers)
    - Laat leeg als onduidelijk

    Aanbevolen factcheck-bronnen:
    - VRT NWS Factcheck: https://www.vrt.be/vrtnws/nl/factcheck/
    - PolitiFact: https://www.politifact.com/
    - Knack Factcheck: https://www.knack.be/tag/factcheck/
    """
    ensure_dirs(os.path.dirname(output_path))
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Berichten opgeslagen voor handmatige labeling: {output_path}")
    logger.info(
        f"VOLGENDE STAP: Open {output_path} en vul de 'label' kolom in "
        f"(1=waar, 0=onwaar) aan de hand van factcheck-bronnen."
    )


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])
    ensure_dirs(config["logging"]["log_dir"])

    output_file = config["mastodon"]["output_file"]

    df = collect_mastodon_posts(config)
    df = add_trust_scores(df)
    save_for_labeling(df, output_file)

    # Statistieken
    logger.info("\n=== VERZAMELSTATISTIEKEN ===")
    logger.info(f"Totaal berichten: {len(df)}")
    logger.info(f"Gemiddelde berichtlengte: {df['content_length'].mean():.1f} tekens")
    logger.info(f"Taalverdeling:\n{df['language'].value_counts().head(5)}")
    logger.info(f"Hashtag-verdeling:\n{df['hashtag_used'].value_counts()}")


if __name__ == "__main__":
    main()

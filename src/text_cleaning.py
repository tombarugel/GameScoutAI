"""
Text-cleaning utilities for App Store game descriptions.

The goal is to preserve terms related to gameplay, mechanics and themes while
removing URLs, legal notices and generic mobile-marketing vocabulary.
"""

from __future__ import annotations

import re


# Generic words that are frequent in App Store descriptions but carry little
# information about the actual mechanics or positioning of a game.
DOMAIN_STOPWORDS = {
    "app",
    "apps",
    "apple",
    "best",
    "com",
    "download",
    "facebook",
    "free",
    "fun",
    "game",
    "games",
    "gaming",
    "google",
    "http",
    "https",
    "iphone",
    "ipad",
    "mobile",
    "new",
    "online",
    "play",
    "player",
    "players",
    "playing",
    "privacy",
    "support",
    "terms",
    "today",
    "website",
    "www",
}


URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

HTML_PATTERN = re.compile(
    r"<[^>]+>"
)

NON_LETTER_PATTERN = re.compile(
    r"[^a-zA-Z\s-]"
)

MULTIPLE_SPACES_PATTERN = re.compile(
    r"\s+"
)


def clean_game_description(text: object) -> str:
    """
    Clean one App Store description.

    The function removes:
    - URLs;
    - email addresses;
    - basic HTML;
    - numbers and punctuation;
    - generic App Store marketing terms;
    - immediately repeated words.

    Args:
        text:
            Raw description or any object convertible to text.

    Returns:
        A normalized string suitable for TF-IDF.
    """
    if text is None:
        return ""

    cleaned_text = str(text).lower()

    cleaned_text = URL_PATTERN.sub(
        " ",
        cleaned_text,
    )

    cleaned_text = EMAIL_PATTERN.sub(
        " ",
        cleaned_text,
    )

    cleaned_text = HTML_PATTERN.sub(
        " ",
        cleaned_text,
    )

    cleaned_text = NON_LETTER_PATTERN.sub(
        " ",
        cleaned_text,
    )

    cleaned_text = MULTIPLE_SPACES_PATTERN.sub(
        " ",
        cleaned_text,
    ).strip()

    filtered_words: list[str] = []
    previous_word: str | None = None

    for word in cleaned_text.split():
        normalized_word = word.strip("-")

        if len(normalized_word) < 3:
            continue

        if normalized_word in DOMAIN_STOPWORDS:
            continue

        # Marketing descriptions sometimes repeat the same word many times.
        if normalized_word == previous_word:
            continue

        filtered_words.append(normalized_word)
        previous_word = normalized_word

    return " ".join(filtered_words)


def normalize_genres(text: object) -> str:
    """
    Normalize the genre information and repeat it to give it controlled weight.

    Repetition is intentional: genres are reliable structured information and
    should influence the text representation more than a single occurrence
    would.
    """
    if text is None:
        return ""

    genre_text = str(text).lower()

    genre_text = genre_text.replace(
        "|",
        " ",
    )

    genre_text = NON_LETTER_PATTERN.sub(
        " ",
        genre_text,
    )

    genre_text = MULTIPLE_SPACES_PATTERN.sub(
        " ",
        genre_text,
    ).strip()

    genre_words = [
        word
        for word in genre_text.split()
        if word not in DOMAIN_STOPWORDS
        and len(word) >= 3
    ]

    normalized_genres = " ".join(
        genre_words
    )

    # Two repetitions give genres additional influence without dominating
    # the complete game description.
    return (
        f"{normalized_genres} "
        f"{normalized_genres}"
    ).strip()
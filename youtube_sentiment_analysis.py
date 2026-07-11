import argparse
import re
import sys
from collections import Counter
from typing import List, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)


ANALYZER = SentimentIntensityAnalyzer()


def extract_video_id(url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname.endswith("youtube.com"):
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts"}:
            return path_parts[1]
    elif hostname.endswith("youtu.be"):
        return parsed.path.strip("/")

    match = re.search(r"([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    raise ValueError("Could not extract a YouTube video ID from the provided URL.")


def get_transcript(video_id: str) -> List[str]:
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return [entry.text for entry in transcript]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return []


def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return hostname.endswith("youtube.com") or hostname.endswith("youtu.be")


def get_website_text(url: str) -> List[str]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "nav", "form", "svg"]):
        tag.decompose()

    texts = []
    for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)

    if not texts:
        full_text = soup.get_text(separator=" ", strip=True)
        if full_text:
            texts = [full_text]

    return texts


def analyze_sentiment(text: str) -> Tuple[str, float]:
    score = ANALYZER.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, score


def summarize(items: List[str]) -> None:
    if not items:
        print("No text was available to analyze.")
        return

    labels = [analyze_sentiment(item)[0] for item in items]
    counts = Counter(labels)
    print("\nSentiment summary:")
    print(f"Positive: {counts.get('positive', 0)}")
    print(f"Neutral: {counts.get('neutral', 0)}")
    print(f"Negative: {counts.get('negative', 0)}")

    average_score = sum(analyze_sentiment(item)[1] for item in items) / len(items)
    print(f"Average compound score: {average_score:.3f}")

    if average_score > 0.05:
        print("Overall sentiment: positive")
    elif average_score < -0.05:
        print("Overall sentiment: negative")
    else:
        print("Overall sentiment: neutral")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sentiment for a YouTube video transcript or website page")
    parser.add_argument("target", help="YouTube video URL/ID or website URL")
    parser.add_argument("--mode", choices=["auto", "transcript", "website"], default="auto")
    args = parser.parse_args()

    if args.mode == "auto":
        mode = "transcript" if is_youtube_url(args.target) else "website"
    else:
        mode = args.mode

    if mode == "transcript":
        video_id = extract_video_id(args.target)
        print(f"Video ID: {video_id}")
        print("Fetching transcript...")
        transcript = get_transcript(video_id)
        if not transcript:
            print("No transcript was found for this video, or transcripts are not available.")
            sys.exit(0)
        print(f"Analyzing {len(transcript)} transcript segments...")
        summarize(transcript)
    else:
        print(f"Fetching website content from: {args.target}")
        website_text = get_website_text(args.target)
        if not website_text:
            print("Could not extract readable text from the website.")
            sys.exit(0)
        print(f"Analyzing {len(website_text)} text blocks from the website...")
        summarize(website_text)


if __name__ == "__main__":
    main()

import argparse
import re
import sys
from collections import Counter
from typing import List, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)


ANALYZER = SentimentIntensityAnalyzer()


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract a YouTube video ID from the provided URL.")


def get_transcript(video_id: str) -> List[str]:
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return [entry.text for entry in transcript]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return []


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
    parser = argparse.ArgumentParser(description="Analyze sentiments for a YouTube video using its transcript")
    parser.add_argument("video", help="YouTube video URL or video ID")
    parser.add_argument("--source", choices=["transcript"], default="transcript")
    args = parser.parse_args()

    video_id = extract_video_id(args.video)
    print(f"Video ID: {video_id}")

    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    if not transcript:
        print("No transcript was found for this video, or transcripts are not available.")
        sys.exit(0)

    print(f"Analyzing {len(transcript)} transcript segments...")
    summarize(transcript)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scrape before/after surgery images from Reddit for SurgeryAudit training data.

Collects REAL surgery before/after photos from medical/cosmetic subreddits.
These serve as the "authentic" class in the AI vs real classifier.

Usage:
    # First: create a Reddit app at https://www.reddit.com/prefs/apps
    # Set environment variables:
    #   REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

    python scrape_reddit.py --subreddit PlasticSurgery --limit 500 --output-dir ./data/real
    python scrape_reddit.py --all --limit 200 --output-dir ./data/real
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import praw
import requests
from tqdm import tqdm

# Subreddits with real surgery before/after photos
REAL_SURGERY_SUBS = [
    "PlasticSurgery",
    "cosmeticsurgery",
    "Rhinoplasty",
    "jawsurgery",
    "PlasticSurgeryBeforeAfter",
]

# Subreddits with manipulated/filtered/AI photos (for negative examples)
MANIPULATED_SUBS = [
    "InstagramReality",       # Edited/filtered vs reality
    "Instagramvsreality",     # Same
]

# Image extensions we accept
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_reddit(client_id=None, client_secret=None, user_agent=None):
    """Initialize Reddit API client."""
    client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT", "SurgeryAuditDataPipeline/1.0")

    if not client_id or not client_secret:
        print("ERROR: Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        print("Create an app at: https://www.reddit.com/prefs/apps")
        print("Choose 'script' type, use http://localhost:8080 as redirect URI.")
        sys.exit(1)

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def is_image_url(url):
    """Check if URL points to an image."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def extract_image_urls(submission):
    """Extract image URLs from a Reddit submission."""
    urls = []

    # Direct image link
    if is_image_url(submission.url):
        urls.append(submission.url)

    # Reddit gallery
    if hasattr(submission, "is_gallery") and submission.is_gallery:
        if hasattr(submission, "media_metadata") and submission.media_metadata:
            for item in submission.media_metadata.values():
                if item.get("status") == "valid" and "s" in item:
                    url = item["s"].get("u", "")
                    # Reddit encodes URLs in gallery metadata
                    url = url.replace("&amp;", "&")
                    if url:
                        urls.append(url)

    # i.redd.it links
    if "i.redd.it" in submission.url:
        if submission.url not in urls:
            urls.append(submission.url)

    # imgur single image (not album)
    if "imgur.com" in submission.url and not "/a/" in submission.url:
        img_url = submission.url
        if not is_image_url(img_url):
            img_url = img_url + ".jpg"
        if img_url not in urls:
            urls.append(img_url)

    return urls


def download_image(url, output_path, timeout=15):
    """Download an image from URL."""
    try:
        headers = {"User-Agent": "SurgeryAuditDataPipeline/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return False

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify it's a valid image (at least 10KB)
        if os.path.getsize(output_path) < 10240:
            os.remove(output_path)
            return False

        return True
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def scrape_subreddit(reddit, subreddit_name, output_dir, limit=500, label="real",
                     sort="top", time_filter="all"):
    """Scrape images from a subreddit."""
    sub_dir = os.path.join(output_dir, label, subreddit_name)
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    subreddit = reddit.subreddit(subreddit_name)

    if sort == "top":
        submissions = subreddit.top(time_filter=time_filter, limit=limit * 3)
    elif sort == "hot":
        submissions = subreddit.hot(limit=limit * 3)
    else:
        submissions = subreddit.new(limit=limit * 3)

    downloaded = 0
    skipped = 0
    metadata_records = []

    print(f"\nScraping r/{subreddit_name} ({sort}/{time_filter}, target: {limit})...")

    for submission in tqdm(submissions, total=limit, desc=f"r/{subreddit_name}"):
        if downloaded >= limit:
            break

        # Skip videos, text posts
        if submission.is_video or submission.is_self:
            continue

        # Skip NSFW if not relevant
        # (PlasticSurgery posts are often marked NSFW but are medical)

        urls = extract_image_urls(submission)
        if not urls:
            skipped += 1
            continue

        for i, url in enumerate(urls):
            if downloaded >= limit:
                break

            # Hash-based filename to avoid duplicates
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = os.path.splitext(urlparse(url).path)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                ext = ".jpg"
            filename = f"{subreddit_name}_{url_hash}{ext}"
            filepath = os.path.join(sub_dir, filename)

            if os.path.exists(filepath):
                skipped += 1
                continue

            if download_image(url, filepath):
                downloaded += 1
                metadata_records.append({
                    "filename": filename,
                    "subreddit": subreddit_name,
                    "label": label,
                    "post_id": submission.id,
                    "title": submission.title[:200],
                    "score": submission.score,
                    "url": url,
                    "post_url": f"https://reddit.com{submission.permalink}",
                    "created_utc": int(submission.created_utc),
                    "num_comments": submission.num_comments,
                    "flair": str(submission.link_flair_text or ""),
                    "is_gallery": getattr(submission, "is_gallery", False),
                    "image_index": i,
                })

                # Rate limiting
                time.sleep(0.5)

    # Save metadata
    meta_file = os.path.join(meta_dir, f"{subreddit_name}_{label}.json")
    with open(meta_file, "w") as f:
        json.dump(metadata_records, f, indent=2)

    print(f"  Downloaded: {downloaded} | Skipped: {skipped} | Metadata: {meta_file}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Scrape Reddit for surgery training data")
    parser.add_argument("--subreddit", "-s", help="Single subreddit to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all configured subreddits")
    parser.add_argument("--real-only", action="store_true", help="Only scrape real surgery subs")
    parser.add_argument("--manipulated-only", action="store_true", help="Only scrape manipulated/filtered subs")
    parser.add_argument("--limit", "-l", type=int, default=200, help="Images per subreddit (default: 200)")
    parser.add_argument("--output-dir", "-o", default="./data", help="Output directory")
    parser.add_argument("--sort", choices=["top", "hot", "new"], default="top")
    parser.add_argument("--time-filter", choices=["all", "year", "month", "week"], default="all")
    parser.add_argument("--client-id", help="Reddit client ID (or set REDDIT_CLIENT_ID)")
    parser.add_argument("--client-secret", help="Reddit client secret (or set REDDIT_CLIENT_SECRET)")
    args = parser.parse_args()

    reddit = get_reddit(args.client_id, args.client_secret)

    total = 0
    if args.subreddit:
        label = "manipulated" if args.subreddit in MANIPULATED_SUBS else "real"
        total += scrape_subreddit(reddit, args.subreddit, args.output_dir, args.limit,
                                  label=label, sort=args.sort, time_filter=args.time_filter)
    elif args.all or args.real_only or args.manipulated_only:
        subs_to_scrape = []
        if args.all or args.real_only:
            subs_to_scrape += [(s, "real") for s in REAL_SURGERY_SUBS]
        if args.all or args.manipulated_only:
            subs_to_scrape += [(s, "manipulated") for s in MANIPULATED_SUBS]

        for sub_name, label in subs_to_scrape:
            total += scrape_subreddit(reddit, sub_name, args.output_dir, args.limit,
                                      label=label, sort=args.sort, time_filter=args.time_filter)
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nTotal images downloaded: {total}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

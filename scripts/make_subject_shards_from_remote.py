#!/usr/bin/env python3
import argparse
import os
import re
import urllib.request
from html.parser import HTMLParser


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def fetch_subjects(url: str, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": "SyntheticGaze/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    parser = LinkParser()
    parser.feed(html)
    out = []
    for href in parser.hrefs:
        m = re.search(r"(subject\d+)\.csv$", href.lower())
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Create subject shard files from remote annotation listing.")
    ap.add_argument(
        "--base-url",
        default="https://dataset.ait.ethz.ch/downloads/T3fODqLSS1/eth-xgaze/raw/data/",
    )
    ap.add_argument("--annotation-subdir", default="annotation_train")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=40)
    ap.add_argument(
        "--out-dir",
        default=os.path.join("ETH-GAZE DATASET", "processed", "subject_shards_train"),
    )
    args = ap.parse_args()

    root = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    ann_url = root + args.annotation_subdir.strip("/") + "/"
    subjects = fetch_subjects(ann_url, timeout=args.timeout)
    if not subjects:
        raise SystemExit(f"No subject*.csv found at {ann_url}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "all_subjects.txt"), "w", encoding="utf-8") as f:
        for s in subjects:
            f.write(s + "\n")

    shards = [[] for _ in range(max(1, args.num_shards))]
    for idx, subject in enumerate(subjects):
        shards[idx % len(shards)].append(subject)

    for i, shard in enumerate(shards):
        shard_path = os.path.join(args.out_dir, f"shard_{i:02d}.txt")
        with open(shard_path, "w", encoding="utf-8") as f:
            for s in shard:
                f.write(s + "\n")

    print(f"Found {len(subjects)} subjects. Wrote {len(shards)} shard files to: {args.out_dir}")


if __name__ == "__main__":
    main()


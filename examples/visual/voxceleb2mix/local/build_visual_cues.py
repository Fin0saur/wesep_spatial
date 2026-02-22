import json
import re
from pathlib import Path
import argparse


def parse_two_speakers_from_key(key):
    parts = key.split("_")
    speakers = []
    i = 0

    while i < len(parts):
        if re.match(r"id\d{5}", parts[i]):

            spk_id = parts[i]
            j = i + 1

            # 找 5 位数字的 segment id
            seg_index = None
            while j < len(parts):
                if re.fullmatch(r"\d{5}", parts[j]):
                    seg_index = j
                    break
                j += 1

            if seg_index is None:
                raise ValueError(f"Cannot find seg_id in key: {key}")

            yt_id = "_".join(parts[i + 1:seg_index])
            seg_id = parts[seg_index]

            speakers.append((spk_id, yt_id, seg_id))

            i = seg_index + 1
        else:
            i += 1

    if len(speakers) != 2:
        raise ValueError(f"Failed parsing two speakers: {key}")

    return speakers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--mp4_root", required=True)
    args = parser.parse_args()

    mp4_root = Path(args.mp4_root)

    visual_index = {}

    with open(args.samples_jsonl) as f:
        for line in f:
            sample = json.loads(line)
            key = sample["key"]

            speakers = parse_two_speakers_from_key(key)

            for spk_id, yt_id, seg_id in speakers:

                mp4_path = (mp4_root / spk_id / yt_id / f"{seg_id}.mp4")

                if not mp4_path.exists():
                    raise FileNotFoundError(
                        f"\nKey: {key}\nExpected file:\n{mp4_path}\n")

                mix_spk_key = f"{key}::{spk_id}"

                visual_index[mix_spk_key] = [{
                    "utt_id": f"{yt_id}_{seg_id}",
                    "path": str(mp4_path)
                }]

    with open(args.outfile, "w") as f:
        json.dump(visual_index, f, indent=2)


if __name__ == "__main__":
    main()

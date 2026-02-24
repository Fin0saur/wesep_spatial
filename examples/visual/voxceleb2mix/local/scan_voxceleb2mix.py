import json
import re
import argparse
from pathlib import Path


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

            speakers.append(spk_id)

            i = seg_index + 1
        else:
            i += 1

    if len(speakers) != 2:
        raise ValueError(f"Failed parsing two speakers: {key}")

    return speakers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mix_dir")
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    mix_dir = Path(args.mix_dir)
    s1_dir = mix_dir.parent / "s1"
    s2_dir = mix_dir.parent / "s2"

    with open(args.outfile, "w") as fout:

        for wav_path in sorted(mix_dir.glob("*.wav")):
            key = wav_path.stem
            spk_ids = parse_two_speakers_from_key(key)

            if len(spk_ids) != 2:
                print(f"Warning: skip {key}, cannot parse speakers")
                continue

            s1_path = s1_dir / wav_path.name
            s2_path = s2_dir / wav_path.name

            sample = {
                "key": key,
                "spk": spk_ids,
                "mix": {
                    "default": [str(wav_path)]
                },
                "src": {
                    spk_ids[0]: [str(s1_path)],
                    spk_ids[1]: [str(s2_path)]
                }
            }

            fout.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()

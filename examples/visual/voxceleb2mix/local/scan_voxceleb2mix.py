import json
import argparse
from pathlib import Path


def parse_speakers_from_key(key):
    """
    Extract speaker IDs from filename.
    Example:
    train_train_id00015_XXX_train_id00187_YYY_-4.2_4.2
    -> ['id00015', 'id00187']
    """
    parts = key.split('_')

    spk_ids = []
    for p in parts:
        if p.startswith("id"):
            spk_ids.append(p)

    # 只取前两个（防止后面还有奇怪字段）
    return spk_ids[:2]


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
            spk_ids = parse_speakers_from_key(key)

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

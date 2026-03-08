#!/usr/bin/env python3
# local/drop_cues.py
import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_json', type=str, required=True, help="Input full JSON file")
    parser.add_argument('--out_json', type=str, required=True, help="Output dropped JSON file")
    parser.add_argument('--keep_ratio', type=float, default=0.6, help="Ratio of keys to keep (0.0 to 1.0)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.in_json, 'r') as f:
        data = json.load(f)

    keys = list(data.keys())
    keys.sort() 
    random.shuffle(keys)

    keep_count = int(len(keys) * args.keep_ratio)
    keep_keys = set(keys[:keep_count])

    filtered_data = {k: v for k, v in data.items() if k in keep_keys}

    with open(args.out_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"[{args.in_json}] Total Keys: {len(keys)} -> Kept: {len(filtered_data)} (Ratio: {args.keep_ratio}, Seed: {args.seed})")

if __name__ == "__main__":
    main()
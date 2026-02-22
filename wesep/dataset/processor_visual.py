# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torchvision.io import read_video

from wesep.utils.file_utils import load_json


def _build_lookup_key(sample, spk_slot, key_field):
    """
    Build lookup key for speaker cue resource.

    key_field semantics:
      - "spk_id"       -> use sample[spk_slot]
      - "mix_spk_id"   -> use f"{sample['key']}::{sample[spk_slot]}"
    """
    if key_field == "spk_id":
        return sample[spk_slot]

    elif key_field == "mix_spk_id":
        mix_key = sample.get("key", None)
        if mix_key is None:
            raise KeyError("sample missing 'key' for mix_spk_id cue")
        return f"{mix_key}::{sample[spk_slot]}"

    else:
        raise ValueError(f"Unsupported key_field for speaker cue: {key_field}")


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def _get_spk_resource(resource_path):
    """
    Lazy-load and cache speaker cue resources.

    Cache is keyed by resource_path to avoid train/val or
    multi-dataset cross-contamination.
    """
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    return _SPK_RESOURCE_CACHE[resource_path]


def sample_fixed_visual_cue(
    data,
    resource_path,
    key_field,
    scope="speaker",
    required=True,
):
    if scope not in ("speaker", "utterance"):
        raise ValueError(f"Unsupported scope: {scope}")

    spk_resource = _get_spk_resource(resource_path)

    for sample in data:

        spk_slots = [k for k in sample.keys() if k.startswith("spk")]

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        if scope == "utterance":
            spk_slots = [spk_slots[0]]

        for slot in spk_slots:
            lookup_key = _build_lookup_key(sample, slot, key_field)

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(f"fixed visual cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(f"empty fixed visual cue: {lookup_key}")
                continue

            enroll_item = items[0]
            video_path = enroll_item["path"]

            try:
                video, _, info = read_video(video_path, pts_unit="sec")
                fps = info["video_fps"]
            except Exception as e:
                logging.warning(f"Failed to read video: {video_path}, err={e}")
                if required:
                    raise
                continue

            # video: [T, H, W, C] uint8
            if video.numel() == 0:
                if required:
                    raise RuntimeError(f"Empty video: {video_path}")
                continue

            # Align video with audio length by truncation or repetition
            if "chunk_ratio" in sample:
                audio_sec = sample["chunk_ratio"]["orig_len"] / sample[
                    "sample_rate"]
            else:
                audio_sec = len(sample["wav_mix"]) / sample["sample_rate"]
            video_sec = video.shape[0] / fps
            if video_sec < audio_sec:
                need_sec = audio_sec - video_sec
                need_frames = int(round(need_sec * fps))

                last_frame = video[-1:].repeat(need_frames, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
            elif video_sec > audio_sec:
                max_frames = int(round(audio_sec * fps))
                video = video[:max_frames]

            # Apply chunk-level cropping if specified
            if "chunk_ratio" in sample:
                ratio = sample["chunk_ratio"]
                start_ratio = ratio["start_ratio"]
                end_ratio = ratio["end_ratio"]

                F = video.shape[0]

                start_f = int(start_ratio * F)
                end_f = int(end_ratio * F)

                start_f = max(0, min(start_f, F))
                end_f = max(start_f + 1, min(end_f, F))

                video = video[start_f:end_f]

            # Normalize to [0, 1] float
            video = video.float() / 255.0

            # Convert to [H, W, C, T]
            video = video.permute(1, 2, 3, 0)

            sample[f"visual_{slot}"] = video

        # utterance-level cue: copy from spk1 to all spk slots
        if scope == "utterance":
            emb = sample[f"visual_{spk_slots[0]}"]
            for slot in [k for k in sample.keys() if k.startswith("spk")]:
                sample[f"visual_{slot}"] = emb
        yield sample

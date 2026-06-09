from __future__ import print_function

import os
import time

import fire
import soundfile
import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    tse_collate_fn,
    AUX_KEY_MAP,
)
import numpy as np
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)
from wesep.utils.file_utils import load_yaml

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0
    accept_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get(
        "save_wav", True)  # Control if save the extracted speech as .wav

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], "infer.log")
    logger.info("Load checkpoint from {}".format(model_path))
    save_audio_dir = os.path.join(configs["exp_dir"], "audio")
    if sign_save_wav:
        if not os.path.exists(save_audio_dir):
            try:
                os.makedirs(save_audio_dir)
                print(f"Directory {save_audio_dir} created successfully.")
            except OSError as e:
                print(f"Error creating directory {save_audio_dir}: {e}")
        else:
            print(f"Directory {save_audio_dir} already exists.")
    else:
        print("Do NOT save the results in wav.")

    model = model.to(device)
    model.eval()

    configs["dataset_args"]["whole_utt"] = True
    test_dataset = Dataset(
        configs["data_type"],
        configs["test_data"],
        configs["dataset_args"],
        state="test",
        repeat_dataset=configs.get("repeat_dataset", False),
        cues_yaml=configs.get("test_cues", None),
    )
    test_collect_keys = build_collect_keys(
        load_yaml(configs["test_cues"]),
        configs["dataset_args"],
        BASE_COLLECT_KEYS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=lambda batch: tse_collate_fn(batch, test_collect_keys))

    with open(configs["test_data"], "r", encoding="utf-8") as f:
        test_iter = sum(1 for _ in f)
    logger.info("test number: {}".format(test_iter))

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            mix, cues, target = extract_model_inputs(batch, device)
            spk = batch["spk"]
            key = batch["key"]

            if cues is None:
                outputs = model(mix)
            else:
                outputs = model(mix, cues)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            reduce_dims = tuple(range(1, outputs.dim()))
            if torch.min(outputs.amax(dim=reduce_dims)) > 0:
                peak = outputs.abs().amax(dim=reduce_dims, keepdim=True)
                outputs = ((outputs / peak.clamp_min(1e-8) * 0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            ests = np.squeeze(outputs, axis=1) if outputs.ndim == 3 and outputs.shape[1] == 1 else outputs
            ref = target.cpu().numpy()
            ref = np.squeeze(ref, axis=1) if ref.ndim == 3 and ref.shape[1] == 1 else ref
            mix = mix.cpu().numpy()
            mix_ref = mix[:, 0, :] if mix.ndim == 3 else mix

            min_len = min(ref.shape[-1], ests.shape[-1], mix_ref.shape[-1])
            ref = ref[..., :min_len]
            ests = ests[..., :min_len]
            mix_ref = mix_ref[..., :min_len]

            for idx in range(ests.shape[0]):
                if sign_save_wav:
                    wav_path = os.path.join(
                        save_audio_dir,
                        f"Utt{total_cnt + 1}-{key[idx]}-T{spk[idx]}.wav",
                    )
                    soundfile.write(wav_path, np.squeeze(ests[idx]), sample_rate)

                sisnr, delta = cal_SISNRi(
                    np.squeeze(ests[idx]),
                    np.squeeze(ref[idx]),
                    np.squeeze(mix_ref[idx]),
                )
                logger.info(
                    "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f}"
                    .format(total_cnt + 1, key[idx], spk[idx], sisnr, delta))
                total_SISNR += sisnr
                total_SISNRi += delta
                total_cnt += 1
                if delta > 1:
                    accept_cnt += 1

        end = time.time()
    # generate the scp file of the enhanced speech for scoring
    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info(
        "Acceptance rate of Utterances with SI-SDRi > 1 dB: {:.2f}".format(
            accept_cnt / total_cnt * 100))


def extract_model_inputs(batch, device):
    """
        Build model inputs from collated batch.

        Args:
            batch: dict from tse_collate_fn
            device: torch.device

        Returns:
            mix:    Tensor [B, 1, T]
            cues:   list[Tensor] or None
            target: Tensor [B, 1, T]
        """
    if "wav_mix" not in batch:
        raise RuntimeError("[executor] Missing required key: wav_mix")
    if "wav_target" not in batch:
        raise RuntimeError("[executor] Missing required key: wav_target")

    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)

    cues = []
    for k in list(AUX_KEY_MAP.values()):
        if k in batch and batch[k] is not None:
            cues.append(batch[k].float().to(device))

    if len(cues) == 0:
        cues = None

    return mix, cues, target


if __name__ == "__main__":
    fire.Fire(infer)

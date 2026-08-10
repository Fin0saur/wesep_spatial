import numpy as np
from scipy import signal
import os
import soundfile as sf
import argparse
import bisect
import gpuRIR
import glob
import random
import yaml
import sys

class DataGenerator:
    def __init__(self, config_path, stage='all'):
        """
        :param config_path: config.yaml 
        :param stage: 'train', 'val', 'test', or 'all'
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        self.stage = stage
        self.sr = self.cfg['audio']['sr']
        self.duration = self.cfg['audio']['duration']
        
        self.mic_pos_template = self._init_mic_array()
        
        self.libri_path = self.cfg['dataset']['paths']['librispeech']
        self.noise_path = self.cfg['dataset']['paths']['noise']
        self.output_base = self.cfg['dataset']['paths']['output']
        
        # self.spk_map, self.spk_list = self._get_files_per_speaker()
        self.noise_list = self._get_noise_files()

    def _init_mic_array(self):
        coords_list = self.cfg['array']['mic_coords']
        mic_pos = np.array(coords_list, dtype=np.float32)
        
        print(f"Loaded microphone array with {mic_pos.shape[0]} mics.")
        self.array_radius = np.max(np.linalg.norm(mic_pos, axis=1))
        print(f"Array Radius (Max distance from center): {self.array_radius:.4f} m")
        
        return mic_pos

    def _get_files_per_speaker(self,sub_folder):
        
        current_libri_path=os.path.join(self.libri_path,sub_folder)
        
        print(f"Indexing LibriSpeech from {current_libri_path}...")
        speaker_map = {}
        extensions = ['*.flac', '*.wav']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(current_libri_path, '**', ext), recursive=True))
        
        if not all_files:
            raise RuntimeError(f"No audio files found in {current_libri_path}")

        for f in all_files:
            filename = os.path.basename(f)  
            spk_id = filename.split('-')[0] 
            
            if spk_id not in speaker_map:
                speaker_map[spk_id] = []
            speaker_map[spk_id].append(f)
        
        valid_speakers = [k for k, v in speaker_map.items() if len(v) >= 2]
        print(f"Found {len(valid_speakers)} valid speakers.")
        return speaker_map, valid_speakers

    def _get_noise_files(self):
        print(f"Indexing Noise from {self.noise_path}...")
        files = glob.glob(os.path.join(self.noise_path, '**', '*.wav'), recursive=True)
        if not files:
            print(f"Warning: No noise files found in {self.noise_path}")
        else:
            print(f"Found {len(files)} noise files.")
        return files

    def _generate_room_config(self):
        r_cfg = self.cfg['room']
        dim_min, dim_max = r_cfg['dimensions']['min'], r_cfg['dimensions']['max']
        rt_min, rt_max = r_cfg['rt60']['min'], r_cfg['rt60']['max']
        
        length = np.random.uniform(dim_min[0], dim_max[0])
        width  = np.random.uniform(dim_min[1], dim_max[1])
        height = np.random.uniform(dim_min[2], dim_max[2])
        room_sz = np.array([length, width, height], dtype=np.float32)
        
        rt60 = np.random.uniform(rt_min, rt_max)
        
        margin = 0.5 + self.array_radius
        
        if room_sz[0] < 2*margin: room_sz[0] = 2*margin + 0.1
        if room_sz[1] < 2*margin: room_sz[1] = 2*margin + 0.1
        if room_sz[2] < 2*margin: room_sz[2] = 2*margin + 0.1 

        mic_center = np.array([
            np.random.uniform(margin, room_sz[0]-margin),
            np.random.uniform(margin, room_sz[1]-margin),
            np.random.uniform(margin, room_sz[2]-margin)
        ], dtype=np.float32)
        
        mic_pos = self.mic_pos_template + mic_center
        
        spk_pos = []
        max_attempts = r_cfg['simulation']['max_attempts']
        dist_min, dist_max = r_cfg['simulation']['dist_range']
        
        for _ in range(2): 
            for _ in range(max_attempts):
                pos = np.array([
                    np.random.uniform(0.5, room_sz[0]-0.5),
                    np.random.uniform(0.5, room_sz[1]-0.5),
                    np.random.uniform(0.5, room_sz[2]-0.5)
                ], dtype=np.float32)
                
                dist = np.linalg.norm(pos - mic_center)
                if dist_min < dist < dist_max: 
                    spk_pos.append(pos)
                    break
            else:
                fallback = mic_center + np.array([1.0, 0, 0], dtype=np.float32)
                fallback = np.clip(fallback, [0.5, 0.5, 0.5], room_sz - 0.5)
                spk_pos.append(fallback)
            
        noise_pos = np.array([
            np.random.uniform(0.5, room_sz[0]-0.5),
            np.random.uniform(0.5, room_sz[1]-0.5),
            np.random.uniform(0.5, room_sz[2]-0.5)
        ], dtype=np.float32)
        
        return room_sz, rt60, mic_pos, mic_center, np.array(spk_pos), np.array([noise_pos])

    @staticmethod
    def _calculate_angles(mic_center, target_pos):
        rel_pos = target_pos - mic_center
        x, y, z = rel_pos[0], rel_pos[1], rel_pos[2]
        azimuth = np.arctan2(y, x)
        xy_dist = np.sqrt(x**2 + y**2)
        elevation = np.arctan2(z, xy_dist)
        return azimuth, elevation

    @staticmethod
    def _read_audio(file_path, target_len_samples, is_noise=False):
        audio, fs = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
            
        if len(audio) < target_len_samples:
            if is_noise:
                repeats = (target_len_samples // len(audio)) + 1
                audio = np.tile(audio, repeats)[:target_len_samples]
            else:
                padding = np.zeros(target_len_samples - len(audio))
                audio = np.concatenate([audio, padding])
        else:
            start = np.random.randint(0, len(audio) - target_len_samples + 1)
            audio = audio[start : start + target_len_samples]
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    @staticmethod
    def _pad_sig(x, target_len):
        curr_len = x.shape[0]
        if curr_len < target_len:
            zeros = np.zeros((target_len - curr_len, x.shape[1]))
            return np.concatenate([x, zeros], axis=0)
        else:
            return x[:target_len, :]

    @staticmethod
    def _get_file_id(path):
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def _sample_unique_source_pairs(speaker_map, speakers, count):
        """Sample unordered cross-speaker utterance pairs without replacement.

        The candidate space is represented as contiguous speaker-pair blocks,
        so only ``count`` integer indices are sampled instead of materializing
        every possible utterance pair.  The source order is then randomized to
        decide which utterance is the target and which is the interferer.
        """
        files_by_speaker = {
            speaker: speaker_map[speaker]
            for speaker in speakers
        }
        block_ends = []
        blocks = []
        total_pairs = 0
        for speaker_index, first_speaker in enumerate(speakers):
            first_files = files_by_speaker[first_speaker]
            for second_speaker in speakers[speaker_index + 1:]:
                second_files = files_by_speaker[second_speaker]
                total_pairs += len(first_files) * len(second_files)
                block_ends.append(total_pairs)
                blocks.append((first_files, second_files))

        if count > total_pairs:
            raise ValueError(
                f"Requested {count} mixtures, but only {total_pairs} unique "
                "cross-speaker utterance pairs are available."
            )

        sampled_pairs = []
        for pair_index in random.sample(range(total_pairs), count):
            block_index = bisect.bisect_right(block_ends, pair_index)
            block_start = 0 if block_index == 0 else block_ends[block_index - 1]
            local_index = pair_index - block_start
            first_files, second_files = blocks[block_index]
            second_count = len(second_files)
            first_file = first_files[local_index // second_count]
            second_file = second_files[local_index % second_count]

            if random.random() < 0.5:
                sampled_pairs.append((first_file, second_file))
            else:
                sampled_pairs.append((second_file, first_file))

        return sampled_pairs, total_pairs

    @staticmethod
    def _get_output_ids(dirs):
        """Collect generated IDs for every output modality."""
        patterns = {
            'mix': '*.wav',
            's1': '*.wav',
            's2': '*.wav',
            'spatial': '*.npy',
        }
        output_ids = {
            name: {
                os.path.splitext(os.path.basename(path))[0]
                for path in glob.glob(os.path.join(dirs[name], pattern))
            }
            for name, pattern in patterns.items()
        }

        return output_ids

    @classmethod
    def _require_empty_output_dirs(cls, dirs):
        """Require a clean destination for a new deterministic-size run."""
        output_ids = cls._get_output_ids(dirs)
        counts = {name: len(ids) for name, ids in output_ids.items()}
        if any(counts.values()):
            raise RuntimeError(
                "Spatial synthesis output is not empty: "
                f"counts={counts}. Choose a new dataset.paths.output or "
                "remove the previous generated split before starting a new "
                "synthesis run. Existing data is never resumed or overwritten."
            )

    @classmethod
    def _validate_output_count(cls, dirs, expected_count):
        """Verify exact count and one-to-one keys across all modalities."""
        output_ids = cls._get_output_ids(dirs)
        reference_ids = output_ids['mix']
        if any(ids != reference_ids for ids in output_ids.values()):
            counts = {name: len(ids) for name, ids in output_ids.items()}
            raise RuntimeError(
                "Generated spatial dataset has inconsistent modality keys: "
                f"counts={counts}."
            )
        actual_count = len(reference_ids)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Spatial synthesis generated {actual_count} complete samples; "
                f"expected exactly {expected_count}."
            )

    def run(self):
        full_len_samples = int(self.duration * self.sr)
        num_mic = self.mic_pos_template.shape[0]
        
        dataset_counts = self.cfg['dataset']['counts']
        dataset_splits = self.cfg['dataset']['splits'] 
                
        mix_cfg = self.cfg['mixing']
        
        if self.stage == 'all':
            target_sets = dataset_counts.keys()
        else:
            target_sets = [self.stage]
            
        for data_type in target_sets:
            subsets = dataset_splits[data_type]
            current_spk_map, current_spk_list = self._get_files_per_speaker(subsets)
            if data_type not in dataset_counts:
                continue
                
            count = dataset_counts[data_type]

            source_pairs, total_unique_pairs = self._sample_unique_source_pairs(
                current_spk_map,
                current_spk_list,
                count,
            )
            print(
                f"Sampled {count} unique source pairs from "
                f"{total_unique_pairs} available cross-speaker pairs."
            )
            
            folder_name = None
            if data_type == "train":
                folder_name="train-100"
            else:
                folder_name=data_type
            
            print(f"Generating {data_type} set (saving to '{folder_name}'): {count} utterances...")
            
            base_dir = os.path.join(self.output_base, folder_name)
            dirs = {
                'mix': os.path.join(base_dir, 'mix_clean'),
                's1': os.path.join(base_dir, 's1'),
                's2': os.path.join(base_dir, 's2'),
                'spatial': os.path.join(base_dir, 'spatial')
            }
            for p in dirs.values():
                os.makedirs(p, exist_ok=True)

            self._require_empty_output_dirs(dirs)

            output_ids = set()
            for idx, (target_src_file, interf_file) in enumerate(source_pairs):
                output_filename = (
                    f"{self._get_file_id(target_src_file)}_"
                    f"{self._get_file_id(interf_file)}"
                )
                if output_filename in output_ids:
                    raise RuntimeError(
                        "Duplicate LibriSpeech utterance IDs produced the same "
                        f"mixture key: {output_filename}"
                    )
                output_ids.add(output_filename)
                    
                if not self.noise_list:
                    noise_audio = np.zeros(full_len_samples) 
                else:
                    noise_file = random.choice(self.noise_list)
                    noise_audio = self._read_audio(noise_file, full_len_samples, is_noise=True)
                    
                ov_min, ov_max = mix_cfg['overlap_ratio_range']
                overlap_ratio = np.random.uniform(ov_min, ov_max)
                    
                actual_len = int(full_len_samples / (2 - overlap_ratio))
                pad_length = int((1 - overlap_ratio) * actual_len)
                    
                src_audio = self._read_audio(target_src_file, actual_len)
                interf_audio = self._read_audio(interf_file, actual_len)
                # --- 3. RIR ---
                room_sz, rt60, mic_pos, mic_center, spk_pos, noise_pos_coord = self._generate_room_config()
                    
                beta = gpuRIR.beta_SabineEstimation(room_sz, rt60)
                nb_img = gpuRIR.t2n(rt60, room_sz)
                    
                rir_speech = gpuRIR.simulateRIR(room_sz, beta, spk_pos, mic_pos, nb_img, rt60, self.sr)
                rir_noise = gpuRIR.simulateRIR(room_sz, beta, noise_pos_coord, mic_pos, nb_img, rt60, self.sr)
                    
                # --- 4. (Simulation) ---
                spk1_echoic_raw = signal.fftconvolve(src_audio[:, None], rir_speech[0].T, mode='full', axes=0)
                spk2_echoic_raw = signal.fftconvolve(interf_audio[:, None], rir_speech[1].T, mode='full', axes=0)
                noise_echoic_raw = signal.fftconvolve(noise_audio[:, None], rir_noise[0].T, mode='full', axes=0)
                    
                # --- 5. (Alignment) ---
                # Spk1 Pad: [Raw, Zeros]
                # Spk2 Pad: [Zeros, Raw]
                zeros_pad = np.zeros((pad_length, num_mic))
                spk1_aligned = np.concatenate([spk1_echoic_raw, zeros_pad], axis=0) 
                spk2_aligned = np.concatenate([zeros_pad, spk2_echoic_raw], axis=0)
                    
                target_echoic = self._pad_sig(spk1_aligned, full_len_samples)
                interf_echoic = self._pad_sig(spk2_aligned, full_len_samples)
                noise_echoic  = self._pad_sig(noise_echoic_raw, full_len_samples)
                    
                # --- 6.(Mixing & Scaling) ---
                target_power = np.mean(target_echoic**2) + 1e-12
                    
                sinr_min, sinr_max = mix_cfg['target_sinr_range']
                target_sinr_db = np.random.uniform(sinr_min, sinr_max)
                desired_noise_total_power = target_power / (10**(target_sinr_db / 10.0))
                    
                u_min, u_max = mix_cfg['interf_ratio_range']
                u = np.random.uniform(u_min, u_max)
                desired_interf_power = desired_noise_total_power * u
                desired_bg_noise_power = desired_noise_total_power * (1 - u)
                    
                curr_interf_power = np.mean(interf_echoic**2) + 1e-12
                curr_bg_noise_power = np.mean(noise_echoic**2) + 1e-12
                    
                interf_scale = np.sqrt(desired_interf_power / curr_interf_power)
                bg_noise_scale = np.sqrt(desired_bg_noise_power / curr_bg_noise_power)
                    
                interf_echoic *= interf_scale
                noise_echoic *= bg_noise_scale
                    
                mixture = target_echoic + interf_echoic + noise_echoic
                    
                sf.write(os.path.join(dirs['mix'], output_filename +'.wav'), mixture, self.sr)
                sf.write(os.path.join(dirs['s1'], output_filename +'.wav'), target_echoic, self.sr)
                sf.write(os.path.join(dirs['s2'], output_filename +'.wav'), interf_echoic, self.sr)
                    
                az1, el1 = self._calculate_angles(mic_center, spk_pos[0])
                az2, el2 = self._calculate_angles(mic_center, spk_pos[1])
                    
                final_p_target = np.mean(target_echoic**2) + 1e-12
                final_p_interf = np.mean(interf_echoic**2) + 1e-12
                final_p_noise  = np.mean(noise_echoic**2) + 1e-12
                    
                real_sir = 10 * np.log10(final_p_target / final_p_interf)
                real_snr = 10 * np.log10(final_p_target / final_p_noise)
                    
                meta_data = {
                    "azimuth_spk1": np.float32(az1),
                    "elevation_spk1": np.float32(el1),
                    "azimuth_spk2": np.float32(az2),
                    "elevation_spk2": np.float32(el2),
                    "sir_db": np.float32(real_sir),
                    "snr_db": np.float32(real_snr),
                    "pos_mic_center": mic_center.astype(np.float32),
                    "pos_spk1": spk_pos[0].astype(np.float32),
                    "pos_spk2": spk_pos[1].astype(np.float32),
                    "room_sz": room_sz.astype(np.float32),
                    "rt60": np.float32(rt60)
                }
                np.save(os.path.join(dirs['spatial'], output_filename + '.npy'), meta_data)
                    
                completed = idx + 1
                if completed % 100 == 0 or completed == count:
                    print(f"  [{data_type}] Generated {completed} / {count}")

            self._validate_output_count(dirs, count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multi-channel Libri2Mix-style data with Config')
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file.")
    parser.add_argument('--stage', default='all', choices=['train', 'dev', 'test', 'all'], help="Override stage.")

    args = parser.parse_args()
    
    generator = DataGenerator(args.config, args.stage)
    generator.run()

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import os
import json
import random
import resampy
import soundfile as sf
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import soundfile as sf
from moisesdb.track import pad_and_mix

class CassetteDBDataModule(LightningDataModule):

    def __init__(
        self,
        cache_dir: str,
        data_dir: str = "/mnt/disks/training-data-refine/fullTracks/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        time_in_seconds: int = 30,
        stereo_to_mono: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        volume_normalization: bool = True,
        average: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = CassetteDBInstructDataset(data_path=data_dir,
                                                 sample_rate=32000,
                                                 time_in_seconds=time_in_seconds,
                                                 cache_dir=cache_dir,
                                                 stereo_to_mono=stereo_to_mono,
                                                 volume_normalization=volume_normalization,
                                                 average=average)

        self.save_hyperparameters(logger=False)

        train_val_test_split_ratio = [0.8, 0.1, 0.1]
        dataset_size = len(self.dataset)
        train_size = int(train_val_test_split_ratio[0] * dataset_size)
        val_size = int(train_val_test_split_ratio[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.dataset,
            lengths=[train_size, val_size, test_size],
        )

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


def stereo_to_mono(audio_data):
    if audio_data.ndim > 1 and audio_data.shape[1] == 2:  # Check if the audio is stereo
        return np.mean(audio_data, axis=1, keepdims=True)  # Average the channels to create a mono signal with shape (N, 1)
    return audio_data[:, None] if audio_data.ndim == 1 else audio_data  # Ensure mono audio has shape (N, 1)
    
class CassetteDBInstructDataset(Dataset):
    def __init__(self, data_path: str,
                 sample_rate: int,
                 cache_dir: str,
                 time_in_seconds: int = 30,
                 stereo_to_mono: bool = True,
                 volume_normalization: bool = True,
                 average: bool = False):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.files = [f for f in os.listdir(data_path) if f.endswith('.wav') and '-beat' not in f]
        self.time_in_seconds = time_in_seconds
        self.stereo_to_mono = stereo_to_mono
        self.cache_dir = cache_dir
        self.instruct_set = ['add']  # Only 'add' instruction as specified
        self.volume_normalization = volume_normalization
        self.average = average
        print(f"Total possible files: {len(self.files)}")

        self.files = self._filter_files()
        print(f"Total valid files with all paired data: {len(self.files)}")

    def _filter_files(self):
        valid_files = []
        num_samples = self.sample_rate * self.time_in_seconds
        for f in tqdm(self.files, desc="Filtering files"):
            file_base = os.path.splitext(f)[0]
            file_id = file_base.split('_')[0]
            file_chunk = file_base.split('_')[1]
            beat_file = os.path.join(self.data_path, f"{file_id}-beat_{file_chunk}.wav")
            full_file = os.path.join(self.data_path, f"{file_base}.wav")
            json_file = os.path.join(self.data_path, f"{file_base}.json")

            if os.path.exists(beat_file) and os.path.exists(full_file) and os.path.exists(json_file):
                # Check length of the files using sf.info for faster access
                beat_info = sf.info(beat_file)
                full_info = sf.info(full_file)
                if beat_info.frames >= num_samples and full_info.frames >= num_samples:
                    valid_files.append(f)
        return valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, retry_count=0, assigned_instruction='add', return_json=False):
        file_name = self.files[idx]
        file_base = os.path.splitext(file_name)[0]
        file_id = file_base.split('_')[0]
        file_chunk = file_base.split('_')[1]
        beat_file = os.path.join(self.data_path, f"{file_id}-beat_{file_chunk}.wav")
        full_file = os.path.join(self.data_path, f"{file_base}.wav")
        json_file = os.path.join(self.data_path, f"{file_base}.json")

        # Read the audio data and sample rate
        vocals_data, vocals_sr = sf.read(full_file)
        beat_data, beat_sr = sf.read(beat_file)

        # Convert to mono if necessary
        vocals_data = stereo_to_mono(vocals_data)
        beat_data = stereo_to_mono(beat_data)

        # Log detailed information about the data
        print(f"{file_name}.wav shape: {vocals_data.shape}, length: {vocals_data.shape[0]}, Sample rate: {vocals_sr}")
        print(f"{file_id}-beat_{file_chunk}.wav shape: {beat_data.shape}, length: {beat_data.shape[0]}, Sample rate: {beat_sr}")

        # Ensure both sample rates are the same, else raise an error
        if vocals_sr != beat_sr:
            raise ValueError(f"Sample rates of vocals ({vocals_sr}) and beat ({beat_sr}) do not match for file {file_name}")

        raw_data = {
            "stems": {
                "vocals": vocals_data,
                "beat": beat_data,
            },
            "sr": vocals_sr
        }
        cache_file = os.path.join(self.cache_dir, f"{idx}.pt")

        if os.path.exists(cache_file):
            stems = torch.load(cache_file)  # stems = {'vocals': ..., 'bass': ...}
        else:
            stems = raw_data['stems']
            torch.save(stems, cache_file)

        instruction = random.choice(self.instruct_set)
        target_stem_key = "vocals"
        target_stem_mix = stems[target_stem_key]

        input_stems_mix, output_stems_mix, instruction_text = None, None, None

        if assigned_instruction is not None:
            instruction = assigned_instruction

        if instruction == 'add':
            input_stems_keys = [stem for stem in stems.keys() if stem != target_stem_key]
            output_stems_keys = [target_stem_key] + input_stems_keys
            input_stems_mix = pad_and_mix([stems[stem] for stem in input_stems_keys])
            output_stems_mix = pad_and_mix([stems[stem] for stem in output_stems_keys])

            with open(json_file, 'r') as f:
                json_data = json.load(f)
                lyrics = json_data['lyrics']

            instruction_text = f"Music piece. Instruction: Add {target_stem_key} with lyrics: {lyrics}"

        if raw_data['sr'] != self.sample_rate:
            input_stems_mix = resampy.resample(input_stems_mix, raw_data['sr'], self.sample_rate, filter='kaiser_fast')
            output_stems_mix = resampy.resample(output_stems_mix, raw_data['sr'], self.sample_rate, filter='kaiser_fast')

        if self.time_in_seconds is not None:
            num_samples = self.sample_rate * self.time_in_seconds
            min_length = min(input_stems_mix.shape[0], output_stems_mix.shape[0])
            if min_length < num_samples:
                raise ValueError(f"The minimum length of stems {min_length} is shorter than the required {num_samples} samples for file {file_name}")

            offset = random.randint(0, min_length - num_samples)
            input_stems_mix = input_stems_mix[offset:offset + num_samples]
            output_stems_mix = output_stems_mix[offset:offset + num_samples]

        # if (np.max(np.abs(input_stems_mix)) < 0.1
        #     or np.max(np.abs(output_stems_mix)) < 0.1
        #     or np.max(np.abs(target_stem_mix)) < 0.1
        # ):
        #     if retry_count < 10:
        #         return self.__getitem__(idx, retry_count + 1, assigned_instruction=instruction, return_json=return_json)
        #     else:
        #         print(f"Retry count exceeds 10 times for {idx}. File: {file_name}")
        #         pass

        if self.volume_normalization:
            if instruction == 'add':
                input_stems_mix *= (len(stems) / len(output_stems_keys))
                output_stems_mix *= (len(stems) / len(output_stems_keys))

            max_volume = max(np.max(np.abs(input_stems_mix)), np.max(np.abs(output_stems_mix)))
            if max_volume > 1.0:
                input_stems_mix /= max_volume
                output_stems_mix /= max_volume

        if return_json:
            instruction_json = {
                "instruction_text": instruction_text,
                "input_stems_list": input_stems_keys,
            }
            return input_stems_mix, output_stems_mix, instruction_json

        return input_stems_mix, output_stems_mix, instruction_text


    def get_test_files(self):
        root_folder = '/mnt/disks/training-data-refine/fullTracksTestData/'

        def process_file(idx, instruction):
            input_stems_mix, output_stems_mix, instruction_json = self.__getitem__(idx, assigned_instruction=instruction, return_json=True)

            instruction_text = instruction_json['instruction_text']
            input_stems_keys_text = instruction_json['input_stems_list']
            os.makedirs(os.path.join(root_folder, instruction, 'input'), exist_ok=True)
            os.makedirs(os.path.join(root_folder, instruction, 'ground_truth'), exist_ok=True)
            os.makedirs(os.path.join(root_folder, instruction, 'instruction'), exist_ok=True)
            sf.write(os.path.join(root_folder, instruction, 'input', f'{idx}.wav'), input_stems_mix, self.sample_rate)
            sf.write(os.path.join(root_folder, instruction, 'ground_truth', f'{idx}.wav'), output_stems_mix, self.sample_rate)

            print(f"Finish processing {root_folder}/{instruction}/{idx}.wav")

            with open(os.path.join(root_folder, instruction, 'instruction', f'{idx}.txt'), 'w') as f:
                f.write(f"Instruction: {instruction_text}\n")
                f.write(f"Stems: {input_stems_keys_text}\n")

            print(f"Finish processing {root_folder}/{instruction}/{idx}.wav")

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, idx, 'add') for idx in range(len(self))]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

if __name__ == "__main__":
    dataset = CassetteDBDataModule(data_dir='/mnt/disks/training-data-refine/fullTracks/')
    print(len(dataset.data_test))
    dataset.data_test.get_test_files()

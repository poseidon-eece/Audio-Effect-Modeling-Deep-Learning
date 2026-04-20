import os
import os.path
import math
import bisect
import numpy as np
import torch
import torch.utils.data
import librosa as lr


class PairedGRUDataset(torch.utils.data.Dataset):
    """
    Dataset that serves paired audio: (clean, processed) for GRU model.
    - Returns: (clean_float_input [B, L, 1], processed_float_target [B*L, 1])
    - Stores raw float audio in the .npz file.
    """
    def __init__(self,
                 dataset_file: str,
                 item_length: int,                 
                 clean_dir: str,
                 processed_dir: str,
                 target_length: int, # This is now the sequence length L
                 sampling_rate: int = 16000,
                 mono: bool = True,
                 normalize: bool = True, # Normalization is crucial for float models
                 dtype = np.float32, # Store as float32
                 train: bool = True,
                 test_stride: int = 100,
                 device=torch.device('cpu')):
        
        self.dataset_file = dataset_file
        self.device = device
        self._item_length = item_length # Will be used as sequence length L
        self._test_stride = test_stride
        self.target_length = target_length # Same as item_length for GRU
        self.train = train

        self.mono = mono
        self.normalize = normalize
        self.sampling_rate = sampling_rate
        self.dtype = dtype

        if not os.path.isfile(self.dataset_file):
            assert os.path.isdir(clean_dir), f"clean_dir not found: {clean_dir}"
            assert os.path.isdir(processed_dir), f"processed_dir not found: {processed_dir}"
            self._create_paired_npz(clean_dir, processed_dir, self.dataset_file)

        # load npz
        self.data = np.load(self.dataset_file, mmap_mode='r')

        # build index over processed arrays (we assume clean/processed have identical length per pair)
        self.proc_keys = sorted([k for k in self.data.files if k.startswith('proc_')], key=lambda x: int(x.split('_')[1]))
        self.clean_keys = sorted([k for k in self.data.files if k.startswith('clean_')], key=lambda x: int(x.split('_')[1]))
        assert len(self.proc_keys) == len(self.clean_keys) > 0, "Paired npz is empty or mismatched."

        # cumulative starts by sample index across all processed arrays
        self.start_samples = [0]
        for i in range(len(self.proc_keys)):
            self.start_samples.append(self.start_samples[-1] + len(self.data[self.proc_keys[i]]))

        # number of item start positions available
        available_length = self.start_samples[-1] - self._item_length
        self._length = max(0, math.floor(available_length / self.target_length))

    def _create_paired_npz(self, clean_dir: str, processed_dir: str, out_file: str):
        print(f"Creating paired GRU dataset from\n  clean_dir: {clean_dir}\n  processed_dir: {processed_dir}")
        
        def map_files(root):
            m = {}
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith((".wav", ".mp3", ".aif", ".aiff")):
                        m[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
            return m

        clean_map = map_files(clean_dir)
        proc_map = map_files(processed_dir)

        common = sorted(set(clean_map.keys()) & set(proc_map.keys()))
        if len(common) == 0:
            raise RuntimeError("No matching filenames between clean and processed dirs. Ensure same basenames.")

        arrays = {}
        for i, base in enumerate(common):
            if (i % 10) == 0:
                print(f"  processed {i}/{len(common)} files…")
            
            # Load raw float audio
            clean_y, _ = lr.load(clean_map[base], sr=self.sampling_rate, mono=self.mono)
            proc_y, _ = lr.load(proc_map[base], sr=self.sampling_rate, mono=self.mono)
            
            if self.normalize:
                # Normalize to [-1, 1] range
                clean_y = lr.util.normalize(clean_y)
                proc_y = lr.util.normalize(proc_y)
                
            # Ensure equal length (trim/pad to min length)
            L = min(len(clean_y), len(proc_y))
            clean_y = clean_y[:L].astype(self.dtype)
            proc_y = proc_y[:L].astype(self.dtype)

            # Store raw float audio
            arrays[f'clean_{i}'] = clean_y
            arrays[f'proc_{i}'] = proc_y

        np.savez(out_file, **arrays)
        print(f"Saved paired GRU npz: {out_file} with {len(common)} pairs")

    def set_item_length(self, l: int):
        self._item_length = l
        # recompute length
        available_length = self.start_samples[-1] - self._item_length
        self._length = max(0, math.floor(available_length / self.target_length))

    def __len__(self):
        test_length = math.floor(self._length / self._test_stride)
        return self._length - test_length if self.train else test_length

    def __getitem__(self, idx: int):
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride - 1))
        else:
            sample_index = self._test_stride * (idx + 1) - 1

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            raise IndexError(f"sample index {sample_index} out of range")

        pos_in_file = sample_index - self.start_samples[file_index]
        end_in_next = sample_index + self._item_length - self.start_samples[file_index + 1]

        proc_key = self.proc_keys[file_index]
        clean_key = self.clean_keys[file_index]

        if end_in_next < 0:
            proc_sample = self.data[proc_key][pos_in_file:pos_in_file + self._item_length]
            clean_sample = self.data[clean_key][pos_in_file:pos_in_file + self._item_length]
        else:
            # crosses boundary: stitch from two files
            proc_arr1 = self.data[self.proc_keys[file_index]]
            proc_arr2 = self.data[self.proc_keys[file_index + 1]]
            clean_arr1 = self.data[self.clean_keys[file_index]]
            clean_arr2 = self.data[self.clean_keys[file_index + 1]]

            proc_sample = np.concatenate((proc_arr1[pos_in_file:], proc_arr2[:end_in_next]))
            clean_sample = np.concatenate((clean_arr1[pos_in_file:], clean_arr2[:end_in_next]))
            
            proc_sample = proc_sample[:self._item_length]
            clean_sample = clean_sample[:self._item_length]

        # Tensors
        # clean_example shape: [L]
        clean_example = torch.from_numpy(clean_sample).float()
        proc_example = torch.from_numpy(proc_sample).float()
        # GRU Input: [L, 1] (Sequence Length, Features)
        gru_input = clean_example.unsqueeze(-1) 
        
        # GRU Target: [L, 1] (Sequence Length, Features) - Flattened to [L] for loss
        gru_target = proc_example.unsqueeze(-1)
        
        return gru_input, gru_target

import os
import random
import torch
import torch.utils.data
import librosa

# from speech_enhance.audio_zen.acoustics.feature import load_wav
# from speech_enhance.audio_zen.utils import expand_path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, noisy_dataset_dir, segment_size, 
                sr, split=True, shuffle=True, n_cache_reuse=1, device=None):
        clean_dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(clean_dataset)), "r")]
        # noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noisy_dataset), "r")]
        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset = noisy_dataset_dir
        self.length = len(self.clean_dataset_list)
        # random.seed(1234)
        # if shuffle:
        #     random.shuffle(self.audio_indexes)
        self.clean_dataset_list = clean_dataset_list
        # self.noise_dataset_list = noise_dataset_list
        self.segment_size = segment_size
        self.sr = sr
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        
    def __len__(self):
        return self.length

    def __getitem__(self, item):
        clean_file = self.clean_dataset_list[item]
        
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(clean_file, sr=self.sr)
            filename = os.path.basename(clean_file)
            noisy_audio, _ = librosa.load(os.path.join(self.noise_dataset, filename), sr=self.sr)
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[: length], noisy_audio[: length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start+self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        return (clean_audio.squeeze(), noisy_audio.squeeze())

def stft(y, n_fft, hop_length, win_length):
    """
    Args:
        y: [B, F, T]
        n_fft:
        hop_length:
        win_length:
        device:

    Returns:
        [B, F, T], **complex-valued** STFT coefficients

    """
    assert y.dim() == 2
    return torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(y.device),
        return_complex=True
    )
    
if __name__ == "__main__":
    dataset = Dataset(clean_dataset='train_data_fsn_voicebank_master/clean_train.txt', noisy_dataset_dir='data/voicebank/noisy_trainset_wav', segment_size=32000, sr=16000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for i, data in enumerate(dataloader):
        clean = data[0]
        noisy = data[1]
        clean_complex = stft(clean, 512, 256, 512)
        print(clean_complex.size())

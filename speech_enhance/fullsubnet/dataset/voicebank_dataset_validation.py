import os
import random
import torch
import torch.utils.data
import librosa


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # # Magnitude Compression
    # mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav


def get_dataset_filelist(input_training_file, input_validation_file):
    with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return training_indexes, validation_indexes


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_training_file, clean_wavs_dir, noisy_wavs_dir, segment_size, 
                sampling_rate, split=True, shuffle=True, n_cache_reuse=1, device=None):
        
        with open(input_training_file, 'r', encoding='utf-8') as fi:
            training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
        self.audio_indexes = training_indexes
        # with open(input_validation_file, 'r', encoding='utf-8') as fi:
        #     validation_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            noisy_audio, _ = librosa.load(os.path.join(self.noisy_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
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

        return noisy_audio.squeeze(), clean_audio.squeeze(), os.path.join(self.noisy_wavs_dir, filename + '.wav'), "No_reverb"

    def __len__(self):
        return len(self.audio_indexes)
    
if __name__ == "__main__":
    input_training_file = "train_data_fsn_voicebank_master/clean_train.txt"
    input_validation_file = "train_data_fsn_voicebank_master/clean_test.txt"
    # training_indexes, validation_indexes = get_dataset_filelist(input_training_file, input_validation_file)
    input_clean_wavs_dir = "data/voicebank/clean_trainset_wav"
    input_noisy_wavs_dir = "data/voicebank/noisy_trainset_wav"
    batch_size = 24
    num_workers = 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fft = 512
    hop_size = 256
    win_size = 512
    dataset = Dataset(input_training_file, input_clean_wavs_dir, input_noisy_wavs_dir, 32000, 16000)
    dataloader = torch.utils.data.DataLoader(dataset,num_workers=num_workers, batch_size=2, shuffle=True)
    for clean_audio, noisy_audio in dataloader:
        print(clean_audio.shape, noisy_audio.shape)
        clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
        noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
        one_labels = torch.ones(batch_size).to(device, non_blocking=True)

        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, n_fft, hop_size, win_size)
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, n_fft, hop_size, win_size)
        print(clean_mag.shape, clean_pha.shape, clean_com.shape)
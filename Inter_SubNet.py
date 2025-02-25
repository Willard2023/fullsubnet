import torch
from torch.nn import functional

from speech_enhance.audio_zen.acoustics.feature import drop_band
from speech_enhance.audio_zen.model.base_model import BaseModel
from speech_enhance.audio_zen.model.module.sequence_model import stacked_SIL_blocks_SequenceModel

# for log
from speech_enhance.utils.logger import log

print = log


class Inter_SubNet(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 sb_num_neighbors,
                 sb_output_activate_function,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
                 sbinter_middle_hidden_times=0.66,
                 ):
        """
        Inter-SubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        subband_input_size = (sb_num_neighbors * 2 + 1)
        self.sb_model = stacked_SIL_blocks_SequenceModel(
            input_size=subband_input_size,
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function,
            middle_tac_hidden_times=sbinter_middle_hidden_times
        )

        self.sb_num_neighbors = sb_num_neighbors
        # self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Unfold noisy input, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1,
                                                        num_frames)
        print(noisy_mag_unfolded.shape)
        sb_input = self.norm(noisy_mag_unfolded)

        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3),
                                 num_groups=self.num_groups_in_drop_band)  # [B, F_s, F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, F_s, T]

        # [B, F//num_groups, F_s, T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        print(sb_mask.shape)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        # output = sb_mask[:, :, :, self.look_ahead:]
        return sb_mask


if __name__ == "__main__":
    model = Inter_SubNet(
        num_freqs=257,
        look_ahead=2,
        sequence_model="LSTM",
        sb_num_neighbors=15,
        sb_output_activate_function=False,
        sb_model_hidden_size=384,
        weight_init=False,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        sbinter_middle_hidden_times=0.8
    )
    input = torch.randn(1, 1, 257, 193)
    output = model(input)
    print(output.shape)

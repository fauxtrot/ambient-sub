"""SNN classifier using snnTorch. Supports sound/silence or multi-class speaker ID."""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SoundSNN(nn.Module):
    """2-layer feedforward SNN with LIF neurons.

    Input: acoustic features rate-encoded over T timesteps.
    Output: N-class spike counts (silence, speaker IDs, etc.).
    """

    def __init__(self, num_inputs=37, hidden_size=128, num_outputs=2,
                 beta=0.95, num_steps=25):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Rate-encoded input [num_steps, batch_size, num_inputs]

        Returns:
            spk_rec: Output spikes [num_steps, batch_size, 2]
            mem_rec: Output membrane potentials [num_steps, batch_size, 2]
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)

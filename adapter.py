import torch
import torch.nn as nn


class InputAdapter(nn.Module):

    def __init__(self, hidden_size=4096, bottleneck=256):
        super().__init__()

        self.layernorm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)

    def forward(self, x):

        residual = x

        x = self.layernorm(x)
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)

        return x + residual


class EmbeddingWithAdapter(nn.Module):

    def __init__(self, embedding, adapter):
        super().__init__()

        self.embedding = embedding
        self.adapter = adapter

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = self.adapter(x)

        return x
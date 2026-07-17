
import torch
import torch.nn as nn

class SimpleTransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=2,
        max_len=512
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.position = nn.Embedding(max_len,embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim,num_classes)

    def forward(self,x):

        positions=torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        x=self.embedding(x)+self.position(positions)

        x=self.encoder(x)

        x=x.mean(dim=1)

        return self.fc(x)

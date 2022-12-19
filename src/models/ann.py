from torch import nn


class Ann(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        encoder,
        num_class=2,
        encoder_dim=32,
        n4=64,
        n5=32,
        n6=16,
    ):
        super(Ann, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, n4),
            nn.ReLU(),
            nn.Linear(n4, n5),
            nn.ReLU(),
            nn.Linear(n5, n6),
            nn.ReLU(),
            nn.Linear(n6, num_class),
        )

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, text, offsets):
        embedding_output = self.embedding(text, offsets)
        encoded = self.encoder(embedding_output)
        decoded = self.decoder(encoded)
        return decoded

    def embed(self, text, offsets):
        return self.embedding(text, offsets)

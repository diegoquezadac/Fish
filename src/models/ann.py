from torch import nn


class Ann(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, encoder):
        super(Ann, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_class),
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

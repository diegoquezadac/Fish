from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Autoencoder, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
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
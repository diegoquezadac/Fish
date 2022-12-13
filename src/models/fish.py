from torch import nn

class Fish(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(Fish, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        self.adhoc_encoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fish = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_class),
        )

        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.adhoc_encoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        for layer in self.fish:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        adhoc = self.adhoc_encoder(embedded)
        return self.softmax(self.fish(adhoc))
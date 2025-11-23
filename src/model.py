import torch
from torch import nn

class TabularModel(nn.Module):
    def __init__(self, num_numeric_features, cat_cardinalities, num_classes, emb_dim_rule=2):
        super().__init__()
        self.use_cat = len(cat_cardinalities) > 0

        if self.use_cat:
            self.embeddings = nn.ModuleList()
            emb_out_sizes = []
            for card in cat_cardinalities:
                emb_dim = min(emb_dim_rule * int(round(card ** 0.25)), 50)
                emb_dim = max(emb_dim, 2)
                self.embeddings.append(nn.Embedding(num_embeddings=card, embedding_dim=emb_dim))
                emb_out_sizes.append(emb_dim)
            self.emb_dim_total = sum(emb_out_sizes)
        else:
            self.emb_dim_total = 0

        input_dim = num_numeric_features + self.emb_dim_total

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_num, x_cat):
        if self.use_cat and x_cat.numel() > 0:
            emb_list = []
            for i, emb_layer in enumerate(self.embeddings):
                emb_list.append(emb_layer(x_cat[:, i]))
            x = torch.cat([x_num] + emb_list, dim=1)
        else:
            x = x_num
        return self.net(x)

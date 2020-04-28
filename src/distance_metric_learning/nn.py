import numpy as np
import torch


class NNMetric(torch.nn.Module):
    def __init__(self, shape, n_layers=1):
        super(NNMetric, self).__init__()
        self.shape = shape
        self.n_layers = n_layers
        self.nn_layers = torch.nn.ModuleList()
        self.activation = torch.nn.ELU()
        for _ in np.arange(n_layers):
            layer = torch.nn.Linear(shape, shape, bias=False)
            weights = np.full((shape, shape), 0.0001)
            np.fill_diagonal(weights, 1)
            layer.weight.data = torch.from_numpy(weights).float()
            self.nn_layers.append(layer)

    def forward(self, x):
        embedding = x
        for i in np.arange(self.n_layers - 1):
            embedding = self.activation(self.nn_layers[i](embedding))
        embedding = self.nn_layers[-1](embedding)
        embedding_normalized = torch.nn.functional.normalize(embedding)
        return embedding_normalized

    # def forward(self, x, y):
    #     x_embedding = self.metric(x)
    #     y_embedding = self.metric(y)
    #     return x_embedding - y_embedding


class MetricLearning():
    def __init__(self, steps=500, batch_size=128, n_layers=1, random_state=42, lr=1e-4):
        self.steps = steps
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.model = None
        self.shape = None
        self.random_state = random_state
        self.loss_history_ = None
        self.lr = lr

    def fit(self, X, side_information, th=0.5):
        self.shape = X.shape[1]
        self.model = NNMetric(self.shape, self.n_layers)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        np.random.seed(self.random_state)
        data_size = X.shape[0]

        self.loss_history_ = []
        for step in np.arange(1, self.steps + 1):
            losses = []
            try:
                idxs_1 = np.random.choice(data_size, data_size)
                idxs_2 = np.random.choice(data_size, data_size)
                for pos in np.arange(0, data_size, self.batch_size):
                    idx1 = idxs_1[pos:min(pos + self.batch_size, data_size)]
                    idx2 = idxs_2[pos:min(pos + self.batch_size, data_size)]
                    x_sample = torch.from_numpy(X[idx1]).float()
                    y_sample = torch.from_numpy(X[idx2]).float()
                    side_information_sample = torch.from_numpy(side_information[idx1, idx2].reshape(-1, 1)).float()

                    x_embedding = self.model(x_sample)
                    y_embedding = self.model(y_sample)
                    distance_predict = x_embedding - y_embedding
                    loss = (
                        ((side_information_sample - th) * distance_predict.pow(2)).sum() * 10
                        # + (x_embedding.sum(dim=1) - x_sample.sum(dim=1)).pow(2).mean()
                        # + (y_embedding.sum(dim=1) - y_sample.sum(dim=1)).pow(2).mean()
                    )
                    losses.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if step % 1 == 0:
                    print(step, np.mean(losses) / self.batch_size)
                self.loss_history_.extend(losses)
            except KeyboardInterrupt:
                break
        self.model.eval()

    def transform(self, X):
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            X_transformed = self.model(X_tensor)
            return X_transformed.numpy()

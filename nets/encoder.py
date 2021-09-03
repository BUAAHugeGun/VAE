import torch
import torch.nn as nn


class vae_encoder(nn.Module):
    def __init__(self, in_channels, output_length, depth, image_size, class_num):
        super(vae_encoder, self).__init__()
        self.in_channels = in_channels + 1
        self.output_length = output_length
        self.depth = depth
        self.image_size = image_size
        self.class_num = class_num
        self.build()

    def build(self):
        layers = []
        in_channels = self.in_channels
        out_channels = 32
        for i in range(self.depth):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU()
                )
            )
            in_channels = out_channels
            out_channels = out_channels * 2
        self.conv = nn.Sequential(*layers)
        out_size = self.image_size // (2 ** self.depth)
        out_length = out_size * out_size * in_channels
        self.fc_mu = nn.Linear(out_length, self.output_length)
        self.fc_var = nn.Linear(out_length, self.output_length)
        self.embedding = nn.Embedding(self.class_num, self.image_size * self.image_size)
        self.data_embedding = nn.Conv2d(self.in_channels - 1, self.in_channels - 1, 3, 1, 1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, x, class_label):
        label_em = self.embedding(class_label.long())
        label_em = label_em.reshape(-1, 1, self.image_size, self.image_size)
        x = self.data_embedding(x)
        x = self.conv(torch.cat([x, label_em], 1)).flatten(1)
        mu, log_var = self.fc_mu(x), self.fc_var(x)
        # z = self.reparameterize(mu, log_var)
        return mu, log_var


if __name__ == "__main__":
    encoder = vae_encoder(4, 128, 5, 64, 10)
    num_params = 0
    for param in encoder.parameters():
        num_params += param.numel()
    print(encoder)
    print(num_params / 1e6)
    torch.save(encoder.state_dict(), "test.pth")
    x = torch.randn([16, 3, 64, 64])
    y = encoder(x, )
    print(y.shape)

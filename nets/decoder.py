import torch
import torch.nn as nn


class vae_decoder(nn.Module):
    def __init__(self, in_channels, output_length, depth, image_size, class_num):
        super(vae_decoder, self).__init__()
        self.in_channels = in_channels
        self.output_length = output_length
        self.depth = depth
        self.image_size = image_size
        self.class_num = class_num
        self.build()

    def build(self):
        layers = []
        in_channels = 16 * (2 ** self.depth)
        out_channels = in_channels // 2

        out_size = self.image_size // (2 ** self.depth)
        out_length = out_size * out_size * in_channels
        self.fc = nn.Linear(self.output_length + self.output_length, out_length)
        self.embedding = nn.Embedding(self.class_num, self.output_length)

        for i in range(self.depth - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU()
                )
            )
            in_channels = out_channels
            out_channels = out_channels // 2
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU()
            )
        )
        self.conv = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, class_label):
        label_em = self.embedding(class_label.long())
        y = self.fc(torch.cat([z, label_em], 1))
        y = torch.reshape(y,
                          [y.shape[0], -1, self.image_size // (2 ** self.depth), self.image_size // (2 ** self.depth)])
        y = self.conv(y)
        y = self.final(y)

        return y


if __name__ == "__main__":
    encoder = vae_decoder(4, 128, 5, 64, 10)
    num_params = 0
    for param in encoder.parameters():
        num_params += param.numel()
    print(encoder)
    print(num_params / 1e6)
    torch.save(encoder.state_dict(), "test.pth")

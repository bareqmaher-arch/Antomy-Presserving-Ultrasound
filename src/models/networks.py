"""Generator and Discriminator architectures used by CUT and CycleGAN."""
import torch
import torch.nn as nn
import functools


def get_norm_layer(norm_type: str = "instance"):
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    raise ValueError(f"Unknown norm type: {norm_type}")


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, norm_layer, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            norm_layer(dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based generator.  n_blocks=9 is the standard choice for 256×256.
    Input/output: RGB images in [-1, 1] (tanh output).
    Intermediate feature maps at each downsampling block are accessible via
    the `encode` method — needed for PatchNCE loss.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_type: str = "instance",
        use_dropout: bool = False,
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        use_bias = norm_type == "instance"

        # Encoder (downsampling)
        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        )

        # Residual bottleneck
        res_blocks = [ResNetBlock(ngf * 4, norm_layer, use_dropout) for _ in range(n_blocks)]
        self.res = nn.Sequential(*res_blocks)

        # Decoder (upsampling)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        r = self.res(e2)
        return self.dec(r)

    def encode(self, x):
        """Return intermediate feature maps for PatchNCE."""
        feats = []
        e0 = self.enc0(x);  feats.append(e0)
        e1 = self.enc1(e0); feats.append(e1)
        e2 = self.enc2(e1); feats.append(e2)
        r = self.res(e2);   feats.append(r)
        out = self.dec(r);  feats.append(out)
        return feats


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.  Outputs a spatial map of real/fake scores.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        use_bias = norm_type == "instance"

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net: nn.Module, init_gain: float = 0.02):
    """Initialise network weights with normal distribution (standard GAN practice)."""
    def _init(m):
        name = type(m).__name__
        if "Conv" in name or "Linear" in name:
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in name:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(_init)

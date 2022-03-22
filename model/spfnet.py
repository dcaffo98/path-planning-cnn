import torch
from torch import nn
from dataset.map_sample import MapSample
from img_processing.gaussian_kernel import get_gaussian
from model.pe import GaussianRelativePE, PositionalEncoding

class SPFNet(nn.Module):

    def __init__(self, n_layers=3, gaussian_blur_kernel=0):
        super(SPFNet, self).__init__()
        
        class _conv_block_down(nn.Module):
            def __init__(self, in_channels, out_channels, activation=None, norm_first=False, is_first=False):
                super(_conv_block_down, self).__init__()
                self.activation = activation if activation is not None else nn.ReLU()
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.bn3 = nn.BatchNorm2d(out_channels * 2)
                if is_first:
                    self.conv1 = nn.Conv2d(1, out_channels, 3)
                else:
                    self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
                self.conv2 = nn.Conv2d(in_channels, out_channels, 3)
                self.conv3 = nn.Conv2d(in_channels, out_channels * 2, 3, stride=2)
                if norm_first:
                    self._fw = nn.Sequential(
                        self.conv1, self.bn1, self.activation, 
                        self.conv2, self.bn2, self.activation,
                        self.conv3, self.bn3, self.activation)
                else:
                    self._fw = nn.Sequential(
                        self.conv1, self.activation, self.bn1,
                        self.conv2, self.activation, self.bn2,
                        self.conv3, self.activation, self.bn3)
        
            def forward(self, x):
                return self._fw(x)


        class _conv_block_up(nn.Module):
            def __init__(self, in_channels, out_channels, activation=None, norm_first=False, is_last=False):
                super(_conv_block_up, self).__init__()
                self.activation = activation if activation is not None else nn.ReLU()
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.bn3 = nn.BatchNorm2d(out_channels // 2)
                self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3)
                self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 3)
                if is_last:
                    self.conv3 = nn.ConvTranspose2d(in_channels, out_channels // 2, 3, stride=2, padding=2, output_padding=1)
                else:
                    self.conv3 = nn.ConvTranspose2d(in_channels, out_channels // 2, 3, stride=2, padding=2)
                if norm_first:
                    self._fw = nn.Sequential(
                        self.conv1, self.bn1, self.activation, 
                        self.conv2, self.bn2, self.activation,
                        self.conv3, self.bn3, self.activation)
                else:
                    self._fw = nn.Sequential(
                        self.conv1, self.activation, self.bn1,
                        self.conv2, self.activation, self.bn2,
                        self.conv3, self.activation, self.bn3)
        
            def forward(self, x):
                return self._fw(x)
        
        self.gaussian_blur_kernel= gaussian_blur_kernel
        if gaussian_blur_kernel > 0:
            gaussian_kernel = torch.tensor(get_gaussian(gaussian_blur_kernel, sigma=0, normalized=True), dtype=torch.float32).view(1, 1, gaussian_blur_kernel, gaussian_blur_kernel)
            self.blur = nn.Conv2d(1, 1, gaussian_blur_kernel, padding=gaussian_blur_kernel // 2, bias=False)
            self.blur.weight.data = gaussian_kernel
        else:
            self.blur = None
        self.pe = GaussianRelativePE(100)
        self.sigm = nn.Sigmoid()
        n_channels = [64 * (2 ** i) for i in range(n_layers)]
        self.conv_down = nn.ModuleList([_conv_block_down(c, c, is_first=True if i == 0 else False) for i, c in enumerate(n_channels)])
        self.conv_up = nn.ModuleList([_conv_block_up(2 * c, 2 * c, is_last=True if i == len(n_channels) - 1 else False) for i, c in enumerate(n_channels[::-1])])
        self.bottleneck = nn.Conv2d(64, 1, 3, padding=1)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.embd = nn.Embedding(2, 1)  # 0 -> start; 1 -> goal

    def pe_forward(self, x, start, goal):
        zero_to_bsz = torch.arange(x.shape[0], dtype=torch.long, device=x.device)
        x[zero_to_bsz, :, start[:, 0], start[:, 1]] = x[zero_to_bsz, :, start[:, 0], start[:, 1]] + self.embd(torch.tensor([0], dtype=torch.long, device=x.device))
        # x[zero_to_bsz, :, goal[:, 0], goal[:, 1]] = x[zero_to_bsz, :, goal[:, 0], goal[:, 1]] + self.embd(torch.tensor([1], dtype=torch.long, device=x.device))
        return self.pe(x, goal)

    def forward(self, x, start, goal):
        """Forward pass

        Args:
            x (Tensor): (N, C, H, W) batch of 2d maps.
            start (Tensor): (N, 2) start position.
            goal (Tensor)): (N, 2) goal position.

        Returns:
            Tensor: score map. Score of a pixel should be proportional to the probability of belonging to the shortest path.
        """
        if self.gaussian_blur_kernel > 0:
            with torch.no_grad():
                x = self.blur(x)
        skip_conn = []
        x = self.pe_forward(x, start, goal)
        for i, conv in enumerate(self.conv_down):
            x = conv(x)
            if i < len(self.conv_down) - 1:
                skip_conn.append(x.detach())
        for i, conv in enumerate(self.conv_up):
            x = conv(x)
            if i < len(skip_conn):
                x = x + skip_conn[-1 - i]
        x = self.bottleneck(x)
        # x = self.pe_forward(x, start, goal)        
        x = self.conv_out(x)
        x = self.sigm(x)
        return x


if __name__ == '__main__':
    model = SPFNet(3)
    path = 'map_dataset/validation/0a2ae4a1-72e8-4daf-9b29-dcd6a24b5af6.pt'
    sample = MapSample.load(path)
    import cv2
    map = sample.bgr_map()
    cv2.imshow('map', cv2.resize(map, (600, 600)))
    cv2.waitKey(0)
    out = model(sample.map.unsqueeze(0).unsqueeze(0), sample.start.unsqueeze(0).long(), sample.goal.unsqueeze(0).long())
    out.sum().backward()
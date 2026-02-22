import torch
import torch.nn as nn
import torch.nn.functional as F


def Muse_load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint
    model_dict = model.state_dict()

    # 1. 检查是否有 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 1.1 如果当前模型是多卡（有 'module.' 前缀）但加载的参数没有 'module.' 前缀
        if k.startswith("module.") and not any(
                key.startswith("module.") for key in model_dict):
            new_key = k[len("module."):]  # 去掉 'module.' 前缀
            new_state_dict[new_key] = v

        # 1.2 如果当前模型是单卡（没有 'module.' 前缀）但加载的参数有 'module.' 前缀
        elif not k.startswith("module.") and any(
                key.startswith("module.") for key in model_dict):
            new_key = "module." + k  # 添加 'module.' 前缀
            new_state_dict[new_key] = v

        # 1.3 当前模型和加载的参数前缀一致
        else:
            new_state_dict[k] = v

    # 2. 检查模型结构是否一致
    for k, v in model_dict.items():
        if k in new_state_dict:
            try:
                model_dict[k].copy_(new_state_dict[k])
            except Exception as e:
                print(f"Error in copying parameter {k}: {e}")
        else:
            print(f"Parameter {k} not found in checkpoint. Skipping...")

    # 3. 更新模型参数
    model.load_state_dict(model_dict)

    return model


class Muse_LipROIProcessor(nn.Module):
    """
    Muse VisualFrontend compatible preprocessing.

    Expected input:
        video: (B, H, W, 3, T)
        float32 in range [0,1]

    Output:
        (B, T, 112, 112)
    """

    def __init__(self,
                 roi_size=112,
                 resize_size=224,
                 norm_mean=0.4161,
                 norm_std=0.1688):
        super().__init__()

        self.roi_size = roi_size
        self.resize_size = resize_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def forward(self, video):
        """
        video: (B, H, W, 3, T)
        return: (B, T, 112, 112)
        """

        B, H, W, C, T = video.shape
        assert C == 3
        assert video.dtype == torch.float32

        # (B, H, W, 3, T) → (B, T, 3, H, W)
        video = video.permute(0, 4, 3, 1, 2).contiguous()

        # RGB → Gray
        R = video[:, :, 0]
        G = video[:, :, 1]
        Bc = video[:, :, 2]

        gray = 0.299 * R + 0.587 * G + 0.114 * Bc  # shape: (B, T, H, W)

        # Resize
        B, T, H, W = gray.shape

        gray_4d = gray.view(B * T, 1, H, W)

        if H != self.resize_size or W != self.resize_size:
            gray_4d = F.interpolate(gray_4d,
                                    size=(self.resize_size, self.resize_size),
                                    mode="bilinear",
                                    align_corners=False)

        # Center crop
        start = self.resize_size // 2 - self.roi_size // 2
        end = start + self.roi_size

        roi_4d = gray_4d[:, :, start:end, start:end]
        # shape: (B*T, 1, 112, 112)

        roi = roi_4d.view(B, T, self.roi_size, self.roi_size)
        # Normalization
        roi = (roi - self.norm_mean) / self.norm_std

        return roi


class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes,
                                outplanes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes,
                                    outplanes,
                                    kernel_size=(1, 1),
                                    stride=stride,
                                    bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes,
                                outplanes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes,
                                outplanes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))
        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class Muse_VisualFrontend(nn.Module):
    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2),
                         padding=(0, 1, 1)),
        )
        self.resnet = ResNet()
        return

    def forward(self, inputBatch):

        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batchsize = inputBatch.shape[0]

        batch = self.frontend3D(inputBatch)
        batch = batch.transpose(1, 2)
        batch = batch.reshape(
            batch.shape[0] * batch.shape[1],
            batch.shape[2],
            batch.shape[3],
            batch.shape[4],
        )
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1, 2)
        return outputBatch


class Muse_VisualConv1D(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(channels)
        dsconv = nn.Conv1d(channels,
                           channels,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=channels,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(channels)
        pw_conv = nn.Conv1d(channels, channels, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x

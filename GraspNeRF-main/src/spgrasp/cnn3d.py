import torch
import torchvision


class C3D(torch.nn.Module):
    def __init__(self, output_depths):
        super().__init__()

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 7, padding=7 // 2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(16, 32, 5, padding=5 // 2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(32, 16, 3, padding=3 // 2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(True),
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, 3, stride=2, padding=3 // 2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, 3, stride=2, padding=3 // 2),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(True),
        )

        self.inner1 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, 1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(True),
        )
        self.inner2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 64, 1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(True),
        )

        self.out1 = torch.nn.Sequential(
            torch.nn.Conv3d(64, output_depths[0], 1, bias=False),
            torch.nn.BatchNorm3d(output_depths[0]),
            torch.nn.ReLU(True),
        )
        self.out2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, output_depths[1], 3, bias=False, padding=3 // 2),
            torch.nn.BatchNorm3d(output_depths[1]),
            torch.nn.ReLU(True),
        )
        self.out3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, output_depths[2], 3, bias=False, padding=3 // 2),
            torch.nn.BatchNorm3d(output_depths[2]),
            torch.nn.ReLU(True),
        )

        torch.nn.init.kaiming_normal_(self.conv0[0].weight)
        torch.nn.init.kaiming_normal_(self.conv0[3].weight)
        torch.nn.init.kaiming_normal_(self.conv0[6].weight)
        torch.nn.init.kaiming_normal_(self.conv1[0].weight)
        torch.nn.init.kaiming_normal_(self.conv2[0].weight)
        torch.nn.init.kaiming_normal_(self.inner1[0].weight)
        torch.nn.init.kaiming_normal_(self.inner2[0].weight)
        torch.nn.init.kaiming_normal_(self.out1[0].weight)
        torch.nn.init.kaiming_normal_(self.out2[0].weight)
        torch.nn.init.kaiming_normal_(self.out3[0].weight)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["coarse"] = out

        intra_feat = torch.nn.functional.interpolate(
            intra_feat, scale_factor=2, mode="trilinear", align_corners=False
        ) + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["medium"] = out

        intra_feat = torch.nn.functional.interpolate(
            intra_feat, scale_factor=2, mode="trilinear", align_corners=False
        ) + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["dense"] = out

        return outputs


class C3Dmain(torch.nn.Module):
    def __init__(self, output_depths):
        super().__init__()

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 7, padding=7 // 2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(16, 32, 5, padding=5 // 2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(32, 16, 3, padding=3 // 2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(True),
        )

        self.inner2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 64, 1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(True),
        )

        self.out3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, output_depths[2], 3, bias=False, padding=3 // 2),
            torch.nn.BatchNorm3d(output_depths[2]),
            torch.nn.ReLU(True),
        )

        torch.nn.init.kaiming_normal_(self.conv0[0].weight)
        torch.nn.init.kaiming_normal_(self.conv0[3].weight)
        torch.nn.init.kaiming_normal_(self.conv0[6].weight)
        torch.nn.init.kaiming_normal_(self.inner2[0].weight)
        torch.nn.init.kaiming_normal_(self.out3[0].weight)

    def forward(self, x):
        conv0 = self.conv0(x)
        intra_feat = self.inner2(conv0)
        out = self.out3(intra_feat)

        return out

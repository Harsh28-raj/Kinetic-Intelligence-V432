import torch
import torch.nn as nn

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, grid_w, grid_h):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        # Refinement layers
        self.refinement = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU()
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # DINov2 output format: [Batch, Tokens, Channels]
        b, n, c = x.shape
        # Reshape tokens back to spatial grid
        x = x.reshape(b, self.grid_h, self.grid_w, c).permute(0, 3, 1, 2)
        
        x = self.stem(x)
        x = self.refinement(x)
        return self.classifier(x)
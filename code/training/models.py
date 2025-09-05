from sympy import use
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Input is of shape (batch_size, 1, 155, 240, 240)

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(Simple3DCNN, self).__init__()
        # input shape: (batch_size, in_channels, 155, 224, 224)
        
        self.encoder = nn.Sequential(
            nn.Sequential(                                              # Layer 0
                nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),   
                nn.ReLU(),                                              
                nn.MaxPool3d(2)                                         
            ), # output shape: (batch_size, 16, 77, 112, 112)
            nn.Sequential(                                              # Layer 1 
                nn.Conv3d(16, 64, kernel_size=3, padding=1),
                nn.ReLU(),                                              
                nn.MaxPool3d(2)                                          
            ), # output shape: (batch_size, 64, 38, 56, 56)
            nn.Sequential(                                              # Layer 2
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),                                              
                nn.MaxPool3d(2)                                
            ) # output shape: (batch_size, 128, 19, 28, 28)
        )
        
        self.gap = nn.AdaptiveAvgPool3d(1) # output shape: (batch_size, 128, 1, 1, 1)
        
        self.projection_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, out_channels)
        )
        
        self.hooks = []
        self.features = {}
        
    def register_hooks(self):
        """
        Registers forward hooks on all layers in the encoder.
        The hooks will sae intermediate outputs into self.features.
        """
        
        for idx, layer in enumerate(self.encoder):
            def hook_fn(module, input, output, idx=idx):
                self.features[f"{idx}"] = output.clone().detach()
                
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook) 
            
    def remove_hooks(self):
        """
        Remove all hooks to prevent duplicate entries.
        """
        
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, x):
        features = self.encoder(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1) # flatten
        projections = self.projection_head(features)
        return features, projections


class Enhanced3DCNN1(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(Enhanced3DCNN1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(  # Layer 0
                nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU()
            ), # output shape: (batch_size, 32, 77, 112, 112)
            nn.Sequential(  # Layer 1
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU()
            ), # output shape: (batch_size, 64, 38, 56, 56)
            nn.Sequential(  # Layer 2
                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU()
            ), # output shape: (batch_size, 128, 19, 28, 28)
            nn.Sequential(  # Layer 3
                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 256),
                nn.GELU()
            ), # output shape: (batch_size, 256, 9, 28, 28)
            nn.Sequential(  # Layer 4
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(16, 256),
                nn.GELU()
            ) # output shape: (batch_size, 256, 9, 28, 28)
        )
        
        # input shape flattened: (batch_size, 256 * 9 * 14 * 14)
        self.gap = nn.AdaptiveAvgPool3d(1) # output shape: (batch_size, 256, 1, 1, 1)

        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

        self.hooks = []
        self.features = {}

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def register_hooks(self):
        for idx, layer in enumerate(self.encoder):
            def hook_fn(module, input, output, idx=idx):
                self.features[f"{idx}"] = output.clone().detach()
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, x):
        features = self.encoder(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        projections = self.projection_head(features)
        return features, projections


class Enhanced3DCNN2(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(Enhanced3DCNN2, self).__init__()

        def conv_block(in_c, out_c, stride=1, dilation=1):
            """Defines a convolutional block with optional residual connection."""
            layers = [
                nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, out_c),
                nn.GELU()
            ]
            return nn.Sequential(*layers)
        
        # Input: (B, 1, 155, 224, 224)
        
        # ====== Encoder ====== 
        self.encoder_layers = nn.ModuleList()
        
        # Layer 0: Initial downsampling (Stride 2)
        layer = conv_block(in_channels, 16, stride=2) 
        self.encoder_layers.append(layer) # (B, 16, 77, 112, 112)

        # Layer 1: Strided convolution (Stride 2)
        layer = conv_block(16, 32, stride=2)
        self.encoder_layers.append(layer) # (B, 32, 38, 56, 56)

        # Layer 2: Dilated Convolution
        layer = conv_block(32, 64, stride=1, dilation=2)
        self.encoder_layers.append(layer) # (B, 64, 38, 56, 56)

        # Layer 3: Strided convolution
        layer = conv_block(64, 128, stride=2, dilation=1)
        self.encoder_layers.append(layer) # (B, 128, 19, 28, 28)

        # Layer 4: Dilated convolution
        layer = conv_block(128, 256, stride=1, dilation=2)
        self.encoder_layers.append(layer) # (B, 256, 19, 28, 28)

        # ====== Global Average Pooling ======
        self.gap = nn.AdaptiveAvgPool3d(1)  # (B, 256, 1, 1, 1)

        # ====== Projection Head ======
        self.projection_head = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, out_channels)
        )

        # ====== Hook Management ======
        self.hooks = []
        self.features = {}

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stability and convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def register_hooks(self):
        """Registers forward hooks on all layers in the encoder to extract features."""
        for idx, layer in enumerate(self.encoder_layers):
            def hook_fn(module, input, output, idx=idx):
                self.features[f"{idx}"] = output.clone().detach()
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def remove_hooks(self):
        """Removes hooks to prevent memory issues."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, x):
        """Forward pass with residual connections inside the encoder."""
        for layer in self.encoder_layers:
            x = layer(x)

        features = self.gap(x)  # Global Avg Pooling
        features = features.view(features.size(0), -1)  # Flatten
        projections = self.projection_head(features)
        return features, projections
    

def widen(ch, a):
    return max(1, int(ch * a))    

    
class Enhanced3DCNN3(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, pretrained=False, use_ckpt=True, a=0.75):
        super(Enhanced3DCNN3, self).__init__()
        
        self.use_ckpt = use_ckpt
        
        # ====== Encoder: Using r3d_18 ======
        r3d = r3d_18(pretrained=pretrained)
        # Modifying the first convolution layer to accept a single channel
        r3d.stem[0] = nn.Conv3d(
            in_channels, 64, 
            kernel_size=(3, 7, 7), 
            stride=(1, 2, 2), 
            padding=(1, 3, 3), 
            bias=False
        )
        # Build our encoder using the stem and subsequent residual layers.
        self.stem   = r3d.stem          # (B, 64, …)
        self.layer1 = r3d.layer1        # (B, 64, …)
        self.layer2 = r3d.layer2        # (B, 128, …)
        self.layer3 = r3d.layer3        # (B, 256, …)
        self.layer4 = r3d.layer4        # (B, 512, …)
        
        # ====== Global Average Pooling ======
        self.gap = nn.AdaptiveAvgPool3d(1)  # (B, 512, 1, 1, 1)

        # ====== Projection Head ======
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

        # ====== Hook Management ======
        self.hooks = []
        self.features = {}

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stability and convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def register_hooks(self):
        """Registers forward hooks on all layers in the encoder to extract features."""
        for idx, layer in enumerate(self.encoder_layers):
            def hook_fn(module, input, output, idx=idx):
                self.features[f"{idx}"] = output.clone().detach()
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def remove_hooks(self):
        """Removes hooks to prevent memory issues."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _maybe_ckpt(self, layer, x):
        """Run `layer(x)` with or without checkpoint depending on flag."""
        if not self.use_ckpt:
            return layer(x)
        return checkpoint_sequential(layer, segments=len(layer), input=x, use_reentrant=False)

    def forward(self, x):
        x = self.stem(x)                    # keep stem activations

        # layer‑1 is lightweight → run normally
        x = self.layer1(x)

        # memory‑heavy stages: optionally checkpoint
        x = self._maybe_ckpt(self.layer2, x)
        x = self._maybe_ckpt(self.layer3, x)
        x = self._maybe_ckpt(self.layer4, x)

        x = self.gap(x).flatten(1)          # (B, 512)
        z = self.projection_head(x)         # (B, out_channels)
        return x, z
    
    
    
    
# -------- (2+1)-D factorised 3 × 3 × 3 conv --------------------------
class R2Plus1D(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, pad=1):
        mid = max(1, (cin * cout * k * k) // (cin * k * k + cout * k))
        super().__init__(
            nn.Conv3d(cin, mid, (1, k, k), (1, stride, stride), (0, pad, pad), bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=True),
            nn.Conv3d(mid, cout, (k, 1, 1), (stride, 1, 1), (pad, 0, 0), bias=False),
            nn.BatchNorm3d(cout),
        )


# -------- scaled BasicBlock using R(2+1)D convs ----------------------
# ------------------- FIXED basic block (no second scaling) -------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, conv_factory):
        super().__init__()
        self.conv1 = conv_factory(cin,  cout, 3, stride, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv_factory(cout, cout, 3, 1,      1)

        self.down = (
            nn.Identity()
            if stride == 1 and cin == cout else
            nn.Sequential(nn.Conv3d(cin, cout, 1, stride, 0, bias=False),
                          nn.BatchNorm3d(cout))
        )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + self.down(x))


# --------------------------------------------------------------------
class Enhanced3DCNN4(nn.Module):
    """
    • R(2+1)D convolutions
    • width multiplier `α` (channel scaling)
    • optional checkpointing
    """
    def __init__(self, in_channels=1, out_channels=128, α=0.75, use_ckpt=True):
        super().__init__()
        self.use_ckpt = use_ckpt
        conv_factory = R2Plus1D
        c = lambda ch: max(1, int(ch * α))           # scaled channels

        # stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c(64), (3, 7, 7), (1, 2, 2), (1, 3, 3), bias=False),
            nn.BatchNorm3d(c(64)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )

        # residual stages (2-2-2-2 blocks like ResNet-18)
        self.layer1 = self._make_layer(c(64),  c(64),  2, 1, conv_factory)
        self.layer2 = self._make_layer(c(64),  c(128), 2, 2, conv_factory)
        self.layer3 = self._make_layer(c(128), c(256), 2, 2, conv_factory)
        self.layer4 = self._make_layer(c(256), c(512), 2, 2, conv_factory)

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(c(512), c(256)),
            nn.ReLU(),
            nn.Linear(c(256), out_channels)
        )
        
        self.encoder_layers = [self.stem, self.layer1, self.layer2,
                               self.layer3, self.layer4]
        self.features: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    # --------------------  hooks  -----------------------------------
    def register_hooks(self):
        for idx, layer in enumerate(self.encoder_layers):
            def _fn(_, __, out, idx=idx):
                self.features[str(idx)] = out.detach()
            self.hooks.append(layer.register_forward_hook(_fn))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.features.clear()

    # -- helpers -------------------------------------------------------
    def _make_layer(self, cin, cout, blocks, stride, α):
        layers = [BasicBlock(cin, cout, stride, α)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(cout, cout, 1, α))
        return nn.Sequential(*layers)

    def _ckpt(self, layer, x):
        return (checkpoint_sequential(layer, len(layer), x, use_reentrant=False)
                if self.use_ckpt else layer(x))

    # -- forward -------------------------------------------------------
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self._ckpt(self.layer2, x)
        x = self._ckpt(self.layer3, x)
        x = self._ckpt(self.layer4, x)
        feats = self.gap(x).flatten(1)
        z     = self.head(feats)
        return feats, z
    
import numpy as np
import torch
import jax.numpy as jnp
import mediapy

from videoprism import models as vp

class TorchFeedForward(torch.nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, use_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=use_bias)
        self.fc2 = torch.nn.Linear(hidden_dim, dim, bias=use_bias)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate='none')  # Use exact GELU like JAX
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TorchLayerNorm(torch.nn.Module):
    
    def __init__(self, dim: int, use_bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim)) if use_bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # JAX-compatible layer norm computation
        mean = x.mean(dim=-1, keepdim=True)
        # JAX uses: jnp.mean(jnp.square(inputs - mean), axis=[-1], keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        # JAX uses: jax.lax.rsqrt(var + self.epsilon)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def _image_to_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, height, width, channels = image.shape
    assert height == width, "Height and width must be equal for this model."
    
    num_patches = (height // patch_size) ** 2
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, num_patches, patch_size * patch_size * channels)
    
    return patches


class FeedForwardPyTorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, has_bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=has_bias)
    
    def forward(self, x):
        return self.linear(x)


class TrainablePositionalEmbedding(torch.nn.Module):
    def __init__(self, seq_length: int, emb_dim: int):
        super().__init__()
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.pos_emb = torch.nn.Parameter(torch.randn(seq_length, emb_dim))

    def forward(self, seq_length: int) -> torch.Tensor:
        position = torch.arange(seq_length, dtype=torch.int64).unsqueeze(0)
        pos_emb_var = self.pos_emb[:seq_length]
        one_hot_ids = torch.nn.functional.one_hot(position, num_classes=seq_length).float()
        embs = torch.einsum('...y,yz->...z', one_hot_ids, pos_emb_var)
        return embs


class VisionTransformer(torch.nn.Module):
    def __init__(self, patch_size: int, pos_emb_shape: tuple, model_dim: int,
                 num_spatial_layers: int, num_temporal_layers: int, num_heads: int,
                 mlp_dim: int, atten_logit_cap: float, scan: bool):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb_shape = pos_emb_shape
        self.model_dim = model_dim
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.atten_logit_cap = atten_logit_cap
        self.scan = scan

        self.layers = torch.nn.ModuleList([
            TorchFeedForward(model_dim, mlp_dim, use_bias=True) for _ in range(num_spatial_layers)
        ])

    def forward(self, x: torch.Tensor, paddings: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FactorizedEncoder(torch.nn.Module):
    def __init__(self, patch_size: int, pos_emb_shape: tuple, model_dim: int,
                 num_spatial_layers: int, num_temporal_layers: int, num_heads: int,
                 mlp_dim: int, atten_logit_cap: float, scan: bool):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb_shape = pos_emb_shape
        self.model_dim = model_dim
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.atten_logit_cap = atten_logit_cap
        self.scan = scan

        self.patch_projection = torch.nn.Linear(
            in_features=patch_size * patch_size * 3,
            out_features=model_dim,
            bias=True
        )

        self.spatial_pos_emb = TrainablePositionalEmbedding(
            seq_length=np.prod(pos_emb_shape[1:]),
            emb_dim=model_dim
        )

        self.vision = VisionTransformer(
            patch_size=patch_size,
            pos_emb_shape=pos_emb_shape,
            model_dim=model_dim,
            num_spatial_layers=num_spatial_layers,
            num_temporal_layers=num_temporal_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            atten_logit_cap=atten_logit_cap,
            scan=scan
        )

        self.temporal_pos_emb = TrainablePositionalEmbedding(
            seq_length=pos_emb_shape[0],
            emb_dim=model_dim
        )

        self.temporal_transformer = VisionTransformer(
            patch_size=patch_size,
            pos_emb_shape=pos_emb_shape,
            model_dim=model_dim,
            num_spatial_layers=num_temporal_layers,
            num_temporal_layers=num_temporal_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            atten_logit_cap=atten_logit_cap,
            scan=scan
        )

        self.temporal_ln = TorchLayerNorm(
            dim=model_dim,
            use_bias=True,
            eps=1e-6
        )

    def encode_with_patches(self, patches: torch.Tensor, image_shape: tuple) -> torch.Tensor:
        t, h, w = image_shape
        b = patches.shape[0] // t

        # Get patches from the input tensor
        patches = self.patch_projection(patches)

        spatial_pos_emb_shape = self.pos_emb_shape[1:]
        spatial_seq_length = np.prod(spatial_pos_emb_shape)

        # Create spatial positional embeddings
        spatial_pos_emb = self.spatial_pos_emb(spatial_seq_length)

        # TODO: values start to diverge after this layer norm.. not sure why
        self.spatial_ln = TorchLayerNorm(
            dim=self.model_dim,
            use_bias=True,
            eps=1e-6
        )

        num_row_patches = h // self.patch_size
        num_col_patches = w // self.patch_size

        if spatial_pos_emb_shape != (num_row_patches, num_col_patches):
            # TODO: handle resizing of positional embeddings
            pass

        # Add spatial positional embeddings to patches
        patches = patches + spatial_pos_emb

        # Extract spatial features from the patches via the vision transformer
        features = self.vision(patches)
        features = self.spatial_ln(features)
        spatial_features = features

        # Reshape features to prepare for temporal processing
        n = spatial_features.shape[1]
        d = spatial_features.shape[2]

        features = features.view(b * n, t, d)
        temporal_paddings = None

        temporal_seq_length = self.pos_emb_shape[0]

        # Create temporal positional embeddings
        temporal_pos_emb = self.temporal_pos_emb(temporal_seq_length)

        if temporal_seq_length != t:
            # handle resizing of temporal positional embeddings
            pass

        # Add temporal positional embeddings to features
        features = features + temporal_pos_emb

        # Extract temporal features from the reshaped features via the temporal transformer
        features = self.temporal_transformer(
            features,
            paddings=temporal_paddings
        )
        features = self.temporal_ln(features)
        
        # TODO: review if we should return the spatial features as well
        # Reshape features and return
        features = features.view(b, t * n, d)
        embeddings, outputs = features, {}
        return embeddings, outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, frames, height, width, channels = x.shape
        assert height == width, "Height and width must be equal for this model."
        
        reshaped_inputs = x.reshape(batch_size * frames, height, width, channels)
        patches = _image_to_patch(reshaped_inputs, self.patch_size)
        patches = self.encode_with_patches(patches, (frames, height, width))
        return patches


class VideoPrismModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = FactorizedEncoder(
            patch_size=18,
            pos_emb_shape=(16, 16, 16),
            model_dim=768,
            num_spatial_layers=12,
            num_temporal_layers=4,
            num_heads=12,
            mlp_dim=3072,
            atten_logit_cap=50.0,
            scan=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def read_and_preprocess_video(filename, target_num_frames=16, target_frame_size=(288, 288)):
    frames = mediapy.read_video(filename)
    frame_indices = np.linspace(0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32)
    frames = np.array([frames[i] for i in frame_indices])
    frames = mediapy.resize_video(frames, shape=target_frame_size)
    frames = mediapy.to_float01(frames)
    return frames


def load_and_convert_weights(model, loaded_state):
    jax_kernel = loaded_state['params']['patch_projection']['linear']['kernel']
    jax_bias = loaded_state['params']['patch_projection']['linear']['bias']

    # TODO: copy all weight

    # just copy the first few weights for now to debug
    model.encoder.patch_projection.weight.data.copy_(torch.from_numpy(np.array(jax_kernel).T).float())
    model.encoder.patch_projection.bias.data.copy_(torch.from_numpy(np.array(jax_bias)).float())
    model.encoder.spatial_pos_emb.pos_emb.data = torch.tensor(loaded_state['params']['spatial_pos_emb']['emb_var'])


if __name__ == "__main__":
    
    model_name = 'videoprism_public_v1_base'
    flax_model = vp.MODELS[model_name]()
    loaded_state = vp.load_pretrained_weights(model_name, checkpoint_path='./videoprism_public_v1_base.npz')

    # @jax.jit
    def forward_fn(inputs, train=False):
        return flax_model.apply(loaded_state, inputs, train=train)

    num_frames = 16
    torch_frames = torch.randn(1, num_frames, 288, 288, 3, dtype=torch.float32)
    torch_frames = torch.round(torch_frames * 1e4) / 1e4
    frames = jnp.asarray(torch_frames.numpy())

    embeddings, _ = forward_fn(frames)

    model = VideoPrismModel(model_name='videoprism_public_v1_base')
    load_and_convert_weights(model, loaded_state)
    torch_embeddings, _ = model(torch_frames)

    print("Ref shape:\t\t\t   ", embeddings.shape)
    print("Torch output shape:\t", torch_embeddings.shape)
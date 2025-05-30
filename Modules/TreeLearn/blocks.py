# Definition of blocks needed for TreeLearn

from collections import OrderedDict
import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, kernel_size, indice_key=None):
        super().__init__()

        # residual connection either for unchanged number of channels or for increased number of channels
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        # 2 subsequent conv blocks, RF = 3x3x3 each
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=int((kernel_size-1)/2),
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=int((kernel_size-1)/2),
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output

class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, kernel_size, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, kernel_size, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }

        blocks = OrderedDict(blocks)
        # 2 residual blocks with RF 5x5x5 each; RF += 8 ()
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:

            # downsample by factor 2; RF *= 2
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, kernel_size, indice_key_id=indice_key_id + 1)

            # used to upsample higher level size to current level size
            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            # used to combine higher level and current level features into feature map of size nPlanes[0]
            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    kernel_size,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output

# The following calculations do not account for the input convolutions which further increase the receptive field.
# receptive field of ublock with n = len(nPlanes): (1 + block_reps * ((kernel_size - 1) * 2) * 2^(n-1)) + block_reps * ((kernel_size - 1) * 2) * (2^(n-1) - 1) (proof by induction)

# derivation for block_reps = 2 and kernel_size = 3
# derivation: 9 * 2 + 8
# derivation: (9 * 2 + 8) * 2 + 8
# derivation: ((9 * 2 + 8) * 2 + 8) * 2 + 8
# derivation: (((9 * 2 + 8) * 2 + 8) * 2 + 8) * 2 + 8
# etc ...

# for standard configuration of n = 7, it follows that RF = 9 * 2^6 + 8 * (2^6 - 1) = 1080x1080x1080
# for kernel size of 5 instead of 3 it follows for standard configuration that RF = 17 * 2^6 + 16 * (2^6 - 1)
# for kernel size of 3 and n = 2, it follows that RF = 9 * 2^1 + 8 * (2^1 - 1) = 26

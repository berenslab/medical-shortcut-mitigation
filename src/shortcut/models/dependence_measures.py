from typing import List, Union

import math
import torch
import torch.nn as nn
from torch import Tensor


class MIEstimator(nn.Module):
    """Lower bound mutual information (MI) estimator.

    Reference: https://arxiv.org/abs/1801.04062

    Attributes:
        feature_dim: Dimension of input feature tensor to estimate MI from.
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = "cuda"

        # Create statistics network.
        self.stats_network = statistics_network(self.feature_dim)
        self.register_buffer("running_exp", torch.tensor(float("nan")))

    def forward(self, x, y):
        batch_size = x.shape[0]
        xy = torch.cat(
            [x.repeat_interleave(batch_size, dim=0), y.tile(batch_size, 1)], -1
        )
        stats = self.stats_network(xy).reshape(batch_size, batch_size)

        diag = torch.diagonal(stats).mean()
        logmeanexp = self.logmeanexp_off_diagonal(stats)
        mi_estimate = diag - logmeanexp
        return mi_estimate

    def logmeanexp_off_diagonal(self, x):
        batch_size = x.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
        off_diag = x[mask].view(batch_size, batch_size - 1)
        logsumexp = torch.logsumexp(off_diag, dim=1) - math.log(batch_size - 1)
        return logsumexp.mean()


class statistics_network(torch.nn.Module):
    """Statistice neural network for mutual information estimator `MIEstimator`

    Attributes:
        in_feature: Input feature dimension.
    """

    def __init__(
        self,
        in_feature: int,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_feature, 400, bias=False),
            torch.nn.BatchNorm1d(400),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(400, 400, bias=False),
            torch.nn.BatchNorm1d(400),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(400, 400, bias=False),
            torch.nn.BatchNorm1d(400),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(400, 1),
        )

    def forward(self, x):
        return self.layers(x)


def dCorEstimator(x: torch.Tensor, y: torch.Tensor):
    """Calculate the empirical distance correlation as described in [2].
    This statistic describes the dependence between `x` and `y`, which are
    random vectors of arbitrary length. The statistics' values range between 0
    (implies independence) and 1 (implies complete dependence).

    Args:
        x: Tensor of shape (batch-size, x_dimensions).
        y: Tensor of shape (batch-size, y_dimensions).

    Returns:
        The empirical distance correlation between `x` and `y`

    References:
        [1] https://en.wikipedia.org/wiki/Distance_correlation
        [2] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
            "Measuring and testing dependence by correlation of distances".
            Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.
    """
    # Euclidean distance between vectors.
    a = torch.cdist(x, x, p=2)  # N x N
    b = torch.cdist(y, y, p=2)  # N x N

    a_row_means = a.mean(axis=0, keepdims=True)
    b_row_means = b.mean(axis=0, keepdims=True)
    a_col_means = a.mean(axis=1, keepdims=True)
    b_col_means = b.mean(axis=1, keepdims=True)
    a_mean = a.mean()
    b_mean = b.mean()

    # Empirical distance matrices.
    A = a - a_row_means - a_col_means + a_mean
    B = b - b_row_means - b_col_means + b_mean

    # Empirical distance covariance.
    dcov = torch.mean(A * B)

    # Empirical distance variances.
    dvar_x = torch.mean(A * A)
    dvar_y = torch.mean(B * B)

    return torch.sqrt(dcov / torch.sqrt(dvar_x * dvar_y))


class GRLayer(torch.autograd.Function):
    """Gradient reversal layer.

    Acts as an identity function in the forward pass and
    inverts the gradient during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.scale, None


class AdvClassifier(torch.nn.Module):
    """Nonlinear adversarial Classification head with gradient reversal layer (GRL).

    Reference for GRL:
        paper: https://arxiv.org/abs/1505.07818
        example implementation: Adversarial classifier: https://github.com/NaJaeMin92/pytorch-DANN

    Attributes:
        z_shape: Latent space dimension.
        c_shape: Output/class dimension.
    """

    def __init__(
        self,
        z_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(z_shape, z_shape * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(z_shape * 2, z_shape * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(z_shape * 2, c_shape),
        )

    def forward(self, z, alpha):
        reversed_input = GRLayer.apply(z, alpha)
        x = self.layers(reversed_input)
        return x


class MMDEstimator(nn.Module):
    """Quadratic-time Maximum Mean Discrepancy (MMD) estimate.

    - Time complexity O(m^2) and memory O(m^2).
    - Unbiased, low variance, expensive.
    - Uses full pairwise comparisons for kernel computation.

    Reference: Gretton et al., A Kernel Two-Sample Test, 2012.

    Attributes:
        biased_estimate: If True, compute the biased estimate, including the diagonal
            self-similarity terms. Makes the estimate non-negative but slightly
            overestimates the true MMD.
        sigmas: Bandwidths for RBF kernels. If None, defaults to powers of 2 from
            2^-3 to 2^3.
    """

    def __init__(self, biased_estimate: bool, sigmas: Union[List, Tensor] = None):
        super().__init__()
        self.biased_estimate = biased_estimate
        if sigmas is None:
            # sigmas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            sigmas = torch.tensor([2**i for i in range(-3, 4)], dtype=torch.float32)
        self.register_buffer("sigmas", sigmas)

    def _pairwise_squared_distances(self, x, y):
        """Computes pairwise squared Euclidean distances between rows of x and y.

        This uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2⟨x, y⟩

        Inputs:
            x: Tensor of shape [n, d].
            y: Tensor of shape [m, d].

        Returns:
            Tensor of shape [n, m] of squared distances.
        """
        x_norm = (x**2).sum(dim=1).unsqueeze(1)  # shape: [n, 1]
        y_norm = (y**2).sum(dim=1).unsqueeze(0)  # shape: [1, m]
        # Comment: In a deep learning setting, n and m have the same dimension (batch-size).
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist, min=0.0)  # numerical stability

    def _rbf_kernel(self, x, y):
        """Computes the RBF kernel matrix with a mixture of sigmas.

        Inputs:
            x: Tensor of shape [n, d].
            y: Tensor of shape [m, d].

        Returns:
            Averaged kernel matrix: [B, B].
        """
        dist_sq = self._pairwise_squared_distances(x, y)  # [n, m]
        beta = 1.0 / (2.0 * self.sigmas.view(-1, 1, 1))  # [K, 1, 1], K = num_sigmas

        # Vectorized computation of multiple RBF kernels.
        # kernel_vals[k] = exp(-beta[k] * dist_sq), shape: [K, n, m]
        kernel_vals = torch.exp(-beta * dist_sq)  # broadcasted over sigmas

        # Average across all RBF kernels to get mixture kernel
        return kernel_vals.mean(dim=0)  # shape: [n, m]

    def forward(self, x, y):
        """Computes the unbiased quadratic-time MMD loss between two batches.

        Inputs:
            x, y: Tensors of shape [B, D].

        Returns:
            Scalar MMD^2 loss.
        """
        n, d1 = x.shape
        m, d2 = y.shape
        assert d1 == d2, "x and y must have the same feature dimensions"

        # Compute kernel matrices
        k_xx = self._rbf_kernel(x, x)  # [n, n]
        k_yy = self._rbf_kernel(y, y)  # [m, m]
        k_xy = self._rbf_kernel(x, y)  # [n, m]

        if self.biased_estimate:
            mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        else:
            # Masks to exclude diagonal terms (i == j) for unbiased estimate.
            mask_xx = ~torch.eye(n, dtype=torch.bool, device=x.device)
            mask_yy = ~torch.eye(m, dtype=torch.bool, device=y.device)
            mmd = k_xx[mask_xx].mean() + k_yy[mask_yy].mean() - 2.0 * k_xy.mean()
        return mmd


class LinearMMDEstimator(nn.Module):
    """Linear-time Maximum Mean Discrepancy (MMD) estimate.

    - Time complexity O(m) and memory O(1).
    - Biased estimator, high variance, computationally cheap, scalable (data streams).
    - Pairs samples one-to-one instead of doing all pairwise comparisons.

    It assumes equal-sized, aligned batches of samples from the two distributions —
    i.e., x.shape == y.shape == (n, d).

    Reference: Gretton et al., A Kernel Two-Sample Test, Section 6, 2012.

    Attributes:
        sigmas: Bandwidths for RBF kernels. If None, defaults to powers
            of 2 from 2^-3 to 2^3.
    """

    def __init__(self, sigmas: Union[List, Tensor] = None):
        super().__init__()
        if sigmas is None:
            # sigmas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            sigmas = torch.tensor([2**i for i in range(-3, 4)], dtype=torch.float32)
        else:
            sigmas = torch.tensor(sigmas)
        self.register_buffer("sigmas", sigmas)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the linear-time MMD^2 estimate between two batches (vectorized).

        Inputs:
            x, y: Tensors of shape [n, d], paired sample-wise.

        Returns:
            Scalar: Estimated MMD^2.
        """
        assert x.shape == y.shape and x.ndim == 2
        n = x.shape[0]
        assert n % 2 == 0, "Batch size must be even for linear-time MMD"

        # Reshape to [n//2, 2, d] for pairing
        x_pairs = x.view(n // 2, 2, -1)
        y_pairs = y.view(n // 2, 2, -1)

        # Unpack pairs: [n//2, d]
        x1, x2 = x_pairs[:, 0], x_pairs[:, 1]
        y1, y2 = y_pairs[:, 0], y_pairs[:, 1]

        # Compute all pairwise squared distances
        d_xx = (x1 - x2).pow(2).sum(dim=1)  # [n//2]
        d_yy = (y1 - y2).pow(2).sum(dim=1)  # [n//2]
        d_xy = (x1 - y2).pow(2).sum(dim=1)  # [n//2]
        d_yx = (x2 - y1).pow(2).sum(dim=1)  # [n//2]

        # [n//2, 1] x [1, K] -> [n//2, K]
        beta = 1.0 / (2.0 * self.sigmas)  # [K]
        d_xx = torch.exp(-d_xx.unsqueeze(1) * beta)
        d_yy = torch.exp(-d_yy.unsqueeze(1) * beta)
        d_xy = torch.exp(-d_xy.unsqueeze(1) * beta)
        d_yx = torch.exp(-d_yx.unsqueeze(1) * beta)

        # Combine and average
        mmd_batch = d_xx + d_yy - d_xy - d_yx  # [n//2, K]
        mmd = mmd_batch.mean()  # Scalar

        return mmd

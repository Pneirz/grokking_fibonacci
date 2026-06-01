"""Model definitions for Fibonacci grokking experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


def activation_layer(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "leaky_relu":
        return nn.LeakyReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "softplus":
        return nn.Softplus()
    if activation == "elu":
        return nn.ELU()
    raise ValueError(f"unknown activation: {activation}")


class MLPClassifier(nn.Module):
    """A compact MLP-style network.

    With depth=0 this becomes a single linear layer, which is useful as a
    sanity check for raw Fibonacci transition extrapolation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_layer(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedBlock(nn.Module):
    """A generic residual GLU-style block."""

    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_layer(activation)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.activation(self.value(h)) * torch.sigmoid(self.gate(h))


class ProductBlock(nn.Module):
    """A direct multiplicative block without Fibonacci-specific structure."""

    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.left = nn.Linear(hidden_dim, hidden_dim)
        self.right = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_layer(activation)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.activation(self.left(h)) * self.right(h)


class MultiplicativeMLP(nn.Module):
    """MLP with learned internal products.

    This is a generic architecture: it receives the same raw scalar input as
    the additive MLP and has no Fibonacci-specific features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        activation: str = "silu",
        residual_gated: bool = False,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.input = nn.Linear(input_dim, hidden_dim)
        block_class = GatedBlock if residual_gated else ProductBlock
        self.blocks = nn.ModuleList(
            block_class(hidden_dim, activation) for _ in range(depth)
        )
        self.activation = activation_layer(activation)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input(x))
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class FourierFeatureMLP(nn.Module):
    """MLP over a generic Fourier basis of the scalar index.

    This is a task-agnostic feature control: it receives only n and does not
    encode Fibonacci, Binet, recurrence state, or target transforms.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        activation: str = "silu",
        num_frequencies: int = 16,
        max_frequency: float = 1.0,
    ) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.register_buffer(
            "frequencies",
            torch.logspace(
                start=0.0,
                end=math.log10(max_frequency),
                steps=num_frequencies,
            ),
        )
        input_dim = 1 + 2 * num_frequencies
        self.net = MLPClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x[:, :1] * self.frequencies.to(dtype=x.dtype, device=x.device)
        features = torch.cat([x[:, :1], torch.sin(angles), torch.cos(angles)], dim=1)
        return self.net(features)


class SineLayer(nn.Module):
    """SIREN-style sine layer with standard initialization."""

    def __init__(self, in_dim: int, out_dim: int, omega_0: float, is_first: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega_0 = omega_0
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """Sine-activation MLP as another generic functional-basis control."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("SirenMLP requires depth >= 1")
        layers: list[nn.Module] = [SineLayer(input_dim, hidden_dim, omega_0, True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0, False))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NACLayer(nn.Module):
    """Neural accumulator layer from the NAC/NALU family."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self.m_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.w_hat)
        nn.init.xavier_uniform_(self.m_hat)

    def weight(self) -> torch.Tensor:
        return torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight()


class NALULayer(nn.Module):
    """Neural arithmetic logic unit layer."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-7) -> None:
        super().__init__()
        self.nac = NACLayer(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        additive = self.nac(x)
        multiplicative = torch.exp(torch.log(torch.abs(x) + self.eps) @ self.nac.weight())
        gate = torch.sigmoid(self.gate(x))
        return gate * additive + (1.0 - gate) * multiplicative


class ArithmeticNetwork(nn.Module):
    """Stacked NAC or NALU layers for numerical extrapolation baselines."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
        layer_type: str = "nac",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("ArithmeticNetwork requires depth >= 1")
        layer_cls = NACLayer if layer_type == "nac" else NALULayer
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(layer_cls(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        layers.append(layer_cls(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class IterativeRecurrentModel(nn.Module):
    """Generic RNN/GRU/LSTM control indexed by raw n.

    The model receives n only as the number of recurrent iterations. Each step
    consumes the same learned input token; no Fibonacci state is provided.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 8,
        cell_type: str = "gru",
    ) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError("IterativeRecurrentModel currently supports scalar output")
        self.cell_type = cell_type
        self.input_token = nn.Parameter(torch.randn(1) * 0.05)
        self.initial_state = nn.Parameter(torch.randn(hidden_dim) * 0.05)
        if cell_type == "rnn":
            self.cell: nn.Module = nn.RNNCell(1, hidden_dim)
        elif cell_type == "gru":
            self.cell = nn.GRUCell(1, hidden_dim)
        elif cell_type == "lstm":
            self.cell = nn.LSTMCell(1, hidden_dim)
            self.initial_cell = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise ValueError(f"unknown recurrent cell type: {cell_type}")
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = torch.round(x[:, 0]).to(torch.long)
        if torch.any(n < 0):
            raise ValueError("IterativeRecurrentModel expects non-negative n")
        max_n = int(n.max().item()) if n.numel() else 0
        h = self.initial_state.to(dtype=x.dtype, device=x.device).unsqueeze(0)
        token = self.input_token.to(dtype=x.dtype, device=x.device).reshape(1, 1)
        states = [h.squeeze(0)]
        if self.cell_type == "lstm":
            c = self.initial_cell.to(dtype=x.dtype, device=x.device).unsqueeze(0)
            for _ in range(max_n):
                h, c = self.cell(token, (h, c))
                states.append(h.squeeze(0))
        else:
            for _ in range(max_n):
                h = self.cell(token, h)
                states.append(h.squeeze(0))
        state_table = torch.stack(states, dim=0)
        outputs = self.readout(state_table)
        return outputs.index_select(0, n)


class LinearRecurrenceModel(nn.Module):
    """A generic learned linear dynamical system indexed by raw n.

    The model receives only n, then generates a scalar sequence by repeatedly
    applying a learned affine state transition. It has state and iteration, but
    no Fibonacci-specific constants or hand-coded recurrence.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 2,
        transition_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError("LinearRecurrenceModel currently supports scalar output")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.initial_state = nn.Parameter(torch.randn(hidden_dim) * transition_scale)
        transition = torch.eye(hidden_dim) + torch.randn(hidden_dim, hidden_dim) * transition_scale
        self.transition = nn.Parameter(transition)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = torch.round(x[:, 0]).to(torch.long)
        if torch.any(n < 0):
            raise ValueError("LinearRecurrenceModel expects non-negative n")
        max_n = int(n.max().item()) if n.numel() else 0
        h = self.initial_state.to(dtype=x.dtype, device=x.device)
        transition = self.transition.to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device)
        states = [h]
        for _ in range(max_n):
            h = transition @ h + bias
            states.append(h)
        state_table = torch.stack(states, dim=0)
        outputs = self.readout(state_table)
        return outputs.index_select(0, n)


def build_pykan(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    mult_nodes: int,
    grid: int,
    k: int,
    grid_min: float,
    grid_max: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Build a PyKAN classifier head if the optional dependency is installed."""
    try:
        from kan import MultKAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyKAN is not installed. Install it with `python -m pip install pykan` "
            "or run `--model mlp`."
        ) from exc

    if mult_nodes > 0:
        width: list[int | list[int]] = [input_dim, [hidden_dim, mult_nodes], output_dim]
    else:
        width = [input_dim, hidden_dim, output_dim]

    model = MultKAN(
        width=width,
        grid=grid,
        k=k,
        mult_arity=2,
        grid_range=[grid_min, grid_max],
        seed=seed,
        symbolic_enabled=False,
        save_act=False,
        auto_save=False,
        device=str(device),
    )
    if hasattr(model, "speed"):
        model.speed()
    return model


def build_efficient_kan(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    grid: int,
    k: int,
    grid_min: float,
    grid_max: float,
    device: torch.device,
) -> nn.Module:
    """Build a memory-efficient KAN implementation from Blealtan/efficient-kan."""
    try:
        from efficient_kan import KAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "efficient-kan is not installed. Install it with "
            "`python -m pip install git+https://github.com/Blealtan/efficient-kan.git`."
        ) from exc

    return KAN(
        [input_dim, hidden_dim, output_dim],
        grid_size=grid,
        spline_order=k,
        grid_range=[grid_min, grid_max],
    ).to(device)


def build_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation: str,
    kan_grid: int,
    kan_k: int,
    kan_grid_min: float,
    kan_grid_max: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Build a model by name."""
    if model_name == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
        ).to(device)

    if model_name == "gated_mlp":
        return MultiplicativeMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            residual_gated=True,
        ).to(device)

    if model_name == "multiplicative_mlp":
        return MultiplicativeMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            residual_gated=False,
        ).to(device)

    if model_name == "fourier_mlp":
        return FourierFeatureMLP(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            num_frequencies=16,
            max_frequency=16.0,
        ).to(device)

    if model_name == "siren":
        return SirenMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(device)

    if model_name == "nac":
        return ArithmeticNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            layer_type="nac",
        ).to(device)

    if model_name == "nalu":
        return ArithmeticNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            layer_type="nalu",
        ).to(device)

    if model_name == "iter_rnn":
        return IterativeRecurrentModel(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cell_type="rnn",
        ).to(device)

    if model_name == "iter_gru":
        return IterativeRecurrentModel(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cell_type="gru",
        ).to(device)

    if model_name == "iter_lstm":
        return IterativeRecurrentModel(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cell_type="lstm",
        ).to(device)

    if model_name == "linear_recurrence":
        return LinearRecurrenceModel(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ).to(device)

    if model_name == "pykan":
        return build_pykan(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            mult_nodes=0,
            grid=kan_grid,
            k=kan_k,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
            seed=seed,
            device=device,
        ).to(device)

    if model_name == "pykan_mult":
        return build_pykan(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            mult_nodes=max(1, hidden_dim // 4),
            grid=kan_grid,
            k=kan_k,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
            seed=seed,
            device=device,
        ).to(device)

    if model_name == "efficient_kan":
        return build_efficient_kan(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            grid=kan_grid,
            k=kan_k,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
            device=device,
        )

    raise ValueError(f"unknown model: {model_name}")

from typing import Dict
import math
import torch
import torch_geometric
from diffpred.models.networks.graph.graphormer_helpers.data import GraphormerBatch


class GuidanceScheduler:
    def __init__(
        self,
        schedule: str,
        sigma_min: float,
        sigma_max: float,
        constant_guidance: float,
        max_dropout: float = 0.5,
    ):
        self.schedule = schedule
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.constant_guidance = constant_guidance
        self.max_dropout = max_dropout

    def get_guidance(
        self, sigma: float | torch.FloatTensor
    ) -> float | torch.FloatTensor:
        if self.schedule == "constant":
            return self.constant_guidance
        if self.schedule == "log_linear":
            return self._log_linear_schedule(sigma)
        if self.schedule == "cosine":
            return self._cosine_schedule(sigma)
        if self.schedule == "exponential":
            return self._exponential_schedule(sigma)
        raise ValueError(f"Schedule {self.schedule} not recognized.")

    def get_dropout_rate(self, sigma: torch.FloatTensor) -> float:
        # interpolate between no dropout and max dropout of 0.5
        if self.schedule == "log_linear":
            return self.max_dropout * (1 - self._log_linear_schedule(sigma).squeeze())
        if self.schedule == "cosine":
            return self.max_dropout * (1 - self._cosine_schedule(sigma).squeeze())
        if self.schedule == "exponential":
            return self.max_dropout * (1 - self._exponential_schedule(sigma).squeeze())
        if self.schedule == "constant":
            return self.max_dropout
        raise ValueError(f"Schedule {self.schedule} not recognized.")

    def _log_linear_schedule(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        return (torch.log(sigma) - math.log(self.sigma_min)) / (
            math.log(self.sigma_max) - math.log(self.sigma_min)
        )

    def _cosine_schedule(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        log_sigma = torch.log(sigma)
        return 0.5 * (
            1
            + torch.cos(
                (
                    1
                    + (
                        (log_sigma - math.log(self.sigma_min))
                        / (math.log(self.sigma_max) - math.log(self.sigma_min))
                    )
                )
                * math.pi
            )
        )

    def _exponential_schedule(
        self, sigma: torch.FloatTensor, T: float = 0.3
    ) -> torch.FloatTensor:
        return 1 - torch.exp(-sigma / T)


class EDM:
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        sigma_data: float | torch.FloatTensor = 0.5,
        rho: float = 7.0,
        refinement_steps: int = 50,
        guidance: float = 0.0,
        guidance_dropout: float = 0.2,
        guidance_schedule: str = "log_linear",
        device: str | None = None,
        deterministic_sampling: bool = False,
        self_conditioning: bool = False,
        **kwargs,
    ):
        for param in (sigma_min, sigma_max, sigma_data, rho, refinement_steps):
            assert param > 0, f"Parameters must be positive, got {param}"

        assert 1 <= rho <= 50, f"rho parameter must be between 1 and 50, got {rho}"

        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.guidance = guidance
        self.guidance_scheduler = GuidanceScheduler(
            schedule=guidance_schedule,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            constant_guidance=guidance,
            max_dropout=guidance_dropout,
        )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.refinement_steps = refinement_steps
        self.deterministic_sampling = deterministic_sampling
        self.self_conditioning = self_conditioning

    def guidance_dropout(self, sigma: torch.FloatTensor) -> float:
        return self.guidance_scheduler.get_dropout_rate(sigma)

    def get_loss_weights(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        """
        Loss weights for EDM. Corresponds to lambda(sigma) in the paper.
        """
        return (sigma**2 + self.sigma_data**2) / (((sigma) * self.sigma_data) ** 2)

    def sample_sigma(self, shape: torch.Size) -> torch.FloatTensor:
        # Note: this is VE sampling of noise, maybe appropriate in our setting
        return self._sample_ve(shape)

    def _sample_ve(self, shape: torch.Size) -> torch.FloatTensor:
        low, high = torch.math.log(self.sigma_min), torch.math.log(self.sigma_max)
        rnd = torch.rand(shape, device=self.device, dtype=torch.float32)
        rnd = low + rnd * (high - low)
        sigma = torch.exp(rnd)
        return sigma

    def _sample_edm(self, shape: torch.Size) -> torch.FloatTensor:
        mean, std = -1.2, 1.2
        rnd = torch.randn(shape, device=self.device, dtype=torch.float32)
        rnd = rnd * std + mean
        sigma = torch.exp(rnd)
        return sigma

    def _mask_features(
        self, x: torch.FloatTensor, dropout_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Mask out features for guidance, supporting both vector features and torch_geometric.Data objects.

        dropout_mask: torch.LongTensor
            Dropout mask for features.  For Data objects, this contains the batch index to mask.
        x: torch.FloatTensor | torch_geometric.data.Data
            Input features to condition on.
        """

        if isinstance(x, GraphormerBatch):
            dropout_idx = torch.where(dropout_mask == 1)[0].to(x.x.device)
            masked = GraphormerBatch(
                x=x.x.clone(),
                in_degree=x.in_degree.clone(),
                out_degree=x.out_degree.clone(),
                attn_bias=x.attn_bias.clone(),
                spatial_pos=x.spatial_pos.clone(),
                attn_edge_type=x.attn_edge_type.clone(),
                edge_input=x.edge_input.clone(),
            )
            if dropout_idx.numel() > 0:
                masked.x[dropout_idx] = 0
                masked.attn_edge_type[dropout_idx] = 0
                masked.edge_input[dropout_idx] = 0
            return masked

        if isinstance(x, torch_geometric.data.Data):
            dropout_idx = torch.where(dropout_mask == 1)[0].to(x.batch.device)
            x = x.clone()
            # Handle torch_geometric.Data objects
            mask = torch.isin(x.batch, dropout_idx)
            # mask out the node features
            for attr in x.node_attrs():
                if attr != "batch":
                    x[attr][mask] = 0

            # Mask edge features
            edge_mask = torch.isin(x.batch[x.edge_index[0]], dropout_idx)
            for attr in x.edge_attrs():
                if attr != "edge_index":
                    x[attr][edge_mask] = 0

            return x
        else:
            # Handle vector-valued features
            return x * (1 - dropout_mask.to(x.device).unsqueeze(-1))

    def forward(
        self,
        network,
        x: torch.FloatTensor,
        y_k: torch.FloatTensor,
        noise_level: torch.FloatTensor,
        self_cond: torch.FloatTensor | None = None,
    ):
        """
        Wraps the forward pass of the underlying neural network with the preprocessing and postprocessing steps of the EDM model.
        """

        scaled_input = self.input_scaling(noise_level) * y_k
        scaled_noise = self.noise_scaling(noise_level)
        pred = network.forward(x, scaled_input, scaled_noise, self_cond=self_cond)
        out = self.diffuser_output(pred, y_k, noise_level)
        return out

    def forward_process_sample(self, y: torch.FloatTensor, sigma: float | None = None):
        y = y.to(self.device)
        if sigma is None:
            sigma = self.sample_sigma((len(y), 1))
        else:
            assert (
                self.sigma_min <= sigma <= self.sigma_max
            ), f"Provided sigma value: {sigma} not in range [{self.sigma_min}, {self.sigma_max}]"
            sigma = torch.as_tensor(sigma, device=self.device)
            sigma = torch.ones((len(y), 1), device=self.device) * sigma
        noise = torch.randn_like(y) * sigma
        y_k = y + noise

        assert y_k.shape == y.shape, f"Shape mismatch: {y_k.shape} != {y.shape}"
        return y_k, y, sigma

    def effective_target(self, sigma, y, y_k):
        effective_target = (
            1 / self.output_scaling(sigma) * (y - self.skip_scaling(sigma) * y_k)
        )
        return effective_target

    def input_scaling(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        sigma = torch.as_tensor(sigma, device=self.device)
        return 1 / (torch.sqrt((sigma**2 + self.sigma_data**2)))

    def output_scaling(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        sigma = torch.as_tensor(sigma, device=self.device)
        return (sigma * self.sigma_data) / torch.sqrt((sigma**2 + self.sigma_data**2))

    def skip_scaling(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        sigma = torch.as_tensor(sigma, device=self.device)
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def noise_scaling(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        sigma = torch.as_tensor(sigma, device=self.device)
        return 0.25 * torch.log(sigma)

    def diffuser_output(
        self,
        net_output: torch.FloatTensor,
        input: torch.FloatTensor,
        sigma: torch.FloatTensor,
    ):
        # D(x, theta) in EDM paper
        return (
            self.output_scaling(sigma) * net_output + self.skip_scaling(sigma) * input
        )

    def get_y_pred(
        self,
        output: torch.FloatTensor,
        y_k: torch.FloatTensor,
        noise_level: torch.LongTensor,
    ):
        return output  # D(y, theta) in EDM paper predicts the source target.

    def timesteps_discretization(self, num_steps: int):
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        timesteps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        timesteps = torch.cat(
            [timesteps, torch.tensor([0.0], device=self.device)]
        ).float()

        return timesteps

    def sample(
        self,
        model,
        y_dim: int,
        x: torch.FloatTensor,
        num_samples_per_elem: int,
        keep_intermediates: bool = False,
        num_steps: int | None = None,
        deterministic = None,
        solver: str = "euler",
        y=None,
    ):
        """
        Generate samples by solving the reverse-time SDE, starting from samples drawn from the prior distribtution. Implementation drawn from the EDM accompanying code.

        z: torch.FloatTensor
            Samples from the prior distribution.
        x: torch.FloatTensor
            Input feautures to condition on.
        num_steps: int
            Number of steps for the sampling process.
        deterministic: bool
            Whether to use deterministic sampling (True) or stochastic sampling (False).
        solver: str
            Solver to use for the SDE. Options are "euler" or "heun". If "heun", an additional correction term is applied.
        y: torch.FloatTensor
            True target values, for testing purposes these can be provided.
        """
        deterministic = self.deterministic_sampling if deterministic is None else deterministic
        intermediates: Dict[int, torch.Tensor] = {}
        z = self.sample_prior(num_samples_per_elem, len(x), y_dim)
        num_steps = num_steps if num_steps is not None else self.refinement_steps

        if keep_intermediates:
            intermediates[0] = z

        # Discretization of timesteps following EDM
        timesteps = self.timesteps_discretization(num_steps)

        self_cond = torch.zeros_like(z) if self.self_conditioning else None

        # Perform sampling by stepping through the SDE/ODE
        for t in range(num_steps - 1):
            t0, t1 = timesteps[t], timesteps[t + 1]
            if not deterministic:
                z, t0 = self._increase_noise_level(z, t0, num_steps)
            z, self_cond = self.sample_step(model, z, x, t0, t1, solver, y=y, self_cond=self_cond)
            if not self.self_conditioning: # prevent propagation of self-conditioning information if not using self-conditioning
                self_cond = None
            if keep_intermediates:
                intermediates[t + 1] = z

        if keep_intermediates:
            return z, intermediates
        return z

    def _increase_noise_level(
        self,
        z: torch.FloatTensor,
        t0: float,
        num_steps: int,
        S_churn: float = 30,
        S_min: float = 0.01,
        S_max: float = 10,
        S_noise: float = 1.003,
    ):
        """
        Increase the noise level of the current sample according to the churning sampler from EDM.
        """
        gamma = (
            min(S_churn / num_steps, torch.math.sqrt(2) - 1)
            if S_min <= t0 <= S_max
            else 0
        )
        eps = torch.randn_like(z, device=self.device) * S_noise
        t0_prime = t0 + gamma * t0
        z_prime = z + torch.sqrt(t0_prime**2 - t0**2) * eps

        return z_prime, t0_prime

    def sample_step(
        self,
        model,
        z: torch.FloatTensor,
        x: torch.FloatTensor,
        t0: float,
        t1: float,
        solver: str,
        y=None,
        self_cond: torch.FloatTensor | None = None,
    ):
        """
        Perform a single step of sampling.

        z: torch.FloatTensor
            Current sample.
        t0: float
            Start time.
        t1: float
            End time.
        deterministic: bool
            Whether to use deterministic sampling or stochastic sampling.
        solver: str
            Solver to use for the SDE. Options are "euler" or "heun". If "heun", an additional correction term is applied.
        """

        # For testing purposes only, we can provide the true target y
        if y is None:
            net_out = self.forward(model, x=x, y_k=z, noise_level=t0, self_cond=self_cond)
            if (
                self.guidance
            ):  # Use unconditional output to compute unconditional score term
                batch_size = z.shape[1] if z.ndim == 3 else z.shape[0]
                unconditional_out = self.forward(
                    model,
                    x=self._mask_features(x, dropout_mask=torch.ones(batch_size)),
                    y_k=z,
                    noise_level=t0,
                    self_cond=None,
                )
                guidance_level = self.guidance_scheduler.get_guidance(t0)
                net_out = (
                    guidance_level * net_out + (1 - guidance_level) * unconditional_out
                )  # Guidance term, equivalent to doing weighting directly on the score
        else:
            net_out = (y * torch.ones_like(z)).to(self.device)

        assert net_out.shape == z.shape, f"Shape mismatch: {net_out.shape} != {z.shape}"
        neg_score = (z - net_out) / t0
        if solver == "euler":
            z = self.euler_step(neg_score, z, t0, t1)
        elif solver == "heun":
            if self.guidance:
                raise NotImplementedError("Guidance not implemented for Heun solver.")
            z = self.heun_step(neg_score, model, x, z, t0, t1, self_cond=self_cond)
        else:
            raise ValueError(
                f"Solver {solver} not recognized. Options are 'euler' or 'heun'."
            )

        return z, net_out

    def euler_step(self, score, z: torch.FloatTensor, t0: float, t1: float):
        """
        Perform a single step of sampling using the Euler-Maruyama method.

        z: torch.FloatTensor
            Current sample.
        t0: float
            Start time.
        t1: float
            End time.
        deterministic: bool
            Whether to use deterministic sampling or stochastic sampling.
        """
        return z + (t1 - t0) * score  # score is in fact negative score, t1 < t0

    def heun_step(
        self,
        score,
        model,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        t0: float,
        t1: float,
        self_cond: torch.FloatTensor | None = None,
    ):
        """
        Perform a single step of sampling using the Euler-Maruyama method.

        z: torch.FloatTensor
            Current sample.
        t0: float
            Start time.
        t1: float
            End time.
        deterministic: bool
            Whether to use deterministic sampling or stochastic sampling.
        """
        # first compute the Euler step
        step = self.euler_step(score, z, t0, t1)

        # apply the correction term
        d1 = self.forward(model, x=x, y_k=step, noise_level=t1, self_cond=self_cond)
        score1 = (step - d1) / t1
        step = z + 0.5 * (t1 - t0) * (score + score1)
        return step

    def sample_prior(self, num_samples: int, num_inputs: int, dim: int):
        """
        Sample from the prior distribution.

        num_samples: int
            Number of samples to generate.
        dim: int
            Dimensionality of the samples.
        """
        z = torch.randn(num_samples, num_inputs, dim, device=self.device)
        z = z * self.sigma_max

        return z

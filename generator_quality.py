"""Train-only generator monitoring utilities. Author: Jia Rong.

The feature discrepancy below is a ResNet18 polynomial MMD, not Inception KID
or a clinical realism assessment. Fixed samples support visual quality review.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd


def split_generator_training(frame, seed, fraction=0.15):
    """Stratify within the supplied training patients, never a final test set."""
    if not 0 < fraction < 0.5:
        raise ValueError("Invalid generator monitor fraction")
    if frame.groupby("patient_group")["label"].nunique().max() != 1:
        raise ValueError("Generator training contains inconsistent patient labels")
    rng = np.random.default_rng(seed + 271828)
    monitor_groups = set()
    groups = frame.groupby("patient_group", sort=True)["label"].first()
    for label in (0, 1):
        values = groups[groups.eq(label)].index.to_numpy()
        if len(values) < 2:
            raise ValueError("Both classes need at least two training patients")
        count = max(1, min(len(values) - 1, round(len(values) * fraction)))
        monitor_groups.update(rng.permutation(values)[:count])
    mask = frame["patient_group"].isin(monitor_groups)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def update_ema(state, model, decay, updates):
    import torch
    # Warm up EMA so its first validation epochs do not average mostly random
    # initialization. Update only after a real optimizer step.
    effective_decay = min(decay, (1 + updates) / (10 + updates))
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                state[key].mul_(effective_decay).add_(value.detach(), alpha=1 - effective_decay)
            else:
                state[key].copy_(value)


def polynomial_mmd2(left, right):
    """Unbiased degree-three MMD on unit-normalized frozen features."""
    import torch
    if min(len(left), len(right)) < 2:
        raise ValueError("MMD requires at least two samples in each population")
    left = torch.nn.functional.normalize(left.double(), dim=1)
    right = torch.nn.functional.normalize(right.double(), dim=1)
    xx, yy, xy = (left @ left.T + 1)**3, (right @ right.T + 1)**3, (left @ right.T + 1)**3
    return ((xx.sum() - xx.diag().sum()) / (len(left) * (len(left) - 1))
            + (yy.sum() - yy.diag().sum()) / (len(right) * (len(right) - 1)) - 2 * xy.mean())


class GanQualityMonitor:
    # Keep spatial intermediate features in the monitor so checkpoint selection
    # is sensitive to eye-shaped layout and local contours, not only ImageNet
    # global-average-pooled texture statistics.
    metric_name = "class_mean_multiscale_spatial_polynomial_mmd2_resnet18"

    def __init__(self, frame, transform, device, seed, per_class=64, structure_conditions=None):
        import torch
        from PIL import Image
        from torchvision.models import resnet18, ResNet18_Weights
        self.device, self.per_class = device, per_class
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        self.encoder.fc = torch.nn.Identity()
        self.encoder.requires_grad_(False)
        self.real, self.real_images, self.real_sharpness, self.noise = {}, {}, {}, {}
        self.last_generated = {}
        self.structure_conditions = structure_conditions
        if structure_conditions is not None:
            for label in (0, 1):
                conditions = structure_conditions.get(label)
                if conditions is None or conditions.ndim != 4 or conditions.shape[1] != 3:
                    raise ValueError(
                        f"Structure-conditioned GAN monitor needs [N,3,H,W] maps for class {label}"
                    )
                if len(conditions) != per_class:
                    raise ValueError(
                        f"Structure-conditioned GAN monitor needs {per_class} maps for class {label}, "
                        f"found {len(conditions)}"
                    )
                self.structure_conditions[label] = conditions.to(device)
        local_rng = torch.Generator(device=device).manual_seed(seed + 314159)
        for label in (0, 1):
            candidates = frame[frame["label"].eq(label)]
            patient_rows = [group.iloc[int(np.random.default_rng(seed + label * 1009 + i).integers(len(group)))]
                            for i, (_, group) in enumerate(candidates.groupby("patient_group", sort=True))]
            patient_frame = pd.DataFrame(patient_rows)
            subset = patient_frame.sample(n=min(per_class, len(patient_frame)), random_state=seed + label)
            if len(subset) < 2:
                raise ValueError("Generator monitor has fewer than two images in a class")
            tensors = []
            for path in subset["absolute_path"]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            values = torch.stack(tensors).to(device)
            self.real_images[label] = values[:4].cpu()
            self.real[label] = self.features(values)
            self.real_sharpness[label] = float(self.sharpness(values).mean())
            self.noise[label] = torch.randn(per_class, 100, 1, 1, generator=local_rng, device=device)

    def features(self, values):
        import torch
        values = ((values + 1) / 2).clamp(0, 1)
        values = torch.nn.functional.interpolate(values, size=(224, 224), mode="bilinear",
                                                align_corners=False, antialias=True)
        mean = values.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = values.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.no_grad():
            outputs = []
            for chunk in ((values - mean) / std).split(16):
                x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(chunk)))
                x = self.encoder.maxpool(x)
                blocks = []
                for layer, spatial_size in (
                    (self.encoder.layer1, 4),
                    (self.encoder.layer2, 2),
                    (self.encoder.layer3, 1),
                    (self.encoder.layer4, 1),
                ):
                    x = layer(x)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (spatial_size, spatial_size))
                    blocks.append(torch.nn.functional.normalize(pooled.flatten(1), dim=1))
                outputs.append(torch.cat(blocks, dim=1) / np.sqrt(4.0))
            return torch.cat(outputs)

    def sharpness(self, values):
        import torch
        values = ((values + 1) / 2).clamp(0, 1)
        values = torch.nn.functional.interpolate(values, size=(224, 224), mode="bilinear",
                                                align_corners=False, antialias=True)
        gray = values[:, 0:1] * 0.299 + values[:, 1:2] * 0.587 + values[:, 2:3] * 0.114
        kernel = gray.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
        return torch.nn.functional.conv2d(gray, kernel, padding=1).abs().flatten(1).mean(1)

    def evaluate(self, generator):
        import torch
        generator.eval()
        details, previews = {}, []
        with torch.no_grad():
            for label in (0, 1):
                samples_by_chunk = []
                for start in range(0, len(self.noise[label]), 16):
                    noise = self.noise[label][start:start + 16]
                    labels = torch.full((len(noise),), label, dtype=torch.long, device=self.device)
                    arguments = (noise, labels)
                    if self.structure_conditions is not None:
                        arguments = arguments + (self.structure_conditions[label][start:start + len(noise)],)
                    samples_by_chunk.append(generator(*arguments))
                samples = torch.cat(samples_by_chunk)
                self.last_generated[label] = samples.detach().cpu()
                features = self.features(samples)
                details[f"mmd2_class_{label}"] = float(polynomial_mmd2(self.real[label], features))
                normalized = torch.nn.functional.normalize(features, dim=1)
                details[f"feature_diversity_class_{label}"] = float(torch.pdist(normalized).mean())
                details[f"sharpness_real_class_{label}"] = self.real_sharpness[label]
                details[f"sharpness_generated_class_{label}"] = float(self.sharpness(samples).mean())
                previews.extend([self.real_images[label], samples[:4].cpu()])
        score = np.mean([details[f"mmd2_class_{label}"] for label in (0, 1)])
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite GAN quality monitor")
        return float(score), details, torch.cat(previews)


class DiffusionQualityMonitor(GanQualityMonitor):
    """Fixed-noise DDIM samples scored against patient-disjoint train monitors."""

    metric_name = "class_mean_multiscale_spatial_polynomial_mmd2_resnet18_ddim"

    def __init__(self, frame, transform, device, seed, alpha_bar, image_size=256,
                 per_class=16, sampling_steps=16, structure_conditions=None):
        import torch
        super().__init__(frame, transform, device, seed, per_class=per_class)
        self.alpha_bar = alpha_bar
        self.sampling_steps = min(int(sampling_steps), len(alpha_bar))
        self.steps = np.unique(np.linspace(0, len(alpha_bar) - 1, self.sampling_steps, dtype=int))[::-1]
        generator = torch.Generator(device=device).manual_seed(seed + 161803)
        self.noise = {label: torch.randn(per_class, 3, image_size, image_size,
                                         generator=generator, device=device)
                      for label in (0, 1)}
        self.structure_conditions = structure_conditions
        if structure_conditions is not None:
            for label in (0, 1):
                conditions = structure_conditions.get(label)
                if conditions is None or conditions.ndim != 4 or conditions.shape[1] != 3:
                    raise ValueError(
                        f"Structure-conditioned monitor needs [N,3,H,W] maps for class {label}"
                    )
                if len(conditions) != per_class:
                    raise ValueError(
                        f"Structure-conditioned monitor needs {per_class} maps for class {label}, "
                        f"found {len(conditions)}"
                    )
                self.structure_conditions[label] = conditions.to(device)
        self.last_generated = {}

    def evaluate(self, model, ema_state):
        import torch
        details, previews = {}, []
        model.eval()
        with torch.no_grad():
            for label in (0, 1):
                values = self.noise[label].clone()
                labels = torch.full((len(values),), label, dtype=torch.long, device=self.device)
                for position, step in enumerate(self.steps):
                    current = torch.full((len(values),), int(step), dtype=torch.long, device=self.device)
                    arguments = (values, current, labels)
                    if self.structure_conditions is not None:
                        arguments = arguments + (self.structure_conditions[label],)
                    predicted = torch.func.functional_call(model, ema_state, arguments)
                    previous = int(self.steps[position + 1]) if position + 1 < len(self.steps) else -1
                    alpha_previous = self.alpha_bar[previous] if previous >= 0 else values.new_tensor(1.0)
                    values = ddim_step(values, predicted, self.alpha_bar[int(step)], alpha_previous)
                values = values.clamp(-1, 1)
                self.last_generated[label] = values.detach().cpu()
                features = self.features(values)
                details[f"mmd2_class_{label}"] = float(polynomial_mmd2(self.real[label], features))
                normalized = torch.nn.functional.normalize(features, dim=1)
                details[f"feature_diversity_class_{label}"] = float(torch.pdist(normalized).mean())
                details[f"sharpness_real_class_{label}"] = self.real_sharpness[label]
                details[f"sharpness_generated_class_{label}"] = float(self.sharpness(values).mean())
                previews.extend([self.real_images[label], values[:4].cpu()])
        score = float(np.mean([details[f"mmd2_class_{label}"] for label in (0, 1)]))
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite diffusion visual-quality monitor")
        return score, details, torch.cat(previews)


def ema_model(model):
    copied = copy.deepcopy(model).eval()
    copied.requires_grad_(False)
    return copied


def ddim_step(values, predicted_noise, alpha_current, alpha_previous):
    """Deterministic DDIM (eta=0), with x0 clipping and consistent epsilon."""
    import torch
    clean = ((values - torch.sqrt(1 - alpha_current) * predicted_noise)
             / torch.sqrt(alpha_current)).clamp(-1, 1)
    epsilon = (values - torch.sqrt(alpha_current) * clean) / torch.sqrt(1 - alpha_current).clamp_min(1e-12)
    return torch.sqrt(alpha_previous) * clean + torch.sqrt(1 - alpha_previous) * epsilon

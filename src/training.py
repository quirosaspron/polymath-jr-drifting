"""Loss assembly and gradient diagnostics for the drifting experiments.

The functions here are deliberately model-agnostic: a notebook supplies its
raw scalar loss terms and their coefficients, then this module reports the
actual terms that reach the optimiser.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DEFAULT_REPRESENTATION_KEYS = ("recon", "kl", "var", "covar")


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("src.training requires PyTorch.") from exc
    return torch


def _validate_scalar_losses(losses: Mapping[str, Any]) -> dict[str, Any]:
    """Return scalar PyTorch losses after validating their common contract."""
    torch = _torch()
    if not losses:
        raise ValueError("losses must contain at least one named scalar loss.")

    validated = dict(losses)
    for name, loss in validated.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every loss name must be a non-empty string.")
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(f"losses[{name!r}] must be a scalar PyTorch tensor.")
    return validated


def _coefficient(value: Any, name: str) -> float:
    coefficient = float(value)
    if not math.isfinite(coefficient):
        raise ValueError(f"{name} must be finite.")
    return coefficient


def compose_objective(
    losses: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    *,
    representation_keys: Sequence[str] = DEFAULT_REPRESENTATION_KEYS,
    lambda_representation: float = 1.0,
    loss_scale: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Combine named scalar losses and return a differentiable total plus report.

    ``lambda_representation`` scales the VAE-side group (normally
    reconstruction, KL, variance, and covariance) relative to losses such as
    drift or adversarial matching.  ``loss_scale`` is an optional global
    multiplier; leave it at 1.0 unless deliberately testing an equivalent
    global learning-rate change.
    """
    terms = _validate_scalar_losses(losses)
    weights = {} if weights is None else dict(weights)
    representation_names = tuple(
        name for name in representation_keys if name in terms
    )
    representation_set = set(representation_names)
    lambda_representation = _coefficient(
        lambda_representation, "lambda_representation"
    )
    loss_scale = _coefficient(loss_scale, "loss_scale")

    weighted = {
        name: _coefficient(weights.get(name, 1.0), f"weights[{name!r}]") * loss
        for name, loss in terms.items()
    }
    zero = next(iter(terms.values())).new_zeros(())
    representation = sum(
        (weighted[name] for name in representation_names),
        start=zero,
    )
    nonrepresentation = sum(
        (weighted[name] for name in terms if name not in representation_set),
        start=zero,
    )
    representation_scaled = lambda_representation * representation
    unscaled_total = representation_scaled + nonrepresentation
    total = loss_scale * unscaled_total
    contributions = {
        name: (
            lambda_representation * weighted[name]
            if name in representation_set
            else weighted[name]
        )
        for name in terms
    }

    return total, {
        "raw": terms,
        "weighted": weighted,
        "contributions": contributions,
        "weights": {
            name: _coefficient(weights.get(name, 1.0), f"weights[{name!r}]")
            for name in terms
        },
        "representation_keys": representation_names,
        "lambda_representation": lambda_representation,
        "loss_scale": loss_scale,
        "representation": representation,
        "representation_scaled": representation_scaled,
        "nonrepresentation": nonrepresentation,
        "total_unscaled": unscaled_total,
        "total": total,
    }


def loss_report_values(report: Mapping[str, Any]) -> dict[str, float]:
    """Detach a report returned by :func:`compose_objective` into floats."""
    required = {"raw", "weighted", "contributions", "total"}
    missing = required.difference(report)
    if missing:
        raise ValueError(f"report is missing: {', '.join(sorted(missing))}.")

    def as_float(value: Any) -> float:
        return float(value.detach().item()) if hasattr(value, "detach") else float(value)

    values = {
        "total": as_float(report["total"]),
        "total_unscaled": as_float(report.get("total_unscaled", report["total"])),
        "representation": as_float(report.get("representation", 0.0)),
        "representation_scaled": as_float(report.get("representation_scaled", 0.0)),
        "nonrepresentation": as_float(report.get("nonrepresentation", 0.0)),
    }
    for name, value in report["raw"].items():
        values[name] = as_float(value)
        values[f"{name}_weighted"] = as_float(report["weighted"][name])
        values[f"{name}_contribution"] = as_float(report["contributions"][name])
    return values


def format_loss_report(
    report: Mapping[str, Any],
    *,
    order: Sequence[str] | None = None,
    precision: int = 6,
) -> str:
    """Format raw and actually weighted loss terms for concise epoch logging."""
    values = loss_report_values(report)
    names = tuple(report["raw"]) if order is None else tuple(order)
    lines = [f"total={values['total']:.{precision}f}"]
    if report.get("representation_keys"):
        lines.append(
            "representation="
            f"{values['representation']:.{precision}f}    "
            f"scaled={values['representation_scaled']:.{precision}f}"
        )
    for name in names:
        if name not in report["raw"]:
            continue
        lines.append(
            f"{name}={values[name]:.{precision}f}    "
            f"scaled={values[f'{name}_contribution']:.{precision}f}"
        )
    return "\n".join(lines)


def format_epoch_losses(
    losses: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    *,
    representation_keys: Sequence[str] = DEFAULT_REPRESENTATION_KEYS,
    lambda_representation: float = 1.0,
    loss_scale: float = 1.0,
    order: Sequence[str] | None = None,
    precision: int = 6,
) -> str:
    """Format averaged epoch losses using the same weighting as the objective."""
    if not losses:
        raise ValueError("losses must contain at least one value.")
    raw = {name: float(value) for name, value in losses.items()}
    weights = {} if weights is None else dict(weights)
    representation_names = tuple(name for name in representation_keys if name in raw)
    representation_set = set(representation_names)
    lambda_representation = _coefficient(
        lambda_representation, "lambda_representation"
    )
    loss_scale = _coefficient(loss_scale, "loss_scale")
    weighted = {
        name: _coefficient(weights.get(name, 1.0), f"weights[{name!r}]") * value
        for name, value in raw.items()
    }
    representation = sum(weighted[name] for name in representation_names)
    nonrepresentation = sum(
        value for name, value in weighted.items() if name not in representation_set
    )
    representation_scaled = lambda_representation * representation
    total = loss_scale * (representation_scaled + nonrepresentation)
    contributions = {
        name: (
            lambda_representation * value if name in representation_set else value
        )
        for name, value in weighted.items()
    }
    return format_loss_report(
        {
            "raw": raw,
            "weighted": weighted,
            "contributions": contributions,
            "representation_keys": representation_names,
            "representation": representation,
            "representation_scaled": representation_scaled,
            "nonrepresentation": nonrepresentation,
            "total_unscaled": representation_scaled + nonrepresentation,
            "total": total,
        },
        order=order,
        precision=precision,
    )


def component_parameter_groups(model: Any) -> dict[str, list[Any]]:
    """Group common VAE/generator parameters without double-counting them."""
    def collect(*names: str) -> list[Any]:
        parameters: list[Any] = []
        seen: set[int] = set()
        for owner in (model, getattr(model, "vae", None)):
            if owner is None:
                continue
            for name in names:
                module = getattr(owner, name, None)
                if module is None or not hasattr(module, "parameters"):
                    continue
                for parameter in module.parameters():
                    if parameter.requires_grad and id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))
        return parameters

    groups = {
        "encoder": collect("encoder", "fc_mu", "fc_var"),
        "decoder": collect("decoder_input", "decoder"),
        "generator": collect("generator", "generator_model", "embedding", "label_embedding"),
    }
    assigned = {id(parameter) for group in groups.values() for parameter in group}
    other = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    if other:
        groups["other"] = other
    return groups


def gradient_report(
    loss: Any,
    model: Any,
    tolerance: float = 1e-12,
) -> dict[str, dict[str, float | int]]:
    """Report which model components receive gradients from one scalar loss."""
    torch = _torch()
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("loss must be a scalar PyTorch tensor.")
    if not loss.requires_grad:
        raise ValueError("loss must require gradients; use it before detaching it.")

    report: dict[str, dict[str, float | int]] = {}
    for name, parameters in component_parameter_groups(model).items():
        if not parameters:
            report[name] = {
                "total_norm": 0.0,
                "nonzero_parameters": 0,
                "parameter_count": 0,
            }
            continue
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        norms = [
            0.0 if gradient is None else float(gradient.detach().norm().item())
            for gradient in gradients
        ]
        report[name] = {
            "total_norm": float(math.sqrt(sum(norm * norm for norm in norms))),
            "nonzero_parameters": int(sum(norm > tolerance for norm in norms)),
            "parameter_count": len(parameters),
        }
    return report


def per_loss_gradient_norms(
    losses: Mapping[str, Any],
    model: Any,
    weights: Mapping[str, float] | None = None,
    *,
    representation_keys: Sequence[str] = DEFAULT_REPRESENTATION_KEYS,
    lambda_representation: float = 1.0,
    loss_scale: float = 1.0,
    include_total: bool = True,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return component-wise gradients for each *weighted* loss contribution."""
    total, report = compose_objective(
        losses,
        weights,
        representation_keys=representation_keys,
        lambda_representation=lambda_representation,
        loss_scale=loss_scale,
    )
    output = {
        name: gradient_report(contribution, model)
        for name, contribution in report["contributions"].items()
    }
    if include_total:
        output["total"] = gradient_report(total, model)
    return output


def set_requires_grad(module: Any, requires_grad: bool) -> None:
    """Temporarily freeze or unfreeze every parameter in a PyTorch module."""
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


__all__ = [
    "DEFAULT_REPRESENTATION_KEYS",
    "compose_objective",
    "loss_report_values",
    "format_loss_report",
    "format_epoch_losses",
    "component_parameter_groups",
    "gradient_report",
    "per_loss_gradient_norms",
    "set_requires_grad",
]

from __future__ import annotations

import warnings

import optuna
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


def crear_sampler(modo: str, seed: int | None, n_trials: int) -> BaseSampler:
    modo_normalizado = str(modo).upper()
    if modo_normalizado == "QMC":
        return _qmc(seed)
    if modo_normalizado == "TPE":
        return _tpe(seed)
    if modo_normalizado == "HYBRID":
        return HybridSampler(seed=seed, split=max(1, n_trials // 2))

    raise ValueError(f"Sampler no soportado: {modo!r}.")


class HybridSampler(BaseSampler):
    """
    QMC en la primera mitad de trials y TPE en la segunda.

    Optuna no cambia de sampler automaticamente dentro de un estudio, asi que
    esta clase delega cada llamada al sampler que corresponde segun trial.number.
    """

    def __init__(self, *, seed: int | None, split: int) -> None:
        self.split = int(split)
        self.qmc = _qmc(seed)
        self.tpe = _tpe(seed)

    def _sampler(self, trial: FrozenTrial) -> BaseSampler:
        return self.qmc if int(trial.number) < self.split else self.tpe

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return self._sampler(trial).infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, object]:
        return self._sampler(trial).sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> object:
        return self._sampler(trial).sample_independent(
            study,
            trial,
            param_name,
            param_distribution,
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        sampler = self._sampler(trial)
        before = getattr(sampler, "before_trial", None)
        if before is not None:
            before(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: optuna.trial.TrialState,
        values: list[float] | None,
    ) -> None:
        sampler = self._sampler(trial)
        after = getattr(sampler, "after_trial", None)
        if after is not None:
            after(study, trial, state, values)

    def reseed_rng(self) -> None:
        self.qmc.reseed_rng()
        self.tpe.reseed_rng()


def _qmc(seed: int | None) -> BaseSampler:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return optuna.samplers.QMCSampler(seed=seed, scramble=True)


def _tpe(seed: int | None) -> BaseSampler:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)

# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from functools import cached_property, partial
from multiprocessing import cpu_count
from typing import Any, Callable, NamedTuple, TypeVar

import numpy as np
import scipy
from jax import hessian, jit, value_and_grad, vmap
from jax import numpy as jnp
from jax.lax import select
from jax.tree_util import Partial
from jaxtyping import Array, Float, Integer
from numpy import typing as npt
from tqdm.auto import tqdm

from ..eig import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
from ..utils import Pool
from .base import NullModelCollection, NullModelResult

terms_count = TypeVar("terms_count")


class OptimizeInput(NamedTuple):
    eigenvalues: Float[Array, " sample_count"]
    rotated_covariates: Float[Array, " sample_count covariate_count"]
    rotated_phenotype: Float[Array, " sample_count 1"]


class OptimizeJob(NamedTuple):
    phenotype_index: int
    num_nested_threads: int
    optimize_input: OptimizeInput


class OptimizeResult(NamedTuple):
    x: npt.NDArray[np.float64]
    fun: float


class RegressionWeights(NamedTuple):
    regression_weights: Float[Array, " covariate_count 1"]
    scaled_residuals: Float[Array, " sample_count 1"]
    variance: Float[Array, " sample_count 1"]
    inverse_variance: Float[Array, " sample_count 1"]
    scaled_covariates: Float[Array, " sample_count covariate_count"]
    scaled_phenotype: Float[Array, " sample_count 1"]


class MinusTwoLogLikelihoodTerms(NamedTuple):
    sample_count: Integer[Array, ""]
    genetic_variance: Float[Array, ""]
    logarithmic_determinant: Float[Array, ""]
    deviation: Float[Array, ""]
    r: RegressionWeights


class StandardErrors(NamedTuple):
    regression_weights: Float[Array, " covariate_count 1"]
    standard_errors: Float[Array, " covariate_count 1"]
    scaled_residuals: Float[Array, " sample_count 1"]
    variance: Float[Array, " sample_count 1"]


@dataclass
class ProfileMaximumLikelihood:
    sample_count: int
    covariate_count: int

    minimum_variance: float = 1e-4
    maximum_variance_multiplier: float = 2.0

    grid_search_size: int = 100

    enable_softplus_penalty: bool = True
    softplus_beta: Float[Array, ""] = field(default_factory=partial(jnp.asarray, 10000))

    def get_initial_terms(self, o: OptimizeInput) -> list[float]:
        variance: float = o.rotated_phenotype.var().item()
        return [variance / 2] * 2

    def grid_search(self, o: OptimizeInput) -> npt.NDArray[np.float64]:
        variance: Float[Array, ""] = o.rotated_phenotype.var()

        variance_ratios = jnp.linspace(0.01, 0.99, self.grid_search_size)
        variances = np.linspace(
            self.minimum_variance,
            variance * self.maximum_variance_multiplier,
            self.grid_search_size,
        )
        grid = jnp.meshgrid(variance_ratios, variances)

        combinations = jnp.vstack([m.ravel() for m in grid]).transpose()
        genetic_variance = (1 - combinations[:, 0]) * combinations[:, 1]
        error_variance = combinations[:, 0] * combinations[:, 1]

        terms_grid = jnp.vstack([error_variance, genetic_variance]).transpose()
        wrapper = vmap(Partial(self.minus_two_log_likelihood, o=o))

        minus_two_log_likelihoods = wrapper(jnp.asarray(terms_grid))
        i = jnp.argmin(minus_two_log_likelihoods)

        return np.asarray(terms_grid[i, :])

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        variance: float = o.rotated_phenotype.var().item()
        return [
            (self.minimum_variance, variance * self.maximum_variance_multiplier),
            (0, variance * self.maximum_variance_multiplier),
        ]

    @cached_property
    def func_with_grad(
        self,
    ) -> Callable[
        [Float[Array, " terms_count"], OptimizeInput],
        tuple[Float[Array, ""], Float[Array, " terms_count"]],
    ]:
        return jit(value_and_grad(self.minus_two_log_likelihood))

    @cached_property
    def hessian(
        self,
    ) -> Callable[
        [Float[Array, " terms_count"], OptimizeInput],
        tuple[Float[Array, " terms_count terms_count"]],
    ]:
        func = hessian(self.minus_two_log_likelihood)
        return jit(func)

    @staticmethod
    def get_regression_weights(
        terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> RegressionWeights:
        genetic_variance = terms[1]
        error_variance = terms[0]
        variance = (genetic_variance * o.eigenvalues + error_variance)[:, jnp.newaxis]
        inverse_variance = jnp.pow(variance, -0.5)

        scaled_covariates = o.rotated_covariates * inverse_variance
        scaled_phenotype = o.rotated_phenotype * inverse_variance

        regression_weights, _, _, _ = jnp.linalg.lstsq(
            scaled_covariates, scaled_phenotype, rcond=None
        )
        scaled_residuals = scaled_phenotype - scaled_covariates @ regression_weights

        return RegressionWeights(
            regression_weights=regression_weights,
            scaled_residuals=scaled_residuals,
            variance=variance,
            inverse_variance=inverse_variance,
            scaled_covariates=scaled_covariates,
            scaled_phenotype=scaled_phenotype,
        )

    @classmethod
    def get_standard_errors(
        cls, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> StandardErrors:
        r = cls.get_regression_weights(terms, o)

        degrees_of_freedom = r.scaled_covariates.shape[0] - r.scaled_covariates.shape[1]
        residual_variance = jnp.square(r.scaled_residuals).sum() / degrees_of_freedom

        inverse_covariance = jnp.linalg.inv(
            r.scaled_covariates.transpose() @ r.scaled_covariates
        )
        standard_errors = residual_variance * jnp.sqrt(jnp.diagonal(inverse_covariance))
        standard_errors = jnp.reshape(standard_errors, (-1, 1))

        return StandardErrors(
            regression_weights=r.regression_weights,
            standard_errors=standard_errors,
            scaled_residuals=r.scaled_residuals,
            variance=r.variance,
        )

    @classmethod
    def terms_to_tensor(
        cls, numpy_terms: list[float] | npt.NDArray[np.float64]
    ) -> Float[Array, " terms_count"]:
        terms = jnp.asarray(numpy_terms)
        terms = jnp.where(jnp.isfinite(terms), terms, 0.0)
        return terms

    def wrapper_with_grad(
        self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput
    ) -> tuple[float, npt.NDArray[np.float64]]:
        try:
            terms = self.terms_to_tensor(numpy_terms)
            value, grad = self.func_with_grad(terms, o)
            return value.item(), np.asarray(grad)
        except RuntimeError:
            return np.nan, np.full_like(numpy_terms, np.nan)

    def hessian_wrapper(
        self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput
    ) -> npt.NDArray[np.float64]:
        terms = self.terms_to_tensor(numpy_terms)
        hess = self.hessian(terms, o)
        return np.asarray(hess)

    @staticmethod
    def get_heritability(
        terms: Float[Array, " terms_count"] | npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        genetic_variance = float(terms[1])
        error_variance = float(terms[0])
        heritability = float(genetic_variance / (genetic_variance + error_variance))
        return heritability, genetic_variance, error_variance

    def optimize(
        self,
        o: OptimizeInput,
        method: str = "L-BFGS-B",
        enable_hessian: bool = False,
        disp: bool = False,
    ) -> OptimizeResult:
        init = self.grid_search(o)
        bounds = self.bounds(o)

        minimizer_kwargs: dict[str, Any] = dict(
            method=method,
            jac=True,
            bounds=bounds,
            args=(o,),
            # options=dict(disp=disp),
        )
        if enable_hessian:
            minimizer_kwargs.update(dict(hess=self.hessian_wrapper))
        optimize_result = scipy.optimize.basinhopping(
            self.wrapper_with_grad,
            init,
            minimizer_kwargs=minimizer_kwargs,
            stepsize=float(init.mean()) / 8,
            niter=2**10,
            niter_success=2**4,
            disp=disp,
        )

        return OptimizeResult(x=optimize_result.x, fun=optimize_result.fun)

    def apply(self, optimize_job: OptimizeJob) -> tuple[int, NullModelResult]:
        phenotype_index = optimize_job.phenotype_index
        o = optimize_job.optimize_input
        try:
            optimize_result = self.optimize(o)
            terms = jnp.asarray(optimize_result.x)
            se = self.get_standard_errors(terms, o)
            minus_two_log_likelihood = float(optimize_result.fun)
            null_model_result = NullModelResult(
                -0.5 * minus_two_log_likelihood,
                *self.get_heritability(terms),
                np.asarray(se.regression_weights),
                np.asarray(se.standard_errors),
                np.asarray(se.scaled_residuals),
                np.asarray(se.variance),
            )
            return phenotype_index, null_model_result
        except Exception as exception:
            logger.error(
                f"Failed to fit null model for phenotype {phenotype_index}",
                exc_info=exception,
            )
            return phenotype_index, NullModelResult(
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )

    @classmethod
    def fit(
        cls,
        eig: Eigendecomposition,
        vc: VariableCollection,
        nm: NullModelCollection,
        num_threads: int = cpu_count(),
    ) -> None:
        eigenvectors = eig.eigenvectors
        covariates = vc.covariates.to_numpy().copy()
        phenotypes = vc.phenotypes.to_numpy()

        # Subtract column mean from covariates (except intercept).
        covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

        # Rotate covariates and phenotypes.
        eigenvalues = jnp.asarray(eig.eigenvalues)
        rotated_covariates = jnp.asarray(eigenvectors.transpose() @ covariates)
        rotated_phenotypes = jnp.asarray(eigenvectors.transpose() @ phenotypes)

        ml = cls(vc.sample_count, vc.covariate_count)

        # Fit null model for each phenotype.
        num_processes = min(num_threads, vc.phenotype_count)
        num_nested_threads = num_threads // num_processes
        optimize_jobs = list(
            OptimizeJob(
                phenotype_index,
                num_nested_threads,
                OptimizeInput(
                    eigenvalues,
                    rotated_covariates,
                    rotated_phenotypes[:, phenotype_index, np.newaxis],
                ),
            )
            for phenotype_index in range(vc.phenotype_count)
        )

        with Pool(processes=num_processes) as pool:
            for i, r in tqdm(
                pool.imap_unordered(ml.apply, optimize_jobs),
                desc="fitting null models",
                unit="phenotypes",
                total=vc.phenotype_count,
            ):
                nm.put(i, r)

    def softplus_penalty(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        maximum_variance = o.rotated_phenotype.var() * jnp.asarray(
            self.maximum_variance_multiplier
        )
        upper_penalty = softplus(
            terms[:2] - maximum_variance,
            beta=self.softplus_beta,
        )
        lower_penalty = softplus(
            -terms[:2],
            beta=self.softplus_beta,
        )
        penalty = jnp.asarray(self.softplus_beta) * (
            lower_penalty.sum() + upper_penalty.sum()
        )
        return penalty

    def get_minus_two_log_likelihood_terms(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> MinusTwoLogLikelihoodTerms:
        sample_count = jnp.asarray(self.sample_count)
        genetic_variance = terms[1]
        r = self.get_regression_weights(terms, o)

        logarithmic_determinant = jnp.log(r.variance).sum()
        deviation = jnp.square(r.scaled_residuals).sum()

        return MinusTwoLogLikelihoodTerms(
            sample_count=sample_count,
            genetic_variance=genetic_variance,
            logarithmic_determinant=logarithmic_determinant,
            deviation=deviation,
            r=r,
        )

    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, ""]:
        t = self.get_minus_two_log_likelihood_terms(terms, o)

        minus_two_log_likelihood = t.logarithmic_determinant + t.deviation

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return jnp.where(
            jnp.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            jnp.inf,
        )


threshold = jnp.asarray(20.0)


def softplus(x: Float[Array, ""], beta: Float[Array, ""]) -> Float[Array, ""]:
    if beta is None:
        beta = jnp.asarray(1.0)
    # Taken from https://github.com/google/jax/issues/18443 and
    # mirroring the pytorch implementation at
    # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    x_safe = select(x * beta < threshold, x, jnp.ones_like(x))
    return select(
        x * beta < threshold,
        1 / beta * jnp.log(1 + jnp.exp(beta * x_safe)),
        x,
    )


def logdet(a: Float[Array, " n n"]) -> Float[Array, ""]:
    """A re-implementation of torch.logdet that returns infinity instead of NaN, which
    prevents an error in autodiff.

    Args:
        a (Float[Array, "..."]): _description_

    Returns:
        _type_: _description_
    """
    sign, logabsdet = jnp.linalg.slogdet(a)
    assert isinstance(sign, jnp.ndarray)
    assert isinstance(logabsdet, jnp.ndarray)
    logdet = jnp.where(sign == -1.0, jnp.inf, logabsdet)
    return logdet


@dataclass
class RestrictedMaximumLikelihood(ProfileMaximumLikelihood):
    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        t = self.get_minus_two_log_likelihood_terms(terms, o)
        penalty = logdet(t.r.scaled_covariates.transpose() @ t.r.scaled_covariates)
        deviation = (t.r.scaled_phenotype * t.r.scaled_residuals).sum()
        minus_two_log_likelihood = t.logarithmic_determinant + deviation + penalty

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return jnp.where(
            jnp.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            jnp.inf,
        )


@dataclass
class MaximumPenalizedLikelihood(ProfileMaximumLikelihood):
    """
    - Chung, Y., Rabe-Hesketh, S., Dorie, V., Gelman, A., & Liu, J. (2013).
      A nondegenerate penalized likelihood estimator for variance parameters in
      multilevel models.
      Psychometrika, 78, 685-709.
    - Chung, Y., Rabe-Hesketh, S., & Choi, I. H. (2013).
      Avoiding zero between-study variance estimates in random-effects meta-analysis.
      Statistics in medicine, 32(23), 4071-4089.
    - Chung, Y., Rabe-Hesketh, S., Gelman, A., Liu, J., & Dorie, V. (2012).
      Avoiding boundary estimates in linear mixed models through weakly informative
      priors.
    """

    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        penalty = -2 * jnp.log(terms).sum()
        return super().minus_two_log_likelihood(terms, o) + penalty


@dataclass
class MaximumLikelihood(ProfileMaximumLikelihood):
    def get_initial_terms(self, o: OptimizeInput) -> list[float]:
        terms = super().get_initial_terms(o)
        r = super().get_regression_weights(
            self.terms_to_tensor(terms),
            o,
        )
        regression_weights = list(np.asarray(r.regression_weights).ravel())
        return terms + regression_weights

    def grid_search(self, o: OptimizeInput) -> npt.NDArray[np.float64]:
        pml = ProfileMaximumLikelihood(**vars(self))
        terms = pml.grid_search(o)
        r = pml.get_regression_weights(
            self.terms_to_tensor(terms),
            o,
        )
        regression_weights = np.asarray(r.regression_weights).ravel()
        return np.hstack([terms, regression_weights])

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        return super().bounds(o) + [(-np.inf, np.inf)] * self.covariate_count

    @staticmethod
    def get_regression_weights(
        terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> RegressionWeights:
        terms = jnp.where(
            jnp.isfinite(terms),
            terms,
            0,
        )

        variance = terms[1] * o.eigenvalues + terms[0]
        inverse_variance = jnp.pow(variance, -0.5)[:, jnp.newaxis]

        scaled_covariates = o.rotated_covariates * inverse_variance
        scaled_phenotype = o.rotated_phenotype * inverse_variance

        regression_weights = terms[2:]
        regression_weights = jnp.reshape(regression_weights, (-1, 1))
        scaled_residuals = scaled_phenotype - scaled_covariates @ regression_weights
        return RegressionWeights(
            regression_weights=regression_weights,
            scaled_residuals=scaled_residuals,
            variance=variance,
            inverse_variance=inverse_variance,
            scaled_covariates=scaled_covariates,
            scaled_phenotype=scaled_phenotype,
        )

    @classmethod
    def get_standard_errors(
        cls, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> StandardErrors:
        r = cls.get_regression_weights(terms, o)

        covariance = hessian(cls.minus_two_log_likelihood)(terms, o)
        inverse_covariance = jnp.linalg.inv(covariance)
        standard_errors = jnp.sqrt(jnp.diagonal(inverse_covariance))
        standard_errors = standard_errors[2:]
        standard_errors = jnp.reshape(standard_errors, (-1, 1))

        return StandardErrors(
            regression_weights=r.regression_weights,
            standard_errors=standard_errors,
            scaled_residuals=r.scaled_residuals,
            variance=r.variance,
        )

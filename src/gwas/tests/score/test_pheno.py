# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from numpy.testing import assert_array_equal
from pytest import FixtureRequest

from .simulation import missing_value_rate

sample_count = 100
phenotype_count = 16
covariate_count = 4

samples = [str(i) for i in range(sample_count)]
permutation = np.random.permutation(sample_count)

phenotype_names = [f"phenotype_{i + 1:02d}" for i in range(phenotype_count)]
covariate_names = [f"covariate_{i + 1:02d}" for i in range(covariate_count)]


def test_pheno(
    tmp_path: Path,
    sw: SharedWorkspace,
    request: FixtureRequest,
) -> None:
    np.random.seed(47)
    allocation_names = set(sw.allocations.keys())

    phenotypes = np.random.rand(sample_count, phenotype_count)
    phenotypes[
        np.random.choice(
            a=[False, True],
            size=phenotypes.shape,
            p=[1 - missing_value_rate, missing_value_rate],
        )
    ] = np.nan
    covariates = np.random.rand(sample_count, covariate_count)

    phenotype_frame = pd.DataFrame(
        phenotypes[permutation, :],
        columns=phenotype_names,
        index=[samples[i] for i in permutation],
    )
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_frame.to_csv(
        phenotype_path, sep="\t", index=True, header=True, na_rep="n/a"
    )

    covariate_frame = pd.DataFrame(
        covariates[permutation, :],
        columns=covariate_names,
        index=[samples[i] for i in permutation],
    )
    covariate_path = tmp_path / "covariates.tsv"
    covariate_frame.to_csv(
        covariate_path, sep="\t", index=True, header=True, na_rep="n/a"
    )

    variable_collection0 = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=samples,
        missing_value_strategy="listwise_deletion",
    )
    request.addfinalizer(variable_collection0.free)
    assert_array_equal(variable_collection0.phenotypes.to_numpy(), phenotypes)
    assert_array_equal(variable_collection0.covariates.to_numpy()[:, 1:], covariates)

    truncated_samples = samples[1:]
    variable_collection1 = VariableCollection.from_txt(
        [phenotype_path],
        [covariate_path],
        sw,
        samples=truncated_samples,
        missing_value_strategy="listwise_deletion",
    )
    request.addfinalizer(variable_collection1.free)
    assert_array_equal(variable_collection1.phenotypes.to_numpy(), phenotypes[1:, :])
    assert_array_equal(
        variable_collection1.covariates.to_numpy()[:, 1:], covariates[1:, :]
    )

    truncated_phenotypes = phenotype_names[1:]
    variable_collection2 = variable_collection0.copy(
        samples=truncated_samples, phenotype_names=truncated_phenotypes
    )
    request.addfinalizer(variable_collection2.free)
    assert_array_equal(variable_collection2.phenotypes.to_numpy(), phenotypes[1:, 1:])
    assert_array_equal(
        variable_collection2.covariates.to_numpy()[:, 1:], covariates[1:, :]
    )

    new_allocation_names = {
        variable_collection0.phenotypes.name,
        variable_collection0.covariates.name,
        variable_collection1.phenotypes.name,
        variable_collection1.covariates.name,
        variable_collection2.phenotypes.name,
        variable_collection2.covariates.name,
    }
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def test_pheno_zero_variance(
    sw: SharedWorkspace,
    request: FixtureRequest,
) -> None:
    np.random.seed(47)

    allocation_names = set(sw.allocations.keys())

    phenotypes = np.random.rand(sample_count, phenotype_count)
    covariates = np.random.rand(sample_count, covariate_count)
    covariates[:, 2] = 1337

    variable_collection = VariableCollection.from_arrays(
        samples, phenotype_names, phenotypes, covariate_names, covariates, sw
    )
    request.addfinalizer(variable_collection.free)

    assert variable_collection.sample_count == sample_count
    assert variable_collection.covariates.shape == (sample_count, covariate_count)
    assert variable_collection.covariate_names == [
        "intercept",
        "covariate_01",
        "covariate_02",
        "covariate_04",
    ]

    new_allocation_names = {
        variable_collection.phenotypes.name,
        variable_collection.covariates.name,
    }
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)

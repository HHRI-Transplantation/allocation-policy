from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from allocation_policy.acuity_circles import sort_ptr


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_sort_ptr_tiers(polars_frame_function):

    frame = polars_frame_function(
        data={
            'tier': [5, 1, None, 22],
            'liver_allocation_points': [0, 0, 0, 0],
            'allocation_meld': [0, 0, 0, 0],
            'CAN_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'DON_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    expected = polars_frame_function(
        data={
            'tier': [1, 5, 22, None],
            'liver_allocation_points': [0, 0, 0, 0],
            'allocation_meld': [0, 0, 0, 0],
            'CAN_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'DON_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    assert_frame_equal(sort_ptr(frame), expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_sort_liver_allocation_points(polars_frame_function):

    frame = polars_frame_function(
        data={
            'tier': [1, 1, 1, 1],
            'liver_allocation_points': [8, 2, 10, 1],
            'allocation_meld': [0, 0, 0, 0],
            'CAN_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'DON_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    expected = polars_frame_function(
        data={
            'tier': [1, 1, 1, 1],
            'liver_allocation_points': [10, 8, 2, 1],
            'allocation_meld': [0, 0, 0, 0],
            'CAN_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'DON_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    assert_frame_equal(sort_ptr(frame), expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_sort_allocation_meld(polars_frame_function):

    frame = polars_frame_function(
        data={
            'tier': [1, 1, 1, 1],
            'liver_allocation_points': [0, 0, 0, 0],
            'allocation_meld': [15, 6, 40, 22],
            'CAN_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'DON_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    expected = polars_frame_function(
        data={
            'tier': [1, 1, 1, 1],
            'liver_allocation_points': [0, 0, 0, 0],
            'allocation_meld': [40, 22, 15, 6],
            'CAN_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'DON_ABO': ['A1B', 'A1B', 'A1B', 'A1B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    assert_frame_equal(sort_ptr(frame), expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_sort_abo_compatibility(polars_frame_function):

    frame = polars_frame_function(
        data={
            'tier': [1, 1, 1, 1],
            'liver_allocation_points': [0, 0, 0, 0],
            'allocation_meld': [0, 0, 0, 0],
            'CAN_ABO': ['B  ', 'O  ', 'O  ', 'A1B'],
            'DON_ABO': ['O  ', 'O  ', 'A2B', 'A2B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    expected = polars_frame_function(
        data={
            'tier': [1, 1, 1, 1],
            'liver_allocation_points': [0, 0, 0, 0],
            'allocation_meld': [0, 0, 0, 0],
            'CAN_ABO': ['O  ', 'A1B', 'B  ', 'O  '],
            'DON_ABO': ['O  ', 'A2B', 'O  ', 'A2B'],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [0, 0, 0, 0],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    assert_frame_equal(sort_ptr(frame), expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_sort_age(polars_frame_function):

    frame = polars_frame_function(
        data={
            'tier': [1, 1],
            'liver_allocation_points': [0, 0],
            'allocation_meld': [0, 0],
            'CAN_ABO': ['O  ', 'O  '],
            'DON_ABO': ['O  ', 'O  '],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [12*19, 12*17],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    expected = polars_frame_function(
        data={
            'tier': [1, 1],
            'liver_allocation_points': [0, 0],
            'allocation_meld': [0, 0],
            'CAN_ABO': ['O  ', 'O  '],
            'DON_ABO': ['O  ', 'O  '],
            'CAN_AGE_IN_MONTHS_AT_LISTING': [12*17, 12*19],
        },
        schema={
            'tier': pl.Int64,
            'liver_allocation_points': pl.Float64,
            'allocation_meld': pl.Int64,
            'CAN_ABO': pl.String,
            'DON_ABO': pl.String,
            'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        },
    )

    assert_frame_equal(sort_ptr(frame), expected)

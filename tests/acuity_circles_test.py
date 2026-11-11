from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from allocation_policy.acuity_circles import allocation_pathway
from allocation_policy.acuity_circles import dummy_policy
from allocation_policy.acuity_circles import policy_9_8_E
from allocation_policy.acuity_circles import policy_9_8_F
from allocation_policy.acuity_circles import policy_9_8_G
from allocation_policy.acuity_circles import policy_9_8_H


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_dummy_policy(polars_frame_function):

    frame = polars_frame_function()
    expected = polars_frame_function(schema={'tier': pl.Int64})

    assert_frame_equal(dummy_policy(frame), expected)


@pytest.mark.parametrize(
    ('don_non_hr_beat', 'don_age_in_months', 'expected'),
    (
        ('N', 12*18, policy_9_8_E),
        ('N', 12*70 - 1, policy_9_8_E),
        (None, 12*18, policy_9_8_E),
        (None, 12*70 - 1, policy_9_8_E),

        ('N', 12*18 - 1, policy_9_8_F),
        ('N', 12*11, policy_9_8_F),
        (None, 12*18 - 1, policy_9_8_F),
        (None, 12*11, policy_9_8_F),

        ('N', 12*11 - 1, policy_9_8_G),
        ('N', 12*0, policy_9_8_G),
        (None, 12*11 - 1, policy_9_8_G),
        (None, 12*0, policy_9_8_G),

        ('N', 12*70, policy_9_8_H),
        (None, 12*70, policy_9_8_H),
        ('Y', 12*0, policy_9_8_H),
        ('Y', 12*5, policy_9_8_H),
        ('Y', 12*11, policy_9_8_H),
        ('Y', 12*12, policy_9_8_H),
        ('Y', 12*18, policy_9_8_H),
        ('Y', 12*70, policy_9_8_H),
        ('Y', 12*75, policy_9_8_H),

        (None, None, dummy_policy),
    ),
)
def test_allocation_pathway(
    don_non_hr_beat,
    don_age_in_months,
    expected,
):
    assert allocation_pathway(
        don_non_hr_beat,
        don_age_in_months,
    ) == expected


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_status_1(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, 500, 'NOT HI/PR', 'ANY', 'ANY', None, 1),
        (6012, 500, 'NOT HI/PR', 'ANY', 'ANY', None, 2),
        (6011, 500, 'HI', 'ANY', 'ANY', None, 1),
        (6012, 500, 'PR', 'ANY', 'ANY', None, 2),
        (6011, 2400, 'HI', 'ANY', 'ANY', None, 3),
        (6011, 1100, 'PR', 'ANY', 'ANY', None, 3),
        (6012, 2400, 'HI', 'ANY', 'ANY', None, 4),
        (6012, 1100, 'PR', 'ANY', 'ANY', None, 4),
        (6011, 501, 'NOT HI/PR', 'ANY', 'ANY', None, 34),
        (6012, 501, 'NOT HI/PR', 'ANY', 'ANY', None, 35),
        (6011, 2401, 'HI', 'ANY', 'ANY', None, 34),
        (6011, 1101, 'PR', 'ANY', 'ANY', None, 34),
        (6012, 2401, 'HI', 'ANY', 'ANY', None, 35),
        (6012, 1101, 'PR', 'ANY', 'ANY', None, 35),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_meld_at_least_37_within_500_nm(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'NOT HI/PR', 'O  ', 'O  ', 37, 5),
        (None, 150, 'NOT HI/PR', 'O  ', 'B  ', 37, 5),
        (None, 150, 'NOT HI/PR', 'A  ', 'ANY', 37, 6),
        (None, 250, 'NOT HI/PR', 'O  ', 'O  ', 37, 7),
        (None, 250, 'NOT HI/PR', 'O  ', 'B  ', 37, 7),
        (None, 250, 'NOT HI/PR', 'A  ', 'ANY', 37, 8),
        (None, 500, 'NOT HI/PR', 'O  ', 'O  ', 37, 9),
        (None, 500, 'NOT HI/PR', 'O  ', 'B  ', 37, 9),
        (None, 500, 'NOT HI/PR', 'A  ', 'ANY', 37, 10),

        (None, 150, 'HI', 'O  ', 'O  ', 37, 5),
        (None, 150, 'HI', 'O  ', 'B  ', 37, 5),
        (None, 150, 'HI', 'A  ', 'ANY', 37, 6),
        (None, 250, 'HI', 'O  ', 'O  ', 37, 7),
        (None, 250, 'HI', 'O  ', 'B  ', 37, 7),
        (None, 250, 'HI', 'A  ', 'ANY', 37, 8),
        (None, 500, 'HI', 'O  ', 'O  ', 37, 9),
        (None, 500, 'HI', 'O  ', 'B  ', 37, 9),
        (None, 500, 'HI', 'A  ', 'ANY', 37, 10),

        (None, 150, 'PR', 'O  ', 'O  ', 37, 5),
        (None, 150, 'PR', 'O  ', 'B  ', 37, 5),
        (None, 150, 'PR', 'A  ', 'ANY', 37, 6),
        (None, 250, 'PR', 'O  ', 'O  ', 37, 7),
        (None, 250, 'PR', 'O  ', 'B  ', 37, 7),
        (None, 250, 'PR', 'A  ', 'ANY', 37, 8),
        (None, 500, 'PR', 'O  ', 'O  ', 37, 9),
        (None, 500, 'PR', 'O  ', 'B  ', 37, 9),
        (None, 500, 'PR', 'A  ', 'ANY', 37, 10),

        (None, 2400, 'HI', 'O  ', 'O  ', 37, 11),
        (None, 2400, 'HI', 'O  ', 'B  ', 37, 11),
        (None, 2400, 'HI', 'A  ', 'ANY', 37, 12),

        (None, 1100, 'PR', 'O  ', 'O  ', 37, 11),
        (None, 1100, 'PR', 'O  ', 'B  ', 37, 11),
        (None, 1100, 'PR', 'A  ', 'ANY', 37, 12),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_meld_at_least_33_within_500_nm(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'ANY', 'O  ', 'O  ', 33, 13),
        (None, 150, 'ANY', 'O  ', 'B  ', 33, 13),
        (None, 150, 'ANY', 'A  ', 'ANY', 33, 14),
        (None, 250, 'ANY', 'O  ', 'O  ', 33, 15),
        (None, 250, 'ANY', 'O  ', 'B  ', 33, 15),
        (None, 250, 'ANY', 'A  ', 'ANY', 33, 16),
        (None, 500, 'ANY', 'O  ', 'O  ', 33, 17),
        (None, 500, 'ANY', 'O  ', 'B  ', 33, 17),
        (None, 500, 'ANY', 'A  ', 'ANY', 33, 18),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_meld_at_least_29_within_500_nm(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'ANY', 'O  ', 'O  ', 30, 19),
        (None, 150, 'ANY', 'O  ', 'B  ', 30, 19),
        (None, 150, 'ANY', 'O  ', 'O  ', 29, 20),
        (None, 150, 'ANY', 'A  ', 'ANY', 29, 21),
        (None, 250, 'ANY', 'O  ', 'O  ', 30, 22),
        (None, 250, 'ANY', 'O  ', 'B  ', 30, 22),
        (None, 250, 'ANY', 'O  ', 'O  ', 29, 23),
        (None, 250, 'ANY', 'A  ', 'ANY', 29, 24),
        (None, 500, 'ANY', 'O  ', 'O  ', 30, 25),
        (None, 500, 'ANY', 'O  ', 'B  ', 30, 25),
        (None, 500, 'ANY', 'O  ', 'O  ', 29, 26),
        (None, 500, 'ANY', 'A  ', 'ANY', 29, 27),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_meld_at_least_15_within_500_nm(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'ANY', 'O  ', 'O  ', 15, 28),
        (None, 150, 'ANY', 'A  ', 'ANY', 15, 29),
        (None, 250, 'ANY', 'O  ', 'O  ', 15, 30),
        (None, 250, 'ANY', 'A  ', 'ANY', 15, 31),
        (None, 500, 'ANY', 'O  ', 'O  ', 15, 32),
        (None, 500, 'ANY', 'A  ', 'ANY', 15, 33),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_meld_at_least_15_nation_nm(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 501, 'ANY', 'O  ', 'O  ', 30, 36),
        (None, 501, 'ANY', 'O  ', 'B  ', 30, 36),
        (None, 501, 'ANY', 'O  ', 'O  ', 15, 37),
        (None, 501, 'ANY', 'A  ', 'ANY', 15, 38),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_any_meld(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'ANY', 'O  ', 'O  ', 1, 39),
        (None, 150, 'ANY', 'A  ', 'ANY', 1, 40),
        (None, 250, 'ANY', 'O  ', 'O  ', 1, 41),
        (None, 250, 'ANY', 'A  ', 'ANY', 1, 42),
        (None, 500, 'ANY', 'O  ', 'O  ', 1, 43),
        (None, 500, 'ANY', 'A  ', 'ANY', 1, 44),
        (None, 501, 'ANY', 'O  ', 'O  ', 1, 45),
        (None, 501, 'ANY', 'A  ', 'ANY', 1, 46),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_can_abo_B_cases(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'ANY', 'O  ', 'B  ', 29, 47),
        (None, 250, 'ANY', 'O  ', 'B  ', 29, 48),
        (None, 500, 'ANY', 'O  ', 'B  ', 29, 49),
        (None, 150, 'ANY', 'O  ', 'B  ', 15, 50),
        (None, 250, 'ANY', 'O  ', 'B  ', 15, 51),
        (None, 500, 'ANY', 'O  ', 'B  ', 15, 52),
        (None, 501, 'ANY', 'O  ', 'B  ', 15, 53),
        (None, 150, 'ANY', 'O  ', 'B  ', 1, 54),
        (None, 250, 'ANY', 'O  ', 'B  ', 1, 55),
        (None, 500, 'ANY', 'O  ', 'B  ', 1, 56),
        (None, 501, 'ANY', 'O  ', 'B  ', 1, 57),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_E_can_abo_A_or_AB(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 150, 'ANY', 'O  ', 'A  ', 37, 58),
        (None, 150, 'ANY', 'O  ', 'AB ', 37, 58),
        (None, 250, 'ANY', 'O  ', 'A  ', 37, 59),
        (None, 250, 'ANY', 'O  ', 'AB ', 37, 59),
        (None, 500, 'ANY', 'O  ', 'A  ', 37, 60),
        (None, 500, 'ANY', 'O  ', 'AB ', 37, 60),
        (None, 150, 'ANY', 'O  ', 'A  ', 33, 61),
        (None, 150, 'ANY', 'O  ', 'AB ', 33, 61),
        (None, 250, 'ANY', 'O  ', 'A  ', 33, 62),
        (None, 250, 'ANY', 'O  ', 'AB ', 33, 62),
        (None, 500, 'ANY', 'O  ', 'A  ', 33, 63),
        (None, 500, 'ANY', 'O  ', 'AB ', 33, 63),
        (None, 150, 'ANY', 'O  ', 'A  ', 29, 64),
        (None, 150, 'ANY', 'O  ', 'AB ', 29, 64),
        (None, 250, 'ANY', 'O  ', 'A  ', 29, 65),
        (None, 250, 'ANY', 'O  ', 'AB ', 29, 65),
        (None, 500, 'ANY', 'O  ', 'A  ', 29, 66),
        (None, 500, 'ANY', 'O  ', 'AB ', 29, 66),
        (None, 150, 'ANY', 'O  ', 'A  ', 15, 67),
        (None, 150, 'ANY', 'O  ', 'AB ', 15, 67),
        (None, 250, 'ANY', 'O  ', 'A  ', 15, 68),
        (None, 250, 'ANY', 'O  ', 'AB ', 15, 68),
        (None, 500, 'ANY', 'O  ', 'A  ', 15, 69),
        (None, 500, 'ANY', 'O  ', 'AB ', 15, 69),
        (None, 501, 'ANY', 'O  ', 'A  ', 15, 70),
        (None, 501, 'ANY', 'O  ', 'AB ', 15, 70),
        (None, 150, 'ANY', 'O  ', 'A  ', 1, 71),
        (None, 150, 'ANY', 'O  ', 'AB ', 1, 71),
        (None, 250, 'ANY', 'O  ', 'A  ', 1, 72),
        (None, 250, 'ANY', 'O  ', 'AB ', 1, 72),
        (None, 500, 'ANY', 'O  ', 'A  ', 1, 73),
        (None, 500, 'ANY', 'O  ', 'AB ', 1, 73),
        (None, 501, 'ANY', 'O  ', 'A  ', 1, 74),
        (None, 501, 'ANY', 'O  ', 'AB ', 1, 74),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_E(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_F_status_1(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18 - 1, 500, 'ANY', 'ANY', 'ANY', None, 1),
        (6011, None, 12*18, 500, 'ANY', 'ANY', 'ANY', None, 2),
        (6012, None, 12*18, 500, 'ANY', 'ANY', 'ANY', None, 3),
        (6011, None, 12*18 - 1, 500, 'HI', 'ANY', 'ANY', None, 1),
        (6011, None, 12*18, 500, 'HI', 'ANY', 'ANY', None, 2),
        (6012, None, 12*18, 500, 'HI', 'ANY', 'ANY', None, 3),
        (6011, None, 12*18 - 1, 500, 'PR', 'ANY', 'ANY', None, 1),
        (6011, None, 12*18, 500, 'PR', 'ANY', 'ANY', None, 2),
        (6012, None, 12*18, 500, 'PR', 'ANY', 'ANY', None, 3),

        (6011, None, 12*18 - 1, 2400, 'HI', 'ANY', 'ANY', None, 4),
        (6011, None, 12*18 - 1, 1100, 'PR', 'ANY', 'ANY', None, 4),
        (6011, None, 12*18, 2400, 'HI', 'ANY', 'ANY', None, 5),
        (6011, None, 12*18, 1100, 'PR', 'ANY', 'ANY', None, 5),
        (6012, None, 12*18, 2400, 'HI', 'ANY', 'ANY', None, 6),
        (6012, None, 12*18, 1100, 'PR', 'ANY', 'ANY', None, 6),

        (6011, None, 12*18 - 1, 501, 'NOT HI/PR', 'ANY', 'ANY', None, 21),
        (6011, None, 12*18 - 1, 2401, 'HI', 'ANY', 'ANY', None, 21),
        (6011, None, 12*18 - 1, 2101, 'PR', 'ANY', 'ANY', None, 21),

        (6011, None, 12*18, 501, 'NOT HI/PR', 'ANY', 'ANY', None, 22),
        (6011, None, 12*18, 2401, 'HI', 'ANY', 'ANY', None, 22),
        (6011, None, 12*18, 2101, 'PR', 'ANY', 'ANY', None, 22),

        (6012, None, 12*18, 501, 'NOT HI/PR', 'ANY', 'ANY', None, 23),
        (6012, None, 12*18, 2401, 'HI', 'ANY', 'ANY', None, 23),
        (6012, None, 12*18, 2101, 'PR', 'ANY', 'ANY', None, 23),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_F(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_F_peld(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 37, 7),
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 37, 7),
        (None, 'P', None, 500, 'ANY', 'A  ', 'ANY', 37, 8),

        (None, 'P', None, 500, 'HI', 'O  ', 'O  ', 37, 7),
        (None, 'P', None, 500, 'HI', 'O  ', 'B  ', 37, 7),
        (None, 'P', None, 500, 'HI', 'A  ', 'ANY', 37, 8),

        (None, 'P', None, 500, 'PR', 'O  ', 'O  ', 37, 7),
        (None, 'P', None, 500, 'PR', 'O  ', 'B  ', 37, 7),
        (None, 'P', None, 500, 'PR', 'A  ', 'ANY', 37, 8),

        (None, 'P', None, 2400, 'HI', 'O  ', 'O  ', 37, 9),
        (None, 'P', None, 2400, 'HI', 'O  ', 'B  ', 37, 9),
        (None, 'P', None, 2400, 'HI', 'A  ', 'ANY', 37, 10),

        (None, 'P', None, 1100, 'PR', 'O  ', 'O  ', 37, 9),
        (None, 'P', None, 1100, 'PR', 'O  ', 'B  ', 37, 9),
        (None, 'P', None, 1100, 'PR', 'A  ', 'ANY', 37, 10),

        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 30, 11),
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 30, 11),
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 29, 12),
        (None, 'P', None, 500, 'ANY', 'A  ', 'ANY', 29, 13),

        (None, 'P', None, 500, 'HI', 'O  ', 'O  ', 30, 11),
        (None, 'P', None, 500, 'HI', 'O  ', 'B  ', 30, 11),
        (None, 'P', None, 500, 'HI', 'O  ', 'O  ', 29, 12),
        (None, 'P', None, 500, 'HI', 'A  ', 'ANY', 29, 13),

        (None, 'P', None, 500, 'PR', 'O  ', 'O  ', 30, 11),
        (None, 'P', None, 500, 'PR', 'O  ', 'B  ', 30, 11),
        (None, 'P', None, 500, 'PR', 'O  ', 'O  ', 29, 12),
        (None, 'P', None, 500, 'PR', 'A  ', 'ANY', 29, 13),

        (None, 'P', None, 501, 'NOT HI/PR', 'O  ', 'O  ', 30, 24),
        (None, 'P', None, 501, 'NOT HI/PR', 'O  ', 'B  ', 30, 24),
        (None, 'P', None, 2401, 'HI', 'O  ', 'O  ', 30, 24),
        (None, 'P', None, 2401, 'HI', 'O  ', 'B  ', 30, 24),
        (None, 'P', None, 1101, 'PR', 'O  ', 'O  ', 30, 24),
        (None, 'P', None, 1101, 'PR', 'O  ', 'B  ', 30, 24),

        (None, 'P', None, 501, 'NOT HI/PR', 'O  ', 'O  ', 29, 25),
        (None, 'P', None, 2401, 'HI', 'O  ', 'O  ', 29, 25),
        (None, 'P', None, 1101, 'PR', 'O  ', 'O  ', 29, 25),

        (None, 'P', None, 501, 'NOT HI/PR', 'A  ', 'ANY', 29, 26),
        (None, 'P', None, 2401, 'HI', 'A  ', 'ANY', 29, 26),
        (None, 'P', None, 1101, 'PR', 'A  ', 'ANY', 29, 26),

        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 29, 36),
        (None, 'P', None, 500, 'HI', 'O  ', 'B  ', 29, 36),
        (None, 'P', None, 500, 'PR', 'O  ', 'B  ', 29, 36),

        (None, 'P', None, 501, 'ANY', 'O  ', 'B  ', 29, 38),
        (None, 'P', None, 501, 'HI', 'O  ', 'B  ', 29, 38),
        (None, 'P', None, 501, 'PR', 'O  ', 'B  ', 29, 38),

        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 6, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 15, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 29, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 33, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 37, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 40, 42),

        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 6, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 15, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 29, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 33, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 37, 42),
        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 40, 42),

        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 6, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 15, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 29, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 33, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 37, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 40, 44),

        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 6, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 15, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 29, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 33, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 37, 44),
        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 40, 44),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_F(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_F_pediatric_meld(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'O  ', 37, 14),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'B  ', 37, 14),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'A  ', 'ANY', 37, 15),


        (None, 'M', 12*18 - 1, 500, 'HI', 'O  ', 'O  ', 37, 14),
        (None, 'M', 12*18 - 1, 500, 'HI', 'O  ', 'B  ', 37, 14),
        (None, 'M', 12*18 - 1, 500, 'HI', 'A  ', 'ANY', 37, 15),

        (None, 'M', 12*18 - 1, 500, 'PR', 'O  ', 'O  ', 37, 14),
        (None, 'M', 12*18 - 1, 500, 'PR', 'O  ', 'B  ', 37, 14),
        (None, 'M', 12*18 - 1, 500, 'PR', 'A  ', 'ANY', 37, 15),

        (None, 'M', 12*18 - 1, 2400, 'HI', 'O  ', 'O  ', 37, 16),
        (None, 'M', 12*18 - 1, 2400, 'HI', 'O  ', 'B  ', 37, 16),
        (None, 'M', 12*18 - 1, 1100, 'PR', 'O  ', 'O  ', 37, 16),
        (None, 'M', 12*18 - 1, 1100, 'PR', 'O  ', 'B  ', 37, 16),

        (None, 'M', 12*18 - 1, 2400, 'HI', 'A  ', 'ANY', 37, 17),
        (None, 'M', 12*18 - 1, 1100, 'PR', 'A  ', 'ANY', 37, 17),

        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'O  ', 30, 18),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'B  ', 30, 18),

        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'O  ', 29, 19),

        (None, 'M', 12*18 - 1, 500, 'ANY', 'A  ', 'ANY', 29, 20),

        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'O  ', 30, 27),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'B  ', 30, 27),

        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'O  ', 29, 28),

        (None, 'M', 12*18 - 1, 501, 'ANY', 'A  ', 'ANY', 29, 29),

        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'B  ', 29, 37),

        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'B  ', 29, 39),

        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'A  ', 6, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'A  ', 15, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'A  ', 29, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'A  ', 33, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'A  ', 37, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'A  ', 40, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'AB ', 6, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'AB ', 15, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'AB ', 29, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'AB ', 33, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'AB ', 37, 43),
        (None, 'M', 12*18 - 1, 500, 'ANY', 'O  ', 'AB ', 40, 43),

        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'A  ', 6, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'A  ', 15, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'A  ', 29, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'A  ', 33, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'A  ', 37, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'A  ', 40, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'AB ', 6, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'AB ', 15, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'AB ', 29, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'AB ', 33, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'AB ', 37, 45),
        (None, 'M', 12*18 - 1, 501, 'ANY', 'O  ', 'AB ', 40, 45),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_F(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_F_adult_meld(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'O  ', 30, 30),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'B  ', 30, 30),

        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'O  ', 29, 31),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'ANY', 6, 32),
        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'ANY', 15, 32),
        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'ANY', 29, 32),
        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'ANY', 33, 32),
        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'ANY', 37, 32),
        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'ANY', 40, 32),

        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'O  ', 30, 33),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'B  ', 30, 33),

        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'O  ', 29, 34),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'ANY', 6, 35),
        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'ANY', 15, 35),
        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'ANY', 29, 35),
        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'ANY', 33, 35),
        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'ANY', 37, 35),
        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'ANY', 40, 35),

        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'B  ', 6, 40),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'B  ', 15, 40),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'B  ', 29, 40),

        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'B  ', 6, 41),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'B  ', 15, 41),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'B  ', 29, 41),

        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 6, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 15, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 29, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 33, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 37, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 40, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 6, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 15, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 29, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 33, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 37, 46),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 40, 46),

        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 6, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 15, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 29, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 33, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 37, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 40, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 6, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 15, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 29, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 33, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 37, 47),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 40, 47),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    df = expected.drop('tier')
    df = policy_9_8_F(df)

    assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_1(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'O  ', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'B  ', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'AB ', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1B', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2B', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A  ', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A1 ', 'A2B', None, 1),

        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'O  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'B  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'A  ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'A1 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'A2 ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'AB ', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'A1B', None, 1),
        (6011, None, 12*18-1, 500, 'ANY', 'A2 ', 'A2B', None, 1),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_2(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'O  ', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'B  ', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'AB ', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1B', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2B', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A  ', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A1 ', 'A2B', None, 2),

        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'O  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'B  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'A  ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'A1 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'A2 ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'AB ', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'A1B', None, 2),
        (6011, None, 12*12-1, 10000, 'ANY', 'A2 ', 'A2B', None, 2),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_3(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'O  ', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'B  ', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'B  ', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'AB ', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'AB ', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'A1B', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1B', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'A2B', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2B', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'A  ', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A  ', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A1 ', 'A2B', None, 3),

        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'O  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'B  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'A  ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'A1 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'A2 ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'AB ', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'A1B', None, 3),
        (6011, None, 12*18, 500, 'ANY', 'A2 ', 'A2B', None, 3),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_4(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A2B', None, 4),

        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'O  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'B  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A  ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A1 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A2 ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'AB ', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A1B', None, 4),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A2B', None, 4),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_5(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'O  ', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'B  ', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'AB ', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1B', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2B', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A  ', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A1 ', 'A2B', None, 5),

        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'O  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'B  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'A  ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'AB ', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'A1B', None, 5),
        (6011, None, 12*18-1, 2400, 'HI', 'A2 ', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'O  ', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'B  ', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'AB ', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1B', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2B', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A  ', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A1 ', 'A2B', None, 5),

        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'O  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'B  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'A  ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'A1 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'A2 ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'AB ', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'A1B', None, 5),
        (6011, None, 12*18-1, 1100, 'PR', 'A2 ', 'A2B', None, 5),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_6(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'O  ', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'B  ', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'B  ', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'AB ', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'AB ', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'A1B', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1B', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'A2B', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2B', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'A  ', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A  ', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A1 ', 'A2B', None, 6),

        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'O  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'B  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'A  ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'A1 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'A2 ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'AB ', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'A1B', None, 6),
        (6011, None, 12*18, 2400, 'HI', 'A2 ', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'O  ', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'O  ', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'B  ', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'B  ', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'AB ', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'AB ', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'A1B', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1B', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'A2B', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2B', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'A  ', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A  ', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A1 ', 'A2B', None, 6),

        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'O  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'B  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'A  ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'A1 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'A2 ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'AB ', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'A1B', None, 6),
        (6011, None, 12*18, 1100, 'PR', 'A2 ', 'A2B', None, 6),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_7(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'O  ', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'B  ', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'AB ', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1B', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2B', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A  ', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A1 ', 'A2B', None, 7),

        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'O  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'B  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'A  ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'AB ', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'A1B', None, 7),
        (6012, None, 12*18-1, 2400, 'HI', 'A2 ', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'O  ', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'B  ', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'AB ', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1B', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2B', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A  ', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A1 ', 'A2B', None, 7),

        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'O  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'B  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'A  ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'A1 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'A2 ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'AB ', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'A1B', None, 7),
        (6012, None, 12*18-1, 1100, 'PR', 'A2 ', 'A2B', None, 7),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_8(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 37, 8),
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 38, 8),
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 39, 8),
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 40, 8),

        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 37, 8),
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 38, 8),
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 39, 8),
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 40, 8),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_9(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 500, 'ANY', 'A  ', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A  ', 'A2B', 40, 9),

        (None, 'P', None, 500, 'ANY', 'A1 ', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2B', 40, 9),

        (None, 'P', None, 500, 'ANY', 'A2 ', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2B', 40, 9),

        (None, 'P', None, 500, 'ANY', 'AB ', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2B', 40, 9),

        (None, 'P', None, 500, 'ANY', 'A1B', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2B', 40, 9),

        (None, 'P', None, 500, 'ANY', 'A2B', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2B', 40, 9),

        (None, 'P', None, 500, 'ANY', 'B  ', 'O  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'O  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'O  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'O  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'B  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'B  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'B  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'B  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A  ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A  ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A  ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A  ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2 ', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2 ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2 ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2 ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'AB ', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'AB ', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'AB ', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1B', 40, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2B', 37, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2B', 38, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2B', 39, 9),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2B', 40, 9),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_10(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 2400, 'HI', 'O  ', 'O  ', 37, 10),
        (None, 'P', None, 2400, 'HI', 'O  ', 'O  ', 38, 10),
        (None, 'P', None, 2400, 'HI', 'O  ', 'O  ', 39, 10),
        (None, 'P', None, 2400, 'HI', 'O  ', 'O  ', 40, 10),

        (None, 'P', None, 2400, 'HI', 'O  ', 'B  ', 37, 10),
        (None, 'P', None, 2400, 'HI', 'O  ', 'B  ', 38, 10),
        (None, 'P', None, 2400, 'HI', 'O  ', 'B  ', 39, 10),
        (None, 'P', None, 2400, 'HI', 'O  ', 'B  ', 40, 10),

        (None, 'P', None, 1100, 'PR', 'O  ', 'O  ', 37, 10),
        (None, 'P', None, 1100, 'PR', 'O  ', 'O  ', 38, 10),
        (None, 'P', None, 1100, 'PR', 'O  ', 'O  ', 39, 10),
        (None, 'P', None, 1100, 'PR', 'O  ', 'O  ', 40, 10),

        (None, 'P', None, 1100, 'PR', 'O  ', 'B  ', 37, 10),
        (None, 'P', None, 1100, 'PR', 'O  ', 'B  ', 38, 10),
        (None, 'P', None, 1100, 'PR', 'O  ', 'B  ', 39, 10),
        (None, 'P', None, 1100, 'PR', 'O  ', 'B  ', 40, 10),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_11(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 2400, 'HI', 'A  ', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A  ', 'A2B', 40, 11),

        (None, 'P', None, 2400, 'HI', 'A1 ', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1 ', 'A2B', 40, 11),

        (None, 'P', None, 2400, 'HI', 'A2 ', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2 ', 'A2B', 40, 11),

        (None, 'P', None, 2400, 'HI', 'AB ', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'AB ', 'A2B', 40, 11),

        (None, 'P', None, 2400, 'HI', 'A1B', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A1B', 'A2B', 40, 11),

        (None, 'P', None, 2400, 'HI', 'A2B', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'A2B', 'A2B', 40, 11),

        (None, 'P', None, 2400, 'HI', 'B  ', 'O  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'O  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'O  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'O  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'B  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'B  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'B  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'B  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A  ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A  ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A  ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A  ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2 ', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2 ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2 ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2 ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'AB ', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'AB ', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'AB ', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A1B', 40, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2B', 37, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2B', 38, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2B', 39, 11),
        (None, 'P', None, 2400, 'HI', 'B  ', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'A  ', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A  ', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'A1 ', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1 ', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'A2 ', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2 ', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'AB ', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'AB ', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'A1B', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A1B', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'A2B', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'A2B', 'A2B', 40, 11),

        (None, 'P', None, 1100, 'PR', 'B  ', 'O  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'O  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'O  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'O  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'B  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'B  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'B  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'B  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A  ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A  ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A  ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A  ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2 ', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2 ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2 ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2 ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'AB ', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'AB ', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'AB ', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A1B', 40, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2B', 37, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2B', 38, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2B', 39, 11),
        (None, 'P', None, 1100, 'PR', 'B  ', 'A2B', 40, 11),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_12_14(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 30, 12),
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 30, 12),

        (None, 'P', None, 500, 'ANY', 'O  ', 'O  ', 6, 13),

        (None, 'P', None, 500, 'ANY', 'A  ', 'O  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'O  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'O  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'O  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'O  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'O  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'O  ', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'B  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'B  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'B  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'B  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'B  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'B  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'B  ', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'AB ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'AB ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'AB ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'AB ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'AB ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'AB ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'AB ', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'A1B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1B', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'A2B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2B', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2B', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'A  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A  ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A  ', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'A1 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A1 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A1 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A1 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A1 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A1 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A1 ', 6, 14),

        (None, 'P', None, 500, 'ANY', 'A  ', 'A2 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1 ', 'A2 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2 ', 'A2 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'AB ', 'A2 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A1B', 'A2 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'A2B', 'A2 ', 6, 14),
        (None, 'P', None, 500, 'ANY', 'B  ', 'A2 ', 6, 14),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_15_16(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'O  ', 37, 15),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'B  ', 37, 15),

        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A2B', 37, 16),

        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A2B', 37, 16),

        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A2B', 37, 16),

        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A2B', 37, 16),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A2B', 37, 16),

        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A2B', 37, 16),

        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'O  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'B  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A  ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A1 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A2 ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'AB ', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A1B', 37, 16),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A2B', 37, 16),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_17_18(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18-1, 2400, 'HI', 'O  ', 'O  ', 37, 17),
        (None, 'M', 12*18-1, 2400, 'HI', 'O  ', 'B  ', 37, 17),

        (None, 'M', 12*18-1, 1100, 'PR', 'O  ', 'O  ', 37, 17),
        (None, 'M', 12*18-1, 1100, 'PR', 'O  ', 'B  ', 37, 17),

        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'B  ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'AB ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1B', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2B', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A  ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A1 ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 2400, 'HI', 'A2 ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'B  ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'AB ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1B', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2B', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A  ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A1 ', 'A2B', 37, 18),

        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'O  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'B  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'A  ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'A1 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'A2 ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'AB ', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'A1B', 37, 18),
        (None, 'M', 12*18-1, 1100, 'PR', 'A2 ', 'A2B', 37, 18),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_19_21(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'O  ', 30, 19),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'B  ', 30, 19),

        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'O  ', 6, 20),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'O  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'O  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'O  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'O  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'O  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'O  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'O  ', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'B  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'B  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'B  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'B  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'B  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'B  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'B  ', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'AB ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'AB ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'AB ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'AB ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'AB ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'AB ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'AB ', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A1B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A1B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A1B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A1B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A1B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A1B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A1B', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A2B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A2B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A2B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A2B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A2B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A2B', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A2B', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A  ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A  ', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A1 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A1 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A1 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A1 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A1 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A1 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A1 ', 6, 21),

        (None, 'M', 12*18-1, 500, 'ANY', 'A  ', 'A2 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1 ', 'A2 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2 ', 'A2 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'AB ', 'A2 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A1B', 'A2 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'A2B', 'A2 ', 6, 21),
        (None, 'M', 12*18-1, 500, 'ANY', 'B  ', 'A2 ', 6, 21),

    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_22(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'O  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'B  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'A  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'A1 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'A2 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'AB ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'A1B', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'O  ', 'A2B', None, 22),

        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'O  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'B  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'A  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'A1 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'A2 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'AB ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'A1B', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'B  ', 'A2B', None, 22),

        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'O  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'B  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'A  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'A1 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'A2 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'AB ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'A1B', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'AB ', 'A2B', None, 22),

        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'O  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'B  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'A  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'A1 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'A2 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'AB ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'A1B', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A1B', 'A2B', None, 22),

        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'O  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'B  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'A  ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'A1 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'A2 ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'AB ', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'A1B', None, 22),
        (6011, None, 12*18-1, 501, 'ANY', 'A2B', 'A2B', None, 22),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_23(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'O  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'B  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'A  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'A1 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'A2 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'AB ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'A1B', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'O  ', 'A2B', None, 23),

        (6011, None, 12*18, 501, 'ANY', 'B  ', 'O  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'B  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'A  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'A1 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'A2 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'AB ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'A1B', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'B  ', 'A2B', None, 23),

        (6011, None, 12*18, 501, 'ANY', 'AB ', 'O  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'B  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'A  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'A1 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'A2 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'AB ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'A1B', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'AB ', 'A2B', None, 23),

        (6011, None, 12*18, 501, 'ANY', 'A1B', 'O  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'B  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'A  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'A1 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'A2 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'AB ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'A1B', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A1B', 'A2B', None, 23),

        (6011, None, 12*18, 501, 'ANY', 'A2B', 'O  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'B  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'A  ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'A1 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'A2 ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'AB ', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'A1B', None, 23),
        (6011, None, 12*18, 501, 'ANY', 'A2B', 'A2B', None, 23),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_24(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'O  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'B  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A1 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A2 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'AB ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A1B', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A2B', None, 24),

        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'O  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'B  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A1 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A2 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'AB ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A1B', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A2B', None, 24),

        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'O  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'B  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A1 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A2 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'AB ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A1B', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A2B', None, 24),

        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'O  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'B  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A1 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A2 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'AB ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A1B', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A2B', None, 24),

        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'O  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'B  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A  ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A1 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A2 ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'AB ', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A1B', None, 24),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A2B', None, 24),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_25_27(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 501, 'ANY', 'O  ', 'O  ', 30, 25),
        (None, 'P', None, 501, 'ANY', 'O  ', 'B  ', 30, 25),

        (None, 'P', None, 501, 'ANY', 'O  ', 'O  ', 6, 26),

        (None, 'P', None, 501, 'ANY', 'A  ', 'O  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'O  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'O  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'O  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'O  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'O  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'O  ', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'B  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'B  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'B  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'B  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'B  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'B  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'B  ', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'AB ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'AB ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'AB ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'AB ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'AB ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'AB ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'AB ', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'A1B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'A1B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'A1B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'A1B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'A1B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'A1B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'A1B', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'A2B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'A2B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'A2B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'A2B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'A2B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'A2B', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'A2B', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'A  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'A  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'A  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'A  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'A  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'A  ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'A  ', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'A1 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'A1 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'A1 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'A1 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'A1 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'A1 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'A1 ', 6, 27),

        (None, 'P', None, 501, 'ANY', 'A  ', 'A2 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1 ', 'A2 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2 ', 'A2 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'AB ', 'A2 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A1B', 'A2 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'A2B', 'A2 ', 6, 27),
        (None, 'P', None, 501, 'ANY', 'B  ', 'A2 ', 6, 27),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_28_30(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'O  ', 30, 28),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'B  ', 30, 28),

        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'O  ', 6, 29),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'O  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'O  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'O  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'O  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'O  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'O  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'O  ', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'B  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'B  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'B  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'B  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'B  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'B  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'B  ', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'AB ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'AB ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'AB ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'AB ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'AB ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'AB ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'AB ', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'A1B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'A1B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'A1B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'A1B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'A1B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'A1B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'A1B', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'A2B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'A2B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'A2B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'A2B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'A2B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'A2B', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'A2B', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'A  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'A  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'A  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'A  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'A  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'A  ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'A  ', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'A1 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'A1 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'A1 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'A1 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'A1 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'A1 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'A1 ', 6, 30),

        (None, 'M', 12*18-1, 501, 'ANY', 'A  ', 'A2 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1 ', 'A2 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2 ', 'A2 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'AB ', 'A2 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A1B', 'A2 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'A2B', 'A2 ', 6, 30),
        (None, 'M', 12*18-1, 501, 'ANY', 'B  ', 'A2 ', 6, 30),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_31_33(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'O  ', 30, 31),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'B  ', 30, 31),

        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'O  ', 6, 32),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'O  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'O  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'O  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'O  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'O  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'O  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'O  ', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'B  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'B  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'B  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'B  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'B  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'B  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'B  ', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'AB ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'AB ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'AB ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'AB ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'AB ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'AB ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'AB ', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'A1B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'A1B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'A1B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'A1B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'A1B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'A1B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'A1B', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'A2B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'A2B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'A2B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'A2B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'A2B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'A2B', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'A2B', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'A  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'A  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'A  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'A  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'A  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'A  ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'A  ', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'A1 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'A1 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'A1 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'A1 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'A1 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'A1 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'A1 ', 6, 33),

        (None, 'M', 12*18, 500, 'ANY', 'A  ', 'A2 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1 ', 'A2 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2 ', 'A2 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'AB ', 'A2 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A1B', 'A2 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'A2B', 'A2 ', 6, 33),
        (None, 'M', 12*18, 500, 'ANY', 'B  ', 'A2 ', 6, 33),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_34_36(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'O  ', 30, 34),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'B  ', 30, 34),

        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'O  ', 6, 35),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'O  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'O  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'O  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'O  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'O  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'O  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'O  ', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'B  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'B  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'B  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'B  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'B  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'B  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'B  ', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'AB ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'AB ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'AB ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'AB ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'AB ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'AB ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'AB ', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'A1B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'A1B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'A1B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'A1B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'A1B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'A1B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'A1B', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'A2B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'A2B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'A2B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'A2B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'A2B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'A2B', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'A2B', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'A  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'A  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'A  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'A  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'A  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'A  ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'A  ', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'A1 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'A1 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'A1 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'A1 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'A1 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'A1 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'A1 ', 6, 36),

        (None, 'M', 12*18, 501, 'ANY', 'A  ', 'A2 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1 ', 'A2 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2 ', 'A2 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'AB ', 'A2 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A1B', 'A2 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'A2B', 'A2 ', 6, 36),
        (None, 'M', 12*18, 501, 'ANY', 'B  ', 'A2 ', 6, 36),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_37_42(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 500, 'ANY', 'O  ', 'B  ', 6, 37),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'B  ', 6, 38),

        (None, 'P', None, 501, 'ANY', 'O  ', 'B  ', 6, 39),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'B  ', 6, 40),

        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'B  ', 6, 41),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'B  ', 6, 42),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_G_tier_43_48(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, 'P', None, 500, 'ANY', 'O  ', 'A  ', 6, 43),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A1 ', 6, 43),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A2 ', 6, 43),
        (None, 'P', None, 500, 'ANY', 'O  ', 'AB ', 6, 43),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A1B', 6, 43),
        (None, 'P', None, 500, 'ANY', 'O  ', 'A2B', 6, 43),

        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'A  ', 6, 44),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'A1 ', 6, 44),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'A2 ', 6, 44),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'AB ', 6, 44),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'A1B', 6, 44),
        (None, 'M', 12*18-1, 500, 'ANY', 'O  ', 'A2B', 6, 44),

        (None, 'P', None, 501, 'ANY', 'O  ', 'A  ', 6, 45),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A1 ', 6, 45),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A2 ', 6, 45),
        (None, 'P', None, 501, 'ANY', 'O  ', 'AB ', 6, 45),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A1B', 6, 45),
        (None, 'P', None, 501, 'ANY', 'O  ', 'A2B', 6, 45),

        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'A  ', 6, 46),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'A1 ', 6, 46),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'A2 ', 6, 46),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'AB ', 6, 46),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'A1B', 6, 46),
        (None, 'M', 12*18-1, 501, 'ANY', 'O  ', 'A2B', 6, 46),

        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A  ', 6, 47),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A1 ', 6, 47),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A2 ', 6, 47),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'AB ', 6, 47),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A1B', 6, 47),
        (None, 'M', 12*18, 500, 'ANY', 'O  ', 'A2B', 6, 47),

        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A  ', 6, 48),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A1 ', 6, 48),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A2 ', 6, 48),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'AB ', 6, 48),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A1B', 6, 48),
        (None, 'M', 12*18, 501, 'ANY', 'O  ', 'A2B', 6, 48),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_G(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_1_2(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, None, 500, 'ANY', 'O  ', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'O  ', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'B  ', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'B  ', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'AB ', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'AB ', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'A1B', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'A1B', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'A2B', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'A2B', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'A  ', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'A  ', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'A1 ', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'A1 ', 'A2B', None, 1),

        (6011, None, None, 500, 'ANY', 'A2 ', 'O  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'B  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'A  ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'A1 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'A2 ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'AB ', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'A1B', None, 1),
        (6011, None, None, 500, 'ANY', 'A2 ', 'A2B', None, 1),

        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'O  ', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'B  ', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'AB ', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1B', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2B', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A  ', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A1 ', 'A2B', None, 2),

        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'O  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'B  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A  ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A1 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A2 ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'AB ', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A1B', None, 2),
        (6012, None, 12*18-1, 500, 'ANY', 'A2 ', 'A2B', None, 2),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_3_5(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 150, 'ANY', 'O  ', 'O  ', 30, 3),
        (None, None, None, 150, 'ANY', 'O  ', 'B  ', 30, 3),

        (None, None, None, 150, 'ANY', 'O  ', 'O  ', 15, 4),

        (None, None, None, 150, 'ANY', 'B  ', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'B  ', 'A2B', 15, 5),

        (None, None, None, 150, 'ANY', 'AB ', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'AB ', 'A2B', 15, 5),

        (None, None, None, 150, 'ANY', 'A1B', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'A1B', 'A2B', 15, 5),

        (None, None, None, 150, 'ANY', 'A2B', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'A2B', 'A2B', 15, 5),

        (None, None, None, 150, 'ANY', 'A  ', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'A  ', 'A2B', 15, 5),

        (None, None, None, 150, 'ANY', 'A1 ', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'A1 ', 'A2B', 15, 5),

        (None, None, None, 150, 'ANY', 'A2 ', 'O  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'B  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'A  ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'A1 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'A2 ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'AB ', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'A1B', 15, 5),
        (None, None, None, 150, 'ANY', 'A2 ', 'A2B', 15, 5),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_6_8(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 250, 'ANY', 'O  ', 'O  ', 30, 6),
        (None, None, None, 250, 'ANY', 'O  ', 'B  ', 30, 6),

        (None, None, None, 250, 'ANY', 'O  ', 'O  ', 15, 7),

        (None, None, None, 250, 'ANY', 'B  ', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'B  ', 'A2B', 15, 8),

        (None, None, None, 250, 'ANY', 'AB ', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'AB ', 'A2B', 15, 8),

        (None, None, None, 250, 'ANY', 'A1B', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'A1B', 'A2B', 15, 8),

        (None, None, None, 250, 'ANY', 'A2B', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'A2B', 'A2B', 15, 8),

        (None, None, None, 250, 'ANY', 'A  ', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'A  ', 'A2B', 15, 8),

        (None, None, None, 250, 'ANY', 'A1 ', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'A1 ', 'A2B', 15, 8),

        (None, None, None, 250, 'ANY', 'A2 ', 'O  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'B  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'A  ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'A1 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'A2 ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'AB ', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'A1B', 15, 8),
        (None, None, None, 250, 'ANY', 'A2 ', 'A2B', 15, 8),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_9_11(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 500, 'ANY', 'O  ', 'O  ', 30, 9),
        (None, None, None, 500, 'ANY', 'O  ', 'B  ', 30, 9),

        (None, None, None, 500, 'ANY', 'O  ', 'O  ', 15, 10),

        (None, None, None, 500, 'ANY', 'B  ', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'B  ', 'A2B', 15, 11),

        (None, None, None, 500, 'ANY', 'AB ', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'AB ', 'A2B', 15, 11),

        (None, None, None, 500, 'ANY', 'A1B', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'A1B', 'A2B', 15, 11),

        (None, None, None, 500, 'ANY', 'A2B', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'A2B', 'A2B', 15, 11),

        (None, None, None, 500, 'ANY', 'A  ', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'A  ', 'A2B', 15, 11),

        (None, None, None, 500, 'ANY', 'A1 ', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'A1 ', 'A2B', 15, 11),

        (None, None, None, 500, 'ANY', 'A2 ', 'O  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'B  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'A  ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'A1 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'A2 ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'AB ', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'A1B', 15, 11),
        (None, None, None, 500, 'ANY', 'A2 ', 'A2B', 15, 11),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_12_13(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (6011, None, None, 501, 'ANY', 'O  ', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'O  ', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'B  ', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'B  ', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'AB ', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'AB ', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'A1B', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'A1B', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'A2B', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'A2B', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'A  ', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'A  ', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'A1 ', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'A1 ', 'A2B', None, 12),

        (6011, None, None, 501, 'ANY', 'A2 ', 'O  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'B  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'A  ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'A1 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'A2 ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'AB ', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'A1B', None, 12),
        (6011, None, None, 501, 'ANY', 'A2 ', 'A2B', None, 12),

        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'O  ', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'B  ', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'AB ', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1B', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2B', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A  ', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A1 ', 'A2B', None, 13),

        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'O  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'B  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'A  ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'A1 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'A2 ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'AB ', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'A1B', None, 13),
        (6012, None, 12*18-1, 501, 'ANY', 'A2 ', 'A2B', None, 13),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_14_16(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 501, 'ANY', 'O  ', 'O  ', 30, 14),
        (None, None, None, 501, 'ANY', 'O  ', 'B  ', 30, 14),

        (None, None, None, 501, 'ANY', 'O  ', 'O  ', 15, 15),

        (None, None, None, 501, 'ANY', 'B  ', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'B  ', 'A2B', 15, 16),

        (None, None, None, 501, 'ANY', 'AB ', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'AB ', 'A2B', 15, 16),

        (None, None, None, 501, 'ANY', 'A1B', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'A1B', 'A2B', 15, 16),

        (None, None, None, 501, 'ANY', 'A2B', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'A2B', 'A2B', 15, 16),

        (None, None, None, 501, 'ANY', 'A  ', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'A  ', 'A2B', 15, 16),

        (None, None, None, 501, 'ANY', 'A1 ', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'A1 ', 'A2B', 15, 16),

        (None, None, None, 501, 'ANY', 'A2 ', 'O  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'B  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'A  ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'A1 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'A2 ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'AB ', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'A1B', 15, 16),
        (None, None, None, 501, 'ANY', 'A2 ', 'A2B', 15, 16),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_17_18(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 150, 'ANY', 'O  ', 'O  ', 6, 17),

        (None, None, None, 150, 'ANY', 'B  ', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'B  ', 'A2B', 6, 18),

        (None, None, None, 150, 'ANY', 'AB ', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'AB ', 'A2B', 6, 18),

        (None, None, None, 150, 'ANY', 'A1B', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'A1B', 'A2B', 6, 18),

        (None, None, None, 150, 'ANY', 'A2B', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'A2B', 'A2B', 6, 18),

        (None, None, None, 150, 'ANY', 'A  ', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'A  ', 'A2B', 6, 18),

        (None, None, None, 150, 'ANY', 'A1 ', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'A1 ', 'A2B', 6, 18),

        (None, None, None, 150, 'ANY', 'A2 ', 'O  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'B  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'A  ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'A1 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'A2 ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'AB ', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'A1B', 6, 18),
        (None, None, None, 150, 'ANY', 'A2 ', 'A2B', 6, 18),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_19_20(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 250, 'ANY', 'O  ', 'O  ', 6, 19),

        (None, None, None, 250, 'ANY', 'B  ', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'B  ', 'A2B', 6, 20),

        (None, None, None, 250, 'ANY', 'AB ', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'AB ', 'A2B', 6, 20),

        (None, None, None, 250, 'ANY', 'A1B', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'A1B', 'A2B', 6, 20),

        (None, None, None, 250, 'ANY', 'A2B', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'A2B', 'A2B', 6, 20),

        (None, None, None, 250, 'ANY', 'A  ', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'A  ', 'A2B', 6, 20),

        (None, None, None, 250, 'ANY', 'A1 ', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'A1 ', 'A2B', 6, 20),

        (None, None, None, 250, 'ANY', 'A2 ', 'O  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'B  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'A  ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'A1 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'A2 ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'AB ', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'A1B', 6, 20),
        (None, None, None, 250, 'ANY', 'A2 ', 'A2B', 6, 20),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_21_22(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 500, 'ANY', 'O  ', 'O  ', 6, 21),

        (None, None, None, 500, 'ANY', 'B  ', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'B  ', 'A2B', 6, 22),

        (None, None, None, 500, 'ANY', 'AB ', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'AB ', 'A2B', 6, 22),

        (None, None, None, 500, 'ANY', 'A1B', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'A1B', 'A2B', 6, 22),

        (None, None, None, 500, 'ANY', 'A2B', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'A2B', 'A2B', 6, 22),

        (None, None, None, 500, 'ANY', 'A  ', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'A  ', 'A2B', 6, 22),

        (None, None, None, 500, 'ANY', 'A1 ', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'A1 ', 'A2B', 6, 22),

        (None, None, None, 500, 'ANY', 'A2 ', 'O  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'B  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'A  ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'A1 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'A2 ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'AB ', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'A1B', 6, 22),
        (None, None, None, 500, 'ANY', 'A2 ', 'A2B', 6, 22),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_23_24(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 501, 'ANY', 'O  ', 'O  ', 6, 23),

        (None, None, None, 501, 'ANY', 'B  ', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'B  ', 'A2B', 6, 24),

        (None, None, None, 501, 'ANY', 'AB ', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'AB ', 'A2B', 6, 24),

        (None, None, None, 501, 'ANY', 'A1B', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'A1B', 'A2B', 6, 24),

        (None, None, None, 501, 'ANY', 'A2B', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'A2B', 'A2B', 6, 24),

        (None, None, None, 501, 'ANY', 'A  ', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'A  ', 'A2B', 6, 24),

        (None, None, None, 501, 'ANY', 'A1 ', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'A1 ', 'A2B', 6, 24),

        (None, None, None, 501, 'ANY', 'A2 ', 'O  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'B  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'A  ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'A1 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'A2 ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'AB ', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'A1B', 6, 24),
        (None, None, None, 501, 'ANY', 'A2 ', 'A2B', 6, 24),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_25_32(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 150, 'ANY', 'O  ', 'B  ', 15, 25),
        (None, None, None, 250, 'ANY', 'O  ', 'B  ', 15, 26),
        (None, None, None, 500, 'ANY', 'O  ', 'B  ', 15, 27),
        (None, None, None, 501, 'ANY', 'O  ', 'B  ', 15, 28),
        (None, None, None, 150, 'ANY', 'O  ', 'B  ', 6, 29),
        (None, None, None, 250, 'ANY', 'O  ', 'B  ', 6, 30),
        (None, None, None, 500, 'ANY', 'O  ', 'B  ', 6, 31),
        (None, None, None, 501, 'ANY', 'O  ', 'B  ', 6, 32),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_policy_9_8_H_tier_33_40(polars_frame_function):

    schema = {
        'CANHX_STAT_CD': pl.Int64,
        'CANHX_SRTR_LAB_MELD_TY': pl.String,
        'CAN_AGE_IN_MONTHS_AT_LISTING': pl.Int64,
        'distance_NM': pl.Float64,
        'PRIMARY_STATE': pl.String,
        'DON_ABO': pl.String,
        'CAN_ABO': pl.String,
        'allocation_meld': pl.Int64,
        'tier': pl.Int64,
    }
    data = (
        (None, None, None, 150, 'ANY', 'O  ', 'A  ', 15, 33),
        (None, None, None, 250, 'ANY', 'O  ', 'A  ', 15, 34),
        (None, None, None, 500, 'ANY', 'O  ', 'A  ', 15, 35),
        (None, None, None, 501, 'ANY', 'O  ', 'A  ', 15, 36),
        (None, None, None, 150, 'ANY', 'O  ', 'A  ', 6, 37),
        (None, None, None, 250, 'ANY', 'O  ', 'A  ', 6, 38),
        (None, None, None, 500, 'ANY', 'O  ', 'A  ', 6, 39),
        (None, None, None, 501, 'ANY', 'O  ', 'A  ', 6, 40),

        (None, None, None, 150, 'ANY', 'O  ', 'A1 ', 15, 33),
        (None, None, None, 250, 'ANY', 'O  ', 'A1 ', 15, 34),
        (None, None, None, 500, 'ANY', 'O  ', 'A1 ', 15, 35),
        (None, None, None, 501, 'ANY', 'O  ', 'A1 ', 15, 36),
        (None, None, None, 150, 'ANY', 'O  ', 'A1 ', 6, 37),
        (None, None, None, 250, 'ANY', 'O  ', 'A1 ', 6, 38),
        (None, None, None, 500, 'ANY', 'O  ', 'A1 ', 6, 39),
        (None, None, None, 501, 'ANY', 'O  ', 'A1 ', 6, 40),

        (None, None, None, 150, 'ANY', 'O  ', 'A2 ', 15, 33),
        (None, None, None, 250, 'ANY', 'O  ', 'A2 ', 15, 34),
        (None, None, None, 500, 'ANY', 'O  ', 'A2 ', 15, 35),
        (None, None, None, 501, 'ANY', 'O  ', 'A2 ', 15, 36),
        (None, None, None, 150, 'ANY', 'O  ', 'A2 ', 6, 37),
        (None, None, None, 250, 'ANY', 'O  ', 'A2 ', 6, 38),
        (None, None, None, 500, 'ANY', 'O  ', 'A2 ', 6, 39),
        (None, None, None, 501, 'ANY', 'O  ', 'A2 ', 6, 40),

        (None, None, None, 150, 'ANY', 'O  ', 'AB ', 15, 33),
        (None, None, None, 250, 'ANY', 'O  ', 'AB ', 15, 34),
        (None, None, None, 500, 'ANY', 'O  ', 'AB ', 15, 35),
        (None, None, None, 501, 'ANY', 'O  ', 'AB ', 15, 36),
        (None, None, None, 150, 'ANY', 'O  ', 'AB ', 6, 37),
        (None, None, None, 250, 'ANY', 'O  ', 'AB ', 6, 38),
        (None, None, None, 500, 'ANY', 'O  ', 'AB ', 6, 39),
        (None, None, None, 501, 'ANY', 'O  ', 'AB ', 6, 40),

        (None, None, None, 150, 'ANY', 'O  ', 'A1B', 15, 33),
        (None, None, None, 250, 'ANY', 'O  ', 'A1B', 15, 34),
        (None, None, None, 500, 'ANY', 'O  ', 'A1B', 15, 35),
        (None, None, None, 501, 'ANY', 'O  ', 'A1B', 15, 36),
        (None, None, None, 150, 'ANY', 'O  ', 'A1B', 6, 37),
        (None, None, None, 250, 'ANY', 'O  ', 'A1B', 6, 38),
        (None, None, None, 500, 'ANY', 'O  ', 'A1B', 6, 39),
        (None, None, None, 501, 'ANY', 'O  ', 'A1B', 6, 40),

        (None, None, None, 150, 'ANY', 'O  ', 'A2B', 15, 33),
        (None, None, None, 250, 'ANY', 'O  ', 'A2B', 15, 34),
        (None, None, None, 500, 'ANY', 'O  ', 'A2B', 15, 35),
        (None, None, None, 501, 'ANY', 'O  ', 'A2B', 15, 36),
        (None, None, None, 150, 'ANY', 'O  ', 'A2B', 6, 37),
        (None, None, None, 250, 'ANY', 'O  ', 'A2B', 6, 38),
        (None, None, None, 500, 'ANY', 'O  ', 'A2B', 6, 39),
        (None, None, None, 501, 'ANY', 'O  ', 'A2B', 6, 40),
    )
    expected = polars_frame_function(
        data=data,
        schema=schema,
        orient='row',
    )
    frame = expected.drop('tier')
    frame = policy_9_8_H(frame)

    assert_frame_equal(frame, expected)

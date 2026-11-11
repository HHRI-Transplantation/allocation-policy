from __future__ import annotations

from collections.abc import Callable

import polars as pl
from sim_utils import collapse_abo
from sim_utils import PolarsFrame

SMALL_CIRCLE_RADIUS = 150
LARGE_CIRCLE_RADIUS = 500
MEDIUM_CIRCLE_RADIUS = 250

HI_CIRCLE_RADIUS = 2400
PR_CIRCLE_RADIUS = 1100

PED_AGE_MONTHS_CUTOFF = 12*18

STAT_1A_CD = 6011
STAT_1B_CD = 6012


def allocation_pathway(
    don_non_hr_beat: str | None,
    don_age_in_months: int | None,
) -> Callable[[PolarsFrame], PolarsFrame]:
    '''

        Based on the donor info in the dataframe,
        returns the allocation pathway for the base
        acuity circles policy.

    '''

    if (
        don_non_hr_beat is None and
        don_age_in_months is None
    ):
        return dummy_policy

    don_age_at_least_18 = (
        don_age_in_months is not None and
        don_age_in_months >= PED_AGE_MONTHS_CUTOFF
    )
    don_age_under_18 = not don_age_at_least_18
    don_age_at_least_11 = (
        don_age_in_months is not None and
        don_age_in_months >= 11*12
    )
    don_age_under_11 = not don_age_at_least_11
    don_age_under_70 = (
        don_age_in_months is not None and
        don_age_in_months < 70*12
    )

    non_dcd = don_non_hr_beat != 'Y'

    if (
        don_age_at_least_18 &
        don_age_under_70 &
        non_dcd
    ):
        pathway = policy_9_8_E

    elif (
        don_age_under_18 &
        don_age_at_least_11 &
        non_dcd
    ):
        pathway = policy_9_8_F

    elif (
        don_age_under_11 &
        non_dcd
    ):
        pathway = policy_9_8_G

    else:
        pathway = policy_9_8_H

    return pathway


def dummy_policy(frame: PolarsFrame) -> PolarsFrame:
    '''

        Should only be invoked if there are no rows in
        the frame.
    '''

    return (
        frame
        .with_columns(
            pl.lit(None)
            .cast(pl.Int64)
            .alias('tier'),
        )
    )


def policy_9_8_E(frame: PolarsFrame) -> PolarsFrame:
    '''

        Implements acuity circles policy 9.8.E. Historically,
        this policy applied specifically to livers from non-DCD
        donors at least 18 years old and less than 70 years old.

        This function simply provides each row (i.e., candidate)
        of the dataframe a tier value.

    '''
    frame = (
        frame
        .with_columns(
            collapse_abo(pl.col('DON_ABO'))
            .alias('DON_ABO_CLEAN'),

            collapse_abo(pl.col('CAN_ABO'))
            .alias('CAN_ABO_CLEAN'),
        )
        .with_columns(

            # Tier 1
            pl.when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(1))

            # Tier 2
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(2))

            # Tier 3
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(3))
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(3))

            # Tier 4
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(4))
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(4))

            # Tier 5
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(5))

            # Tier 6
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(6))

            # Tier 7
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(7))

            # Tier 8
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(8))

            # Tier 9
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(9))

            # Tier 10
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(10))

            # Tier 11
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(11))
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(11))

            # Tier 12
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(12))
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(12))

            # Tier 13
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(13))

            # Tier 14
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(14))

            # Tier 15
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(15))

            # Tier 16
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(16))

            # Tier 17
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(17))

            # Tier 18
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(18))

            # Tier 19
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(19))

            # Tier 20
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(20))

            # Tier 21
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(21))

            # Tier 22
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(22))

            # Tier 23
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(23))

            # Tier 24
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(24))

            # Tier 25
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(25))

            # Tier 26
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(26))

            # Tier 27
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(27))

            # Tier 28
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(28))

            # Tier 29
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(29))

            # Tier 30
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(30))

            # Tier 31
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(31))

            # Tier 32
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(32))

            # Tier 33
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(33))

            # Tier 34
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(34))

            # Tier 35
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(35))

            # Tier 36
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(36))

            # Tier 37
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(37))

            # Tier 38
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(38))

            # Tier 39
            .when(
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(39))

            # Tier 40
            .when(
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(40))

            # Tier 41
            .when(
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(41))

            # Tier 42
            .when(
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(42))

            # Tier 43
            .when(
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(43))

            # Tier 44
            .when(
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(44))

            # Tier 45
            .when(
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(45))

            # Tier 46
            .when(
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(46))

            # Tier 47
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(47))

            # Tier 48
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(48))

            # Tier 49
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(49))

            # Tier 50
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(50))

            # Tier 51
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(51))

            # Tier 52
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(52))

            # Tier 53
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(53))

            # Tier 54
            .when(
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(54))

            # Tier 55
            .when(
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(55))

            # Tier 56
            .when(
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(56))

            # Tier 57
            .when(
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(57))

            # Tier 58
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(58))

            # Tier 59
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(59))

            # Tier 60
            .when(
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(60))

            # Tier 61
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(61))

            # Tier 62
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(62))

            # Tier 63
            .when(
                pl.col('allocation_meld') >= pl.lit(33),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(63))

            # Tier 64
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(64))

            # Tier 65
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(65))

            # Tier 66
            .when(
                pl.col('allocation_meld') >= pl.lit(29),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(66))

            # Tier 67
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(67))

            # Tier 68
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(68))

            # Tier 69
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(69))

            # Tier 70
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(70))

            # Tier 71
            .when(
                pl.col('allocation_meld') < pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(71))

            # Tier 72
            .when(
                pl.col('allocation_meld') < pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(72))

            # Tier 73
            .when(
                pl.col('allocation_meld') < pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(73))

            # Tier 74
            .when(
                pl.col('allocation_meld') < pl.lit(15),
                pl.col('distance_NM') > pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(74))

            .otherwise(None)
            .cast(pl.Int64)
            .alias('tier'),
        )
        .drop('DON_ABO_CLEAN', 'CAN_ABO_CLEAN')
    )

    return frame


def policy_9_8_F(frame: PolarsFrame) -> PolarsFrame:
    '''

        Implements acuity circles policy 9.8.F. Historically,
        this policy applied specifically to livers from non-DCD
        donors at least 11 and under 18.

        This function simply provides each row (i.e., candidate)
        of the dataframe a tier value.

    '''

    frame = (
        frame
        .with_columns(
            collapse_abo(pl.col('DON_ABO'))
            .alias('DON_ABO_CLEAN'),

            collapse_abo(pl.col('CAN_ABO'))
            .alias('CAN_ABO_CLEAN'),

            (
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18)
            )
            .alias('pediatric'),
        )
        .with_columns(

            # Tier 1
            pl.when(
                pl.col('pediatric'),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(1))

            # Tier 2
            .when(
                pl.col('pediatric').not_(),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(2))

            # Tier 3
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(3))

            # Tier 4
            .when(
                pl.col('pediatric'),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(4))
            .when(
                pl.col('pediatric'),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(4))

            # Tier 5
            .when(
                pl.col('pediatric').not_(),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(5))
            .when(
                pl.col('pediatric').not_(),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(5))

            # Tier 6
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(6))
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(6))

            # Tier 7
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(7))

            # Tier 8
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(8))

            # Tier 9
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(9))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(9))

            # Tier 10
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(10))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(10))

            # Tier 11
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(11))

            # Tier 12
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(12))

            # Tier 13
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(13))

            # Tier 14
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(14))

            # Tier 15
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(15))

            # Tier 16
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(16))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(16))

            # Tier 17
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(17))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(17))

            # Tier 18
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(18))

            # Tier 19
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(19))

            # Tier 20
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(20))

            # Tier 21
            .when(
                pl.col('pediatric'),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
            )
            .then(pl.lit(21))

            # Tier 22
            .when(
                pl.col('pediatric').not_(),
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
            )
            .then(pl.lit(22))

            # Tier 23
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
            )
            .then(pl.lit(23))

            # Tier 24
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(24))

            # Tier 25
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(25))

            # Tier 26
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(26))

            # Tier 27
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(27))

            # Tier 28
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(28))

            # Tier 29
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(29))

            # Tier 30
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(30))

            # Tier 31
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(31))

            # Tier 32
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(32))

            # Tier 33
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('pediatric').not_(),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(33))

            # Tier 34
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(34))

            # Tier 35
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(35))

            # Tier 36
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(36))

            # Tier 37
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(37))

            # Tier 38
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(38))

            # Tier 39
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(39))

            # Tier 40
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(40))

            # Tier 41
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(41))

            # Tier 42
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(42))

            # Tier 43
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(43))

            # Tier 44
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(44))

            # Tier 45
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(45))

            # Tier 46
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(46))

            # Tier 47
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('pediatric').not_(),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(47))

            .otherwise(None)
            .cast(pl.Int64)
            .alias('tier'),
        )
        .drop('DON_ABO_CLEAN', 'CAN_ABO_CLEAN', 'pediatric')
    )

    return frame


def policy_9_8_G(frame: PolarsFrame) -> PolarsFrame:
    '''

        Implements acuity circles policy 9.8.G. Historically,
        this policy applied specifically to livers from non-DCD
        donors less than 11 years old.

        This function simply provides each row (i.e., candidate)
        of the dataframe a tier value.

    '''

    frame = (
        frame
        .with_columns(
            collapse_abo(pl.col('DON_ABO'))
            .alias('DON_ABO_CLEAN'),

            collapse_abo(pl.col('CAN_ABO'))
            .alias('CAN_ABO_CLEAN'),
        )
        .with_columns(

            # Tier 1
            pl.when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(
                    PED_AGE_MONTHS_CUTOFF,
                ),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(1))

            # Tier 2
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*12),
            )
            .then(pl.lit(2))

            # Tier 3
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(3))

            # Tier 4
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(
                    PED_AGE_MONTHS_CUTOFF,
                ),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(4))

            # Tier 5
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(
                    PED_AGE_MONTHS_CUTOFF,
                ),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(5))
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(
                    PED_AGE_MONTHS_CUTOFF,
                ),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(5))

            # Tier 6
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(6))
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(6))

            # Tier 7
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(
                    PED_AGE_MONTHS_CUTOFF,
                ),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
            )
            .then(pl.lit(7))
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(
                    PED_AGE_MONTHS_CUTOFF,
                ),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
            )
            .then(pl.lit(7))

            # Tier 8
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(8))

            # Tier 9
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(9))

            # Tier 10
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
            )
            .then(pl.lit(10))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
            )
            .then(pl.lit(10))

            # Tier 11
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
            )
            .then(pl.lit(11))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
            )
            .then(pl.lit(11))

            # Tier 12
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(12))

            # Tier 13
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(13))

            # Tier 14
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(14))

            # Tier 15
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(15))

            # Tier 16
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
            )
            .then(pl.lit(16))

            # Tier 17
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
            )
            .then(pl.lit(17))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
            )
            .then(pl.lit(17))


            # Tier 18
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(HI_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('HI'),
            )
            .then(pl.lit(18))
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(PR_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
                pl.col('allocation_meld') >= pl.lit(37),
                pl.col('PRIMARY_STATE') == pl.lit('PR'),
            )
            .then(pl.lit(18))

            # Tier 19
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(19))

            # Tier 20
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(20))

            # Tier 21
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(21))

            # Tier 22
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
            )
            .then(pl.lit(22))

            # Tier 23
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
            )
            .then(pl.lit(23))

            # Tier 24
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
            )
            .then(pl.lit(24))

            # Tier 25
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(25))

            # Tier 26
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(26))

            # Tier 27
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(27))

            # Tier 28
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(28))

            # Tier 29
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(29))

            # Tier 30
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(30))

            # Tier 31
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') >= pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(31))

            # Tier 32
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') >= pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(32))

            # Tier 33
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') >= pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(33))

            # Tier 34
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') >= pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
                pl.col('allocation_meld') >= pl.lit(30),
            )
            .then(pl.lit(34))

            # Tier 35
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') >= pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(35))

            # Tier 36
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') >= pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(36))

            # Tier 37
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(37))

            # Tier 38
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(38))

            # Tier 39
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(39))

            # Tier 40
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(40))

            # Tier 41
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(41))

            # Tier 42
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(42))

            # Tier 43
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(43))

            # Tier 44
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(44))

            # Tier 45
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('P'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(45))

            # Tier 46
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(46))

            # Tier 47
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(47))

            # Tier 48
            .when(
                pl.col('CANHX_SRTR_LAB_MELD_TY') == pl.lit('M'),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(48))

            .otherwise(pl.lit(None))
            .cast(pl.Int64)
            .alias('tier'),
        )
        .drop('DON_ABO_CLEAN', 'CAN_ABO_CLEAN')
    )

    return frame


def policy_9_8_H(frame: PolarsFrame) -> PolarsFrame:
    '''

        Implements acuity circles policy 9.8.H. Historically,
        this policy applied specifically to livers from non-DCD
        donors at least 70 or DCD donors.

        This function simply provides each row (i.e., candidate)
        of the dataframe a tier value.

    '''

    return (
        frame
        .with_columns(
            collapse_abo(pl.col('DON_ABO'))
            .alias('DON_ABO_CLEAN'),

            collapse_abo(pl.col('CAN_ABO'))
            .alias('CAN_ABO_CLEAN'),
        )
        .with_columns(

            # Tier 1
            pl.when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(1))

            # Tier 2
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(2))

            # Tier 3
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(3))

            # Tier 4
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(4))

            # Tier 5
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(5))

            # Tier 6
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(6))

            # Tier 7
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(7))

            # Tier 8
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(8))

            # Tier 9
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(9))

            # Tier 10
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(10))

            # Tier 11
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(11))

            # Tier 12
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1A_CD),
            )
            .then(pl.lit(12))

            # Tier 13
            .when(
                pl.col('CANHX_STAT_CD') == pl.lit(STAT_1B_CD),
                pl.col('CAN_AGE_IN_MONTHS_AT_LISTING') < pl.lit(12*18),
            )
            .then(pl.lit(13))

            # Tier 14
            .when(
                pl.col('allocation_meld') >= pl.lit(30),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['O', 'B']),
            )
            .then(pl.lit(14))

            # Tier 15
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(15))

            # Tier 16
            .when(
                pl.col('allocation_meld') >= pl.lit(15),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(16))

            # Tier 17
            .when(
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(17))

            # Tier 18
            .when(
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(18))

            # Tier 19
            .when(
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(19))

            # Tier 20
            .when(
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(20))

            # Tier 21
            .when(
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(21))

            # Tier 22
            .when(
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(22))

            # Tier 23
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('O'),
            )
            .then(pl.lit(23))

            # Tier 24
            .when(
                pl.col('DON_ABO_CLEAN') != pl.lit('O'),
            )
            .then(pl.lit(24))

            # Tier 25
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(25))

            # Tier 26
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(26))

            # Tier 27
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(27))

            # Tier 28
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(28))

            # Tier 29
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
            )
            .then(pl.lit(29))

            # Tier 30
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
            )
            .then(pl.lit(30))

            # Tier 31
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(31))

            # Tier 32
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN') == pl.lit('B'),
            )
            .then(pl.lit(32))

            # Tier 33
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(33))

            # Tier 34
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(34))

            # Tier 35
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(35))

            # Tier 36
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('allocation_meld') >= pl.lit(15),
            )
            .then(pl.lit(36))

            # Tier 37
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('distance_NM') <= pl.lit(SMALL_CIRCLE_RADIUS),
            )
            .then(pl.lit(37))

            # Tier 38
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('distance_NM') <= pl.lit(MEDIUM_CIRCLE_RADIUS),
            )
            .then(pl.lit(38))

            # Tier 39
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
                pl.col('distance_NM') <= pl.lit(LARGE_CIRCLE_RADIUS),
            )
            .then(pl.lit(39))

            # Tier 40
            .when(
                pl.col('DON_ABO_CLEAN') == pl.lit('O'),
                pl.col('CAN_ABO_CLEAN').is_in(['A', 'AB']),
            )
            .then(pl.lit(40))

            .otherwise(pl.lit(None))
            .cast(pl.Int64)
            .alias('tier'),
        )
        .drop('DON_ABO_CLEAN', 'CAN_ABO_CLEAN')
    )

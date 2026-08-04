"""
Tests for the pluggable threshold/barrier-factor behaviour (decayshape.threshold).
"""

import numpy as np
import pytest

from decayshape.kmatrix_advanced import KMatrixAdvanced
from decayshape.lineshapes import Exponential, Flatte, Gaussian, GounarisSakurai, RelativisticBreitWigner
from decayshape.particles import Channel, CommonParticles
from decayshape.threshold import BarrierFactor, BlattWeisskopfBarrier, ConstantThreshold
from decayshape.utils import angular_momentum_barrier_factor, blatt_weiskopf_form_factor


@pytest.fixture
def pipi_channel():
    return Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)


class TestBlattWeisskopfBarrier:
    def test_default_parameter_order_is_just_r(self):
        assert BlattWeisskopfBarrier().parameter_order == ["r"]

    def test_q0_is_exposed_once_set(self):
        assert BlattWeisskopfBarrier(q0=0.2).parameter_order == ["r", "q0"]

    def test_matches_raw_utils_computation(self, pipi_channel):
        q = pipi_channel.momentum(0.6)
        threshold = BlattWeisskopfBarrier(r=1.2, q0=0.15)
        expected = blatt_weiskopf_form_factor(q, 1.2, 1) * angular_momentum_barrier_factor(q, 0.15, 1)
        assert threshold(q, pipi_channel, 1) == pytest.approx(expected)

    def test_q0_defaults_from_s0_when_unset(self, pipi_channel):
        s0 = 0.775**2
        q = pipi_channel.momentum(0.6)
        threshold = BlattWeisskopfBarrier(r=1.0)
        expected_q0 = pipi_channel.momentum(s0)
        expected = blatt_weiskopf_form_factor(q, 1.0, 1) * angular_momentum_barrier_factor(q, expected_q0, 1)
        assert threshold(q, pipi_channel, 1, s0=s0) == pytest.approx(expected)

    def test_get_parameters_includes_unset_q0_as_none(self):
        params = BlattWeisskopfBarrier(r=1.0).get_parameters()
        assert params == {"r": 1.0, "q0": None}

    def test_round_trip_serialization(self):
        threshold = BlattWeisskopfBarrier(r=1.3, q0=0.2)
        restored = BlattWeisskopfBarrier.model_validate(threshold.model_dump())
        assert restored == threshold


class TestBarrierFactor:
    def test_parameter_order_empty_until_q0_set(self):
        assert BarrierFactor().parameter_order == []
        assert BarrierFactor(q0=0.2).parameter_order == ["q0"]

    def test_no_form_factor_applied(self, pipi_channel):
        q = pipi_channel.momentum(0.6)
        threshold = BarrierFactor(q0=0.15)
        expected = angular_momentum_barrier_factor(q, 0.15, 1)
        assert threshold(q, pipi_channel, 1) == pytest.approx(expected)


class TestConstantThreshold:
    def test_parameter_order_is_empty(self):
        assert ConstantThreshold().parameter_order == []

    def test_always_returns_one(self, pipi_channel):
        q = pipi_channel.momentum(np.array([0.4, 0.6, 0.9]))
        result = ConstantThreshold()(q, pipi_channel, 3, s0=0.6)
        assert np.allclose(result, 1.0)


class TestLineshapeIntegration:
    def test_default_threshold_behaviour_is_blatt_weisskopf_barrier(self, pipi_channel):
        bw = RelativisticBreitWigner(channels=[pipi_channel], pole_mass=0.775, width=0.15)
        assert isinstance(bw.threshold_behaviour, BlattWeisskopfBarrier)

    def test_r_and_q0_flatten_into_parameter_order(self, pipi_channel):
        bw = RelativisticBreitWigner(
            channels=[pipi_channel], pole_mass=0.775, width=0.15, threshold_behaviour=BlattWeisskopfBarrier(q0=0.2)
        )
        assert bw.parameter_order == ["pole_mass", "width", "width_r", "r", "q0"]

    def test_constant_threshold_removes_outer_barrier(self, pipi_channel):
        s = np.linspace(0.4, 1.0, 5)
        bw_default = RelativisticBreitWigner(s=s, channels=[pipi_channel], pole_mass=0.775, width=0.15)
        bw_constant = RelativisticBreitWigner(
            s=s, channels=[pipi_channel], pole_mass=0.775, width=0.15, threshold_behaviour=ConstantThreshold()
        )

        # At L=0 the Blatt-Weisskopf barrier is trivially 1, so results must agree there...
        assert np.allclose(bw_default(0, 1), bw_constant(0, 1))
        # ...but at higher L, the default (nontrivial) barrier factor must differ from constant=1.
        assert not np.allclose(bw_default(2, 1), bw_constant(2, 1))

    def test_gaussian_and_exponential_default_to_constant_threshold(self):
        assert isinstance(Gaussian().threshold_behaviour, ConstantThreshold)
        assert isinstance(Exponential().threshold_behaviour, ConstantThreshold)
        # ConstantThreshold contributes nothing to the call signature
        assert Gaussian().parameter_order == ["mean", "width"]
        assert Exponential().parameter_order == ["slope"]

    @pytest.mark.parametrize("threshold_behaviour", [BlattWeisskopfBarrier(), BarrierFactor(), ConstantThreshold()])
    def test_every_threshold_type_works_for_every_channel_lineshape(self, pipi_channel, threshold_behaviour):
        """Regression test: mass_dependent_width's own r/q0 (width_r/width_q0, channel_r) must stay
        available regardless of which ThresholdFunction is plugged in - BarrierFactor/ConstantThreshold
        don't declare an `r` field at all, and ConstantThreshold doesn't declare `q0` either, so a design
        that sourced the width calculation's r/q0 straight from threshold_behaviour would KeyError here."""
        s = np.linspace(0.4, 1.0, 5)
        bw = RelativisticBreitWigner(
            s=s, channels=[pipi_channel], pole_mass=0.775, width=0.15, threshold_behaviour=threshold_behaviour
        )
        assert np.all(np.isfinite(bw(2, 1)))

        gs = GounarisSakurai(s=s, channel=pipi_channel, pole_mass=0.775, width=0.15, threshold_behaviour=threshold_behaviour)
        assert np.all(np.isfinite(gs(2, 1)))

        km = KMatrixAdvanced(s=s, channels=[pipi_channel], pole_masses=[0.98], threshold_behaviour=threshold_behaviour)
        assert np.all(np.isfinite(km(2, 2)))

    def test_width_r_independent_of_threshold_behaviour_r(self, pipi_channel):
        """Changing threshold_behaviour.r (outer barrier) must not change the mass-dependent width,
        and changing width_r must not change the outer barrier - they are deliberately decoupled."""
        s = np.linspace(0.4, 1.0, 5)
        base = RelativisticBreitWigner(
            s=s,
            channels=[pipi_channel],
            pole_mass=0.775,
            width=0.15,
            width_r=1.0,
            threshold_behaviour=BlattWeisskopfBarrier(r=1.0),
        )
        different_outer_r = RelativisticBreitWigner(
            s=s,
            channels=[pipi_channel],
            pole_mass=0.775,
            width=0.15,
            width_r=1.0,
            threshold_behaviour=BlattWeisskopfBarrier(r=2.0),
        )
        different_width_r = RelativisticBreitWigner(
            s=s,
            channels=[pipi_channel],
            pole_mass=0.775,
            width=0.15,
            width_r=2.0,
            threshold_behaviour=BlattWeisskopfBarrier(r=1.0),
        )

        assert not np.allclose(base(2, 1), different_outer_r(2, 1))
        assert not np.allclose(base(2, 1), different_width_r(2, 1))
        assert not np.allclose(different_outer_r(2, 1), different_width_r(2, 1))


class TestFlatteBarrier:
    """Flatte previously had no outer barrier factor at all; ConstantThreshold recovers that."""

    def _flatte(self, threshold_behaviour=None):
        s = np.linspace(0.8, 1.1, 5)
        channel1 = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        channel2 = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)
        kwargs = {}
        if threshold_behaviour is not None:
            kwargs["threshold_behaviour"] = threshold_behaviour
        return Flatte(
            s=s,
            channel1=channel1,
            channel2=channel2,
            pole_mass=0.98,
            width1=0.2,
            width2=0.8,
            r1=1.0,
            r2=1.0,
            **kwargs,
        )

    def test_constant_threshold_matches_bare_numerator(self):
        flatte = self._flatte(ConstantThreshold())
        result = flatte(2, 2)

        # Recompute the pre-refactor amplitude by hand: pole_mass * mass_dependent_width(...) / denominator,
        # with no outer barrier multiplying it.
        from decayshape.utils import mass_dependent_width

        channel1 = flatte.channel1.value
        channel2 = flatte.channel2.value
        s = flatte.s.value
        q1 = channel1.momentum(s)
        q2 = channel2.momentum(s)
        q01 = channel1.momentum(flatte.pole_mass**2)
        q02 = channel2.momentum(flatte.pole_mass**2)
        L = 1
        gamma1 = mass_dependent_width(q1, s, q01, flatte.pole_mass, flatte.width1, L, flatte.r1)
        gamma2 = mass_dependent_width(q2, s, q02, flatte.pole_mass, flatte.width2, L, flatte.r2)
        denominator = flatte.pole_mass**2 - s - 1j * flatte.pole_mass * (gamma1 + gamma2)
        numerator = flatte.pole_mass * mass_dependent_width(q1, s, q01, flatte.pole_mass, gamma1, L, flatte.r1)
        expected = numerator / denominator

        assert np.allclose(result, expected)

    def test_default_barrier_differs_from_constant_at_nonzero_l(self):
        flatte_default = self._flatte()
        flatte_constant = self._flatte(ConstantThreshold())
        assert not np.allclose(flatte_default(2, 2), flatte_constant(2, 2))

    def test_default_barrier_matches_constant_at_l_zero(self):
        flatte_default = self._flatte()
        flatte_constant = self._flatte(ConstantThreshold())
        assert np.allclose(flatte_default(0, 0), flatte_constant(0, 0))

"""
Tests for serialization and deserialization of lineshapes.
"""

import json

import numpy as np
import pytest

from decayshape import Flatte, KMatrixAdvanced, RelativisticBreitWigner
from decayshape.particles import Channel, CommonParticles


class TestRelativisticBreitWignerSerialization:
    def test_round_trip_model_dump_validate(self):
        s_vals = np.linspace(0.5, 0.8, 5)
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=s_vals, channel=pipi_channel, pole_mass=0.775, width=0.15, r=1.0)

        data = bw.model_dump()
        assert isinstance(data, dict)

        # Round-trip using model_validate
        restored = RelativisticBreitWigner.model_validate(data)
        assert isinstance(restored, RelativisticBreitWigner)

    def test_json_dump(self):
        s_vals = np.array([0.5, 0.6])
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=s_vals, channel=pipi_channel, pole_mass=0.775, width=0.15)
        json_str = bw.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "pole_mass" in parsed and parsed["pole_mass"] == pytest.approx(0.775)


class TestFlatteSerialization:
    def test_model_dump_and_validate(self):
        s_vals = np.linspace(0.4, 1.2, 4)
        
        # Create channels
        from decayshape.particles import Channel, Particle
        pipi_channel = Channel(
            particle1=Particle(mass=0.139, spin=0, parity=-1),
            particle2=Particle(mass=0.139, spin=0, parity=-1)
        )
        kk_channel = Channel(
            particle1=Particle(mass=0.494, spin=0, parity=-1),
            particle2=Particle(mass=0.494, spin=0, parity=-1)
        )
        
        flatte = Flatte(
            s=s_vals,
            channel1=pipi_channel,
            channel2=kk_channel,
            pole_mass=0.98,
            width1=0.1,
            width2=0.05,
            r1=1.0,
            r2=1.0,
        )

        data = flatte.model_dump()
        # Channels are FixedParam -> dumped as {"value": ...}
        for name in ["channel1", "channel2"]:
            assert isinstance(data[name], dict)
            assert "value" in data[name]

        # Round-trip
        restored = Flatte.model_validate(data)
        assert isinstance(restored, Flatte)
        assert restored.channel1.value.particle1.value.mass == pytest.approx(0.139)
        assert restored.channel2.value.particle2.value.mass == pytest.approx(0.494)


class TestKMatrixAdvancedSerialization:
    def test_model_dump_minimal(self):
        s_vals = np.array([0.5, 0.6, 0.7])
        pipi = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS,
        )
        kmat = KMatrixAdvanced(
            s=s_vals,
            channels=[pipi],
            pole_masses=[0.775],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0,
        )

        data = kmat.model_dump()
        assert "channels" in data and isinstance(data["channels"], dict)
        # channels is FixedParam[List[Channel]]
        assert "value" in data["channels"]
        assert isinstance(data["channels"]["value"], list)

        # JSON dump is at least valid JSON
        json_str = kmat.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "pole_masses" in parsed and parsed["pole_masses"] == [0.775]

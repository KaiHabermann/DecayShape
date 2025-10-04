"""
Tests for JsonSchemaMixin functionality across different models.
"""

import json

import numpy as np

from decayshape import Channel, CommonParticles, RelativisticBreitWigner


class TestJsonSchemaMixinParticle:
    """Test JsonSchemaMixin for Particle class."""

    def test_particle_to_json_schema(self):
        """Test that Particle can generate JSON schema."""
        pion = CommonParticles.PI_PLUS
        schema = pion.to_json_schema()

        # Check basic structure
        assert "model_type" in schema
        assert schema["model_type"] == "Particle"
        assert "description" in schema
        assert "parameters" in schema
        assert "current_values" in schema

        # Check parameters
        params = schema["parameters"]
        assert "mass" in params
        assert "spin" in params
        assert "parity" in params

        # Check current values
        values = schema["current_values"]
        assert values["mass"] == pion.mass
        assert values["spin"] == pion.spin
        assert values["parity"] == pion.parity

    def test_particle_to_json_string(self):
        """Test that Particle can generate JSON string."""
        kaon = CommonParticles.K_PLUS
        json_str = kaon.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "Particle"
        assert parsed["current_values"]["mass"] == kaon.mass


class TestJsonSchemaMixinChannel:
    """Test JsonSchemaMixin for Channel class."""

    def test_channel_to_json_schema(self):
        """Test that Channel can generate JSON schema."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS,
        )
        schema = channel.to_json_schema()

        # Check basic structure
        assert "model_type" in schema
        assert schema["model_type"] == "Channel"
        assert "description" in schema
        assert "fixed_parameters" in schema
        assert "current_values" in schema

        # Check fixed parameters
        fixed = schema["fixed_parameters"]
        assert "particle1" in fixed
        assert "particle2" in fixed

        # Check that particle values are included
        values = schema["current_values"]
        assert "particle1" in values
        assert "particle2" in values

    def test_channel_to_json_string(self):
        """Test that Channel can generate JSON string."""
        channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS,
        )
        json_str = channel.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "Channel"


class TestJsonSchemaMixinLineshape:
    """Test JsonSchemaMixin for Lineshape classes."""

    def test_lineshape_to_json_schema(self):
        """Test that Lineshape can generate JSON schema."""
        s = np.linspace(0.5, 2.0, 100) ** 2
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS,
        )
        bw = RelativisticBreitWigner(
            s=s,
            channel=channel,
            mass=0.770,
            width=0.150,
            angular_momentum=1,
            meson_radius=5.0,
        )
        schema = bw.to_json_schema()

        # Check lineshape-specific structure
        assert "lineshape_type" in schema
        assert schema["lineshape_type"] == "RelativisticBreitWigner"
        assert "optimization_parameters" in schema
        assert "parameter_order" in schema

        # Check that 's' is excluded
        assert "s" not in schema["optimization_parameters"]
        assert "s" not in schema["fixed_parameters"]
        assert "s" not in schema["parameter_order"]

        # Check other parameters are present
        opt_params = schema["optimization_parameters"]
        assert "pole_mass" in opt_params
        assert "width" in opt_params

    def test_lineshape_to_json_string(self):
        """Test that Lineshape can generate JSON string."""
        s = np.array([1.0, 2.0, 3.0])
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS,
        )
        bw = RelativisticBreitWigner(
            s=s,
            channel=channel,
            mass=0.770,
            width=0.150,
            angular_momentum=1,
            meson_radius=5.0,
        )
        json_str = bw.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["lineshape_type"] == "RelativisticBreitWigner"
        assert "s" not in parsed["parameter_order"]

    def test_lineshape_exclude_additional_fields(self):
        """Test excluding additional fields from schema."""
        s = np.array([1.0])
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS,
        )
        bw = RelativisticBreitWigner(
            s=s,
            channel=channel,
            mass=0.770,
            width=0.150,
            angular_momentum=1,
            meson_radius=5.0,
        )
        schema = bw.to_json_schema(exclude_fields=["angular_momentum"])

        # Check that additional excluded field is not present
        assert "angular_momentum" not in schema["optimization_parameters"]
        assert "angular_momentum" not in schema["fixed_parameters"]


class TestJsonSchemaMixinInheritance:
    """Test that JsonSchemaMixin works correctly with inheritance."""

    def test_mixin_is_inherited(self):
        """Test that all models have the mixin methods."""
        pion = CommonParticles.PI_PLUS
        channel = Channel(particle1=pion, particle2=pion)
        s = np.array([1.0])
        bw = RelativisticBreitWigner(
            s=s,
            channel=channel,
            mass=0.770,
            width=0.150,
            angular_momentum=1,
            meson_radius=5.0,
        )

        # All should have the mixin methods
        assert hasattr(pion, "to_json_schema")
        assert hasattr(pion, "to_json_string")
        assert hasattr(channel, "to_json_schema")
        assert hasattr(channel, "to_json_string")
        assert hasattr(bw, "to_json_schema")
        assert hasattr(bw, "to_json_string")

        # All should be callable
        assert callable(pion.to_json_schema)
        assert callable(channel.to_json_schema)
        assert callable(bw.to_json_schema)

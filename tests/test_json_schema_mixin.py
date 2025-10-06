"""
Tests for JsonSchemaMixin functionality across different models.
"""

import json


class TestJsonSchemaMixinParticle:
    """Test JsonSchemaMixin for Particle class."""

    def test_particle_to_json_schema(self):
        """Test that Particle can generate JSON schema."""
        from decayshape.particles import Particle

        schema = Particle.to_json_schema()

        # Check basic structure
        assert "model_type" in schema
        assert schema["model_type"] == "Particle"
        assert "description" in schema
        assert "parameters" in schema

        # Should not have current_values
        assert "current_values" not in schema

        # Check parameters
        params = schema["parameters"]
        assert "mass" in params
        assert "spin" in params
        assert "parity" in params

        # Check parameter structure
        assert params["mass"]["type"] == "number"
        assert params["spin"]["type"] == "number"
        assert params["parity"]["type"] == "integer"

    def test_particle_to_json_string(self):
        """Test that Particle can generate JSON string."""
        from decayshape.particles import Particle

        json_str = Particle.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "Particle"
        assert "parameters" in parsed
        assert "current_values" not in parsed


class TestJsonSchemaMixinChannel:
    """Test JsonSchemaMixin for Channel class."""

    def test_channel_to_json_schema(self):
        """Test that Channel can generate JSON schema."""
        from decayshape.particles import Channel

        schema = Channel.to_json_schema()

        # Check basic structure
        assert "model_type" in schema
        assert schema["model_type"] == "Channel"
        assert "description" in schema
        assert "fixed_parameters" in schema

        # Should not have current_values
        assert "current_values" not in schema

        # Check fixed parameters
        fixed = schema["fixed_parameters"]
        assert "particle1" in fixed
        assert "particle2" in fixed

        # Check that nested schemas are included
        assert "schema" in fixed["particle1"]
        assert "schema" in fixed["particle2"]

        # Check nested Particle schema
        particle_schema = fixed["particle1"]["schema"]
        assert "mass" in particle_schema
        assert "spin" in particle_schema
        assert "parity" in particle_schema

    def test_channel_to_json_string(self):
        """Test that Channel can generate JSON string."""
        from decayshape.particles import Channel

        json_str = Channel.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "Channel"


class TestJsonSchemaMixinLineshape:
    """Test JsonSchemaMixin for Lineshape classes."""

    def test_lineshape_to_json_schema(self):
        """Test that Lineshape can generate JSON schema."""
        from decayshape.lineshapes import RelativisticBreitWigner

        schema = RelativisticBreitWigner.to_json_schema()

        # Check lineshape-specific structure
        assert "lineshape_type" in schema
        assert schema["lineshape_type"] == "RelativisticBreitWigner"
        assert "optimization_parameters" in schema

        # Should not have parameter_order or current_values
        assert "parameter_order" not in schema
        assert "current_values" not in schema

        # Check that 's' is excluded
        assert "s" not in schema["optimization_parameters"]
        assert "s" not in schema["fixed_parameters"]

        # Check other parameters are present
        opt_params = schema["optimization_parameters"]
        assert "pole_mass" in opt_params
        assert "width" in opt_params

        # Check parameter types
        assert opt_params["pole_mass"]["type"] == "number"
        assert opt_params["width"]["type"] == "number"

    def test_lineshape_to_json_string(self):
        """Test that Lineshape can generate JSON string."""
        from decayshape.lineshapes import RelativisticBreitWigner

        json_str = RelativisticBreitWigner.to_json_string()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["lineshape_type"] == "RelativisticBreitWigner"
        assert "parameter_order" not in parsed
        assert "current_values" not in parsed
        assert "s" not in parsed["optimization_parameters"]

    def test_lineshape_exclude_additional_fields(self):
        """Test excluding additional fields from schema."""
        from decayshape.lineshapes import RelativisticBreitWigner

        schema = RelativisticBreitWigner.to_json_schema(exclude_fields=["angular_momentum"])

        # Check that additional excluded field is not present
        assert "angular_momentum" not in schema["optimization_parameters"]
        assert "angular_momentum" not in schema["fixed_parameters"]


class TestJsonSchemaMixinInheritance:
    """Test that JsonSchemaMixin works correctly with inheritance."""

    def test_mixin_is_inherited(self):
        """Test that all models have the mixin methods."""
        from decayshape.lineshapes import RelativisticBreitWigner
        from decayshape.particles import Channel, Particle

        # All should have the mixin methods
        assert hasattr(Particle, "to_json_schema")
        assert hasattr(Particle, "to_json_string")
        assert hasattr(Channel, "to_json_schema")
        assert hasattr(Channel, "to_json_string")
        assert hasattr(RelativisticBreitWigner, "to_json_schema")
        assert hasattr(RelativisticBreitWigner, "to_json_string")

        # All should be callable
        assert callable(Particle.to_json_schema)
        assert callable(Channel.to_json_schema)
        assert callable(RelativisticBreitWigner.to_json_schema)

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


class TestJsonSchemaMixinDiscriminatedUnion:
    """Test that threshold_behaviour (a discriminated union) exposes every strategy option."""

    def test_threshold_behaviour_lists_every_option(self):
        from decayshape.lineshapes import RelativisticBreitWigner

        schema = RelativisticBreitWigner.to_json_schema()
        threshold_info = schema["optimization_parameters"]["threshold_behaviour"]

        assert threshold_info["type"] == "discriminated_union"
        assert threshold_info["discriminator"] == "kind"
        assert threshold_info["default"] == "blatt_weisskopf_barrier"
        assert set(threshold_info["options"].keys()) == {
            "blatt_weisskopf_barrier",
            "barrier_factor",
            "constant_threshold",
        }

    def test_each_option_exposes_only_its_own_parameters(self):
        from decayshape.lineshapes import RelativisticBreitWigner

        options = RelativisticBreitWigner.to_json_schema()["optimization_parameters"]["threshold_behaviour"]["options"]

        assert set(options["blatt_weisskopf_barrier"]["parameters"].keys()) == {"r", "q0"}
        assert options["blatt_weisskopf_barrier"]["class"] == "BlattWeisskopfBarrier"

        assert set(options["barrier_factor"]["parameters"].keys()) == {"q0"}
        assert options["barrier_factor"]["class"] == "BarrierFactor"

        assert options["constant_threshold"]["parameters"] == {}
        assert options["constant_threshold"]["class"] == "ConstantThreshold"

    def test_r_is_no_longer_flattened_to_top_level(self):
        """r/q0 must not leak into optimization_parameters directly - only inside the
        threshold_behaviour options, since they belong to a swappable strategy, not the
        lineshape itself."""
        from decayshape.lineshapes import RelativisticBreitWigner

        opt_params = RelativisticBreitWigner.to_json_schema()["optimization_parameters"]
        assert "r" not in opt_params
        assert "q0" not in opt_params

    def test_default_reflects_each_lineshapes_own_default(self):
        """Gaussian defaults threshold_behaviour to ConstantThreshold; RBW defaults to
        BlattWeisskopfBarrier - the schema's "default" key must reflect the actual per-class
        default, while "options" always lists all three regardless."""
        from decayshape.lineshapes import Gaussian, RelativisticBreitWigner

        rbw_threshold = RelativisticBreitWigner.to_json_schema()["optimization_parameters"]["threshold_behaviour"]
        gaussian_threshold = Gaussian.to_json_schema()["optimization_parameters"]["threshold_behaviour"]

        assert rbw_threshold["default"] == "blatt_weisskopf_barrier"
        assert gaussian_threshold["default"] == "constant_threshold"
        assert set(gaussian_threshold["options"].keys()) == set(rbw_threshold["options"].keys())

    def test_unaffected_fields_stay_flat(self):
        """Fields that were never part of threshold_behaviour (Flatte's r1/r2/q01/q02,
        KMatrixAdvanced's channel_r) must be untouched by this change."""
        from decayshape.kmatrix_advanced import KMatrixAdvanced
        from decayshape.lineshapes import Flatte

        flatte_params = Flatte.to_json_schema()["optimization_parameters"]
        assert {"r1", "r2", "q01", "q02"}.issubset(flatte_params.keys())

        kmatrix_params = KMatrixAdvanced.to_json_schema()["optimization_parameters"]
        assert "channel_r" in kmatrix_params


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

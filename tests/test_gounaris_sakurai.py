import json

import numpy as np

from decayshape.base import FixedParam
from decayshape.lineshapes import GounarisSakurai
from decayshape.particles import Channel, CommonParticles


def test_gounaris_sakurai_instantiation():
    s_vals = np.linspace(0.5, 1.2, 100)
    pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

    gs = GounarisSakurai(s=s_vals, channel=FixedParam(value=pipi_channel), pole_mass=0.775, width=0.15)

    assert gs.pole_mass == 0.775
    assert gs.width == 0.15
    assert gs.channel.value == pipi_channel


def test_gounaris_sakurai_evaluation():
    s_vals = np.linspace(0.4, 1.2, 100) ** 2  # s is mass squared
    pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

    gs = GounarisSakurai(s=s_vals, channel=FixedParam(value=pipi_channel), pole_mass=0.775, width=0.15)

    # Call with L=1 (angular_momentum=2), spin=1 (doubled spin=2)
    amplitude = gs(angular_momentum=2, spin=2)

    assert amplitude.shape == s_vals.shape
    assert not np.any(np.isnan(amplitude))

    # Peak should be roughly around pole mass squared
    peak_idx = np.argmax(np.abs(amplitude))
    peak_s = s_vals[peak_idx]
    peak_m = np.sqrt(peak_s)

    assert 0.7 < peak_m < 0.85


def test_gounaris_sakurai_parameter_override():
    s_vals = np.array([0.775**2])
    pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

    gs = GounarisSakurai(s=s_vals, channel=FixedParam(value=pipi_channel), pole_mass=0.775, width=0.15)

    val1 = gs(angular_momentum=2, spin=2)
    val2 = gs(angular_momentum=2, spin=2, width=0.20)

    assert val1 != val2


def test_gounaris_sakurai_json_serialization():
    s_vals = np.linspace(0.5, 1.2, 10)
    pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

    gs = GounarisSakurai(s=s_vals, channel=FixedParam(value=pipi_channel), pole_mass=0.775, width=0.15)

    # Test JSON schema generation
    schema = gs.to_json_schema()
    assert isinstance(schema, dict)
    assert "lineshape_type" in schema
    assert schema["lineshape_type"] == "GounarisSakurai"
    assert "optimization_parameters" in schema
    assert "fixed_parameters" in schema

    opt_params = schema["optimization_parameters"]
    assert "pole_mass" in opt_params
    assert "width" in opt_params

    fixed_params = schema["fixed_parameters"]
    assert "channel" in fixed_params

    # Test model dumping to dict (serialization)
    model_dict = gs.model_dump()
    assert model_dict["pole_mass"] == 0.775
    assert model_dict["width"] == 0.15

    # Test round-trip (dump -> json string -> load)
    # Note: s array is not serialized by default model_dump unless configured,
    # but parameters should be.
    json_str = gs.model_dump_json()
    data = json.loads(json_str)
    assert data["pole_mass"] == 0.775

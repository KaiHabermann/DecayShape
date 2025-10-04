"""
Example demonstrating JSON schema generation for DecayShape models.

This shows how to use the JsonSchemaMixin to generate JSON representations
of Particle, Channel, and Lineshape objects for frontend consumption.
"""

import json

import numpy as np

from decayshape import Channel, CommonParticles, Flatte, KMatrixAdvanced, RelativisticBreitWigner


def main():
    """Demonstrate JSON schema generation for different model types."""

    print("=" * 80)
    print("JSON Schema Generation Examples")
    print("=" * 80)

    # 1. Particle JSON Schema
    print("\n1. Particle JSON Schema")
    print("-" * 80)
    pion = CommonParticles.PI_PLUS
    particle_schema = pion.to_json_schema()
    print(json.dumps(particle_schema, indent=2))

    # 2. Channel JSON Schema
    print("\n2. Channel JSON Schema")
    print("-" * 80)
    channel = Channel(
        particle1=CommonParticles.PI_PLUS,
        particle2=CommonParticles.PI_MINUS,
    )
    channel_schema = channel.to_json_schema()
    print(json.dumps(channel_schema, indent=2))

    # 3. Lineshape JSON Schema
    print("\n3. Lineshape JSON Schema (Relativistic Breit-Wigner)")
    print("-" * 80)
    s = np.linspace(0.5, 2.0, 100) ** 2
    bw = RelativisticBreitWigner(
        s=s,
        channel=channel,
        mass=0.770,
        width=0.150,
        angular_momentum=1,
        meson_radius=5.0,
    )
    lineshape_schema = bw.to_json_schema()

    # Note: The 's' parameter is automatically excluded from the schema
    # as it's not relevant for frontend configuration
    print(json.dumps(lineshape_schema, indent=2))

    # 4. Excluding additional fields
    print("\n4. Lineshape JSON Schema with Additional Exclusions")
    print("-" * 80)
    # You can exclude additional fields if needed
    minimal_schema = bw.to_json_schema(exclude_fields=["q0"])
    print(json.dumps(minimal_schema, indent=2))

    # 5. K-Matrix JSON Schema
    print("\n5. K-Matrix JSON Schema (Advanced Multi-Channel)")
    print("-" * 80)
    # Create a two-channel K-matrix
    channel1 = Channel(
        particle1=CommonParticles.PI_PLUS,
        particle2=CommonParticles.PI_MINUS,
    )
    channel2 = Channel(
        particle1=CommonParticles.K_PLUS,
        particle2=CommonParticles.K_MINUS,
    )

    kmatrix = KMatrixAdvanced(
        s=s,
        channels=[channel1, channel2],
        pole_masses=[0.65, 1.2],
        couplings=[
            [0.5, 0.3],  # Couplings for pole 1
            [0.4, 0.6],  # Couplings for pole 2
        ],
        scattering_length=[0.1, 0.2],
        effective_range=[0.05, 0.08],
        production_couplings=[1.0, 0.5],
        output_channel=0,
    )

    kmatrix_schema = kmatrix.to_json_schema()
    print(json.dumps(kmatrix_schema, indent=2))

    # 6. Direct JSON String Generation
    print("\n6. Direct JSON String Generation")
    print("-" * 80)
    json_string = pion.to_json_string(indent=2)
    print(json_string)

    print("\n" + "=" * 80)
    print("Use Cases:")
    print("- Send these schemas to a frontend to generate UI forms")
    print("- Document the structure of your models")
    print("- Validate frontend configurations before creating Python objects")
    print("- Generate API documentation automatically")
    print("- K-matrix schemas show complex multi-channel structures")
    print("=" * 80)


def generate_all_lineshape_schemas(output_file: str = "lineshape_schemas_complete.json"):
    """
    Generate JSON schemas for all available lineshapes and save to a file.

    This creates example instances of each lineshape type and extracts their
    schemas without current_values, making them suitable as templates for
    frontend form generation.

    Args:
        output_file: Path to the output JSON file
    """
    print("\n" + "=" * 80)
    print("Generating Schemas for All Available Lineshapes")
    print("=" * 80)

    # Prepare common components
    s = np.linspace(0.5, 2.0, 100) ** 2
    channel_pipi = Channel(
        particle1=CommonParticles.PI_PLUS,
        particle2=CommonParticles.PI_MINUS,
    )
    channel_kk = Channel(
        particle1=CommonParticles.K_PLUS,
        particle2=CommonParticles.K_MINUS,
    )

    # Dictionary to hold all schemas
    all_schemas = {}

    # 1. Relativistic Breit-Wigner
    print("\n1. Generating RelativisticBreitWigner schema...")
    bw = RelativisticBreitWigner(
        s=s,
        channel=channel_pipi,
        mass=0.770,
        width=0.150,
        angular_momentum=1,
        meson_radius=5.0,
    )
    bw_schema = bw.to_json_schema()
    # Remove current_values
    bw_schema.pop("current_values", None)
    all_schemas["RelativisticBreitWigner"] = bw_schema
    print(f"   -> RelativisticBreitWigner: {len(bw_schema['optimization_parameters'])} parameters")

    # 2. Flatté
    print("\n2. Generating Flatte schema...")
    flatte = Flatte(
        s=s,
        channel1_mass1=CommonParticles.PI_PLUS.mass,
        channel1_mass2=CommonParticles.PI_MINUS.mass,
        channel2_mass1=CommonParticles.K_PLUS.mass,
        channel2_mass2=CommonParticles.K_MINUS.mass,
        pole_mass=0.980,
        width1=0.167,
        width2=0.0,
        r1=1.0,
        r2=1.0,
        L1=0,
        L2=0,
    )
    flatte_schema = flatte.to_json_schema()
    # Remove current_values
    flatte_schema.pop("current_values", None)
    all_schemas["Flatte"] = flatte_schema
    print(f"   -> Flatte: {len(flatte_schema['optimization_parameters'])} parameters")

    # 3. K-Matrix Advanced
    print("\n3. Generating KMatrixAdvanced schema...")
    kmatrix = KMatrixAdvanced(
        s=s,
        channels=[channel_pipi, channel_kk],
        pole_masses=[0.65, 1.2],
        couplings=[
            [0.5, 0.3],
            [0.4, 0.6],
        ],
        scattering_length=[0.1, 0.2],
        effective_range=[0.05, 0.08],
        production_couplings=[1.0, 0.5],
        output_channel=0,
    )
    kmatrix_schema = kmatrix.to_json_schema()
    # Remove current_values
    kmatrix_schema.pop("current_values", None)
    all_schemas["KMatrixAdvanced"] = kmatrix_schema
    print(f"   -> KMatrixAdvanced: {len(kmatrix_schema['optimization_parameters'])} parameters")

    # Add metadata
    schemas_with_metadata = {
        "metadata": {
            "version": "1.0",
            "description": "JSON schemas for all available lineshapes in DecayShape",
            "note": "These schemas are templates without current values, suitable for frontend form generation",
            "lineshape_count": len(all_schemas),
        },
        "lineshapes": all_schemas,
    }

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schemas_with_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n=> Successfully saved {len(all_schemas)} lineshape schemas to: {output_file}")
    print(f"   Total file size: {len(json.dumps(schemas_with_metadata))} bytes")

    print("\n" + "=" * 80)
    print("Schema File Structure:")
    print("- metadata: Information about the schema collection")
    print("- lineshapes: Dictionary of lineshape schemas by name")
    print("  - Each schema contains:")
    print("    * lineshape_type: Name of the lineshape class")
    print("    * description: Documentation string")
    print("    * fixed_parameters: Parameters that don't change during optimization")
    print("    * optimization_parameters: Parameters that can be optimized")
    print("    * Nested schemas for complex types (e.g., Channel, Particle)")
    print("=" * 80)

    return schemas_with_metadata


if __name__ == "__main__":
    # Run the basic examples
    main()

    # Generate and save all lineshape schemas
    print("\n\n")
    generate_all_lineshape_schemas()

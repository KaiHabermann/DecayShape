"""
Example demonstrating JSON schema generation for DecayShape models.

This shows how to use the JsonSchemaMixin to generate JSON representations
of Particle, Channel, and Lineshape objects for frontend consumption.
"""

import json


def main():
    """Demonstrate JSON schema generation for different model types."""

    print("=" * 80)
    print("JSON Schema Generation Examples")
    print("=" * 80)

    # 1. Particle JSON Schema
    print("\n1. Particle JSON Schema")
    print("-" * 80)
    from decayshape.particles import Particle

    particle_schema = Particle.to_json_schema()
    print(json.dumps(particle_schema, indent=2))

    # 2. Channel JSON Schema
    print("\n2. Channel JSON Schema")
    print("-" * 80)
    from decayshape.particles import Channel

    channel_schema = Channel.to_json_schema()
    print(json.dumps(channel_schema, indent=2))

    # 3. Lineshape JSON Schema
    print("\n3. Lineshape JSON Schema (Relativistic Breit-Wigner)")
    print("-" * 80)
    from decayshape.lineshapes import RelativisticBreitWigner

    lineshape_schema = RelativisticBreitWigner.to_json_schema()

    # Note: The 's' parameter is automatically excluded from the schema
    # as it's not relevant for frontend configuration
    print(json.dumps(lineshape_schema, indent=2))

    # 4. Excluding additional fields
    print("\n4. Lineshape JSON Schema with Additional Exclusions")
    print("-" * 80)
    # You can exclude additional fields if needed
    minimal_schema = RelativisticBreitWigner.to_json_schema(exclude_fields=["q0"])
    print(json.dumps(minimal_schema, indent=2))

    # 5. K-Matrix JSON Schema
    print("\n5. K-Matrix JSON Schema (Advanced Multi-Channel)")
    print("-" * 80)
    from decayshape.kmatrix_advanced import KMatrixAdvanced

    kmatrix_schema = KMatrixAdvanced.to_json_schema()
    print(json.dumps(kmatrix_schema, indent=2))

    # 6. Direct JSON String Generation
    print("\n6. Direct JSON String Generation")
    print("-" * 80)
    json_string = Particle.to_json_string(indent=2)
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

    # Dictionary to hold all schemas
    all_schemas = {}

    # 1. Relativistic Breit-Wigner
    print("\n1. Generating RelativisticBreitWigner schema...")
    from decayshape.lineshapes import RelativisticBreitWigner

    bw_schema = RelativisticBreitWigner.to_json_schema()
    # Remove current_values
    bw_schema.pop("current_values", None)
    all_schemas["RelativisticBreitWigner"] = bw_schema
    print(f"   -> RelativisticBreitWigner: {len(bw_schema['optimization_parameters'])} parameters")

    # 2. Flatté
    print("\n2. Generating Flatte schema...")
    from decayshape.lineshapes import Flatte

    flatte_schema = Flatte.to_json_schema()
    # Remove current_values
    flatte_schema.pop("current_values", None)
    all_schemas["Flatte"] = flatte_schema
    print(f"   -> Flatte: {len(flatte_schema['optimization_parameters'])} parameters")

    # 3. K-Matrix Advanced
    print("\n3. Generating KMatrixAdvanced schema...")
    from decayshape.kmatrix_advanced import KMatrixAdvanced

    kmatrix_schema = KMatrixAdvanced.to_json_schema()
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

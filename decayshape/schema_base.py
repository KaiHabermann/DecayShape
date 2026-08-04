"""
Leaf-level building blocks shared across the package.

Provides FixedParam (marks values that don't change during optimization) and
JsonSchemaMixin (JSON schema generation for frontend consumption). Kept dependency-free
from the rest of the package so it can be imported by base.py, particles.py, and
threshold.py without creating import cycles.
"""
import json
from typing import Any, Generic, Optional, TypeVar, Union, get_args, get_origin

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

# Template type for marking fixed parameters
T = TypeVar("T")

# Type for general numerical values. float, np.array, jax.numpy.array, etc.

Numerical = Union[float, np.ndarray, jnp.ndarray]


class JsonSchemaMixin:
    """
    Mixin class that provides JSON schema generation for Pydantic models.

    This can be mixed into any Pydantic BaseModel to add to_json_schema() and
    to_json_string() methods for frontend consumption.
    """

    @classmethod
    def to_json_schema(cls, exclude_fields: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Generate a JSON schema representation of the model for frontend use.

        Args:
            exclude_fields: List of field names to exclude from the schema

        Returns:
            Dictionary containing the model structure, parameters, and metadata
        """
        if exclude_fields is None:
            exclude_fields = []

        # Get the class name and description
        class_name = cls.__name__
        class_doc = cls.__doc__ or ""

        # Get model fields information
        model_fields = cls.model_fields

        # Separate fixed and regular parameters
        fixed_params = {}
        regular_params = {}

        for field_name, field_info in model_fields.items():
            # Skip excluded fields
            if field_name in exclude_fields:
                continue

            # Extract field information
            field_type = field_info.annotation
            field_description = field_info.description or ""
            resolved_default = field_info.get_default(call_default_factory=True)
            field_default = resolved_default if resolved_default is not PydanticUndefined else None

            # A discriminated union (e.g. Field(discriminator="kind")) is a choice between several
            # named strategy classes, each with its own parameters - render it as such instead of
            # picking one arbitrary member (the type-hint-only path below has no way to know which).
            if field_info.discriminator is not None:
                param_info = cls._build_discriminated_union_info(field_type, field_info.discriminator, field_default)
                param_info["description"] = field_description
                regular_params[field_name] = param_info
                continue

            # Determine if this is a FixedParam field and get inner type
            inner_type = cls._extract_fixedparam_inner_type(field_type)
            is_fixed_param = inner_type is not None

            # Convert type to JSON-serializable format
            type_info = cls._type_to_json_info(inner_type if is_fixed_param else field_type)

            # Create parameter info
            param_info = {
                "type": type_info["type"],
                "description": field_description,
                "default": cls._serialize_default_value(field_default),
                "constraints": type_info.get("constraints", {}),
                "items": type_info.get("items"),  # For arrays/lists
                "properties": type_info.get("properties"),  # For objects
                "schema": type_info.get("schema"),  # For nested models with JsonSchemaMixin
                "class": type_info.get("class"),  # Class name for object types
                "item_schema": type_info.get("item_schema"),  # Schema for array items
                "optional": type_info.get("optional", False),  # Mark if parameter is optional
            }

            # Remove None values to keep JSON clean
            param_info = {k: v for k, v in param_info.items() if v is not None}

            # Add to appropriate category
            if is_fixed_param:
                fixed_params[field_name] = param_info
            else:
                regular_params[field_name] = param_info

        # Build the complete schema
        schema = {
            "model_type": class_name,
            "description": class_doc.strip(),
            "fixed_parameters": fixed_params,
            "parameters": regular_params,
        }

        return schema

    @classmethod
    def _build_discriminated_union_info(cls, field_type, discriminator: str, default_instance) -> dict[str, Any]:
        """
        Build schema info for a discriminated-union field (e.g. Field(discriminator="kind")): a
        choice between several named strategy classes, each with its own parameter set.

        Args:
            field_type: The field's annotation - Union[ClassA, ClassB, ...]
            discriminator: Name of the literal field each member class uses to tag itself
            default_instance: The field's resolved default value (an instance of one member class)

        Returns:
            Dictionary with "type", "discriminator", "default" (the default instance's
            discriminator value), and "options" (discriminator value -> that class's own schema)
        """
        options = {}
        for member_cls in get_args(field_type):
            member_schema = member_cls.to_json_schema(exclude_fields=[discriminator])
            kind_value = member_cls.model_fields[discriminator].default
            options[kind_value] = {
                "class": member_schema["model_type"],
                "description": member_schema["description"],
                "parameters": member_schema["parameters"],
                "fixed_parameters": member_schema["fixed_parameters"],
            }

        return {
            "type": "discriminated_union",
            "discriminator": discriminator,
            "default": getattr(default_instance, discriminator, None),
            "options": options,
        }

    @classmethod
    def _type_to_json_info(cls, type_hint) -> dict[str, Any]:
        """Convert Python type hints to JSON schema type information."""
        if type_hint is None or type_hint is type(None):
            return {"type": "null"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is str:
            return {"type": "string"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is list:
            return {"type": "array"}
        elif type_hint is dict:
            return {"type": "object"}

        # Handle generic types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Numerical type (Union[float, np.ndarray, jnp.ndarray]) as float
        # Check if this is the Numerical type or matches its Union structure
        if type_hint is Numerical:
            return {"type": "number"}

        # Check if this Union matches Numerical (treat as float)
        if origin is Union:
            # Check if all args are float, np.ndarray, or jnp.ndarray types
            # This matches the Numerical type definition
            try:
                import jax.numpy as jnp
                import numpy as np

                all_numerical = all(
                    arg is float
                    or arg is np.ndarray
                    or arg is jnp.ndarray
                    or (hasattr(arg, "__name__") and arg.__name__ in ("ndarray", "Array"))
                    for arg in args
                )
                if all_numerical and len(args) > 0:
                    return {"type": "number"}
            except ImportError:
                pass

        if origin is list:
            item_type = args[0] if args else Any
            item_info = cls._type_to_json_info(item_type)
            result = {"type": "array"}

            # Simplify items info - just include the basic type info
            items_dict = {"type": item_info["type"]}
            if "class" in item_info:
                items_dict["class"] = item_info["class"]
            result["items"] = items_dict

            # If the item has a nested schema, include it separately
            if "schema" in item_info:
                result["item_schema"] = item_info["schema"]

            return result
        elif origin is dict:
            return {"type": "object"}
        elif origin is Union:
            # Handle Union types (like Optional)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                # This is an Optional type (Union[T, None])
                result = cls._type_to_json_info(non_none_types[0])
                result["optional"] = True
                return result
            else:
                return {"type": "union", "anyOf": [cls._type_to_json_info(arg) for arg in non_none_types]}

        # Handle custom classes that have JsonSchemaMixin
        if isinstance(type_hint, type) and hasattr(type_hint, "__mro__"):
            # Check if this class has JsonSchemaMixin
            if any(base.__name__ == "JsonSchemaMixin" for base in type_hint.__mro__):
                # Create a temporary instance to get the schema structure
                # We'll get the field structure without instantiating
                try:
                    # Get the model fields directly from the class
                    if hasattr(type_hint, "model_fields"):
                        nested_schema = cls._generate_nested_schema(type_hint)
                        return {"type": "object", "class": type_hint.__name__, "schema": nested_schema}
                except Exception:
                    pass

            # For classes without the mixin, just return basic info
            if hasattr(type_hint, "__name__"):
                return {"type": "object", "class": type_hint.__name__}

        # Fallback
        return {"type": "any"}

    @classmethod
    def _extract_fixedparam_inner_type(cls, field_type):
        """
        Extract the inner type from a FixedParam field.

        Handles both standard typing and Pydantic's generic metadata.
        """
        # First try Pydantic's generic metadata
        if hasattr(field_type, "__pydantic_generic_metadata__"):
            metadata = field_type.__pydantic_generic_metadata__
            if metadata.get("origin") is FixedParam and metadata.get("args"):
                return metadata["args"][0]

        # Fall back to standard typing
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union:
            # Handle Optional[FixedParam[...]]
            for arg in args:
                arg_origin = get_origin(arg)
                if arg_origin is FixedParam:
                    arg_args = get_args(arg)
                    return arg_args[0] if arg_args else Any
                # Check if arg itself is a FixedParam subclass with Pydantic metadata
                if hasattr(arg, "__pydantic_generic_metadata__"):
                    metadata = arg.__pydantic_generic_metadata__
                    if metadata.get("origin") is FixedParam and metadata.get("args"):
                        return metadata["args"][0]
        elif origin is FixedParam:
            return args[0] if args else Any
        elif isinstance(field_type, type) and issubclass(field_type, FixedParam):
            return Any

        return None

    @classmethod
    def _generate_nested_schema(cls, model_class) -> dict[str, Any]:
        """Generate a nested schema for a model class that has JsonSchemaMixin."""
        if not hasattr(model_class, "model_fields"):
            return {}

        model_fields = model_class.model_fields
        schema_fields = {}

        for field_name, field_info in model_fields.items():
            field_type = field_info.annotation
            field_description = field_info.description or ""

            # Determine if this is a FixedParam field and get inner type
            inner_type = cls._extract_fixedparam_inner_type(field_type)
            is_fixed_param = inner_type is not None

            # Get type info for the appropriate type
            type_info = cls._type_to_json_info(inner_type if is_fixed_param else field_type)

            field_schema = {
                "type": type_info["type"],
                "description": field_description,
            }

            # Add nested schema if present
            if "schema" in type_info:
                field_schema["schema"] = type_info["schema"]
            if "item_schema" in type_info:
                field_schema["item_schema"] = type_info["item_schema"]
            if "items" in type_info:
                field_schema["items"] = type_info["items"]
            if "class" in type_info:
                field_schema["class"] = type_info["class"]

            schema_fields[field_name] = field_schema

        return schema_fields

    @classmethod
    def _serialize_default_value(cls, value):
        """Serialize default values to JSON-compatible format."""
        if value is None or value is ...:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, list):
            return [cls._serialize_default_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: cls._serialize_default_value(v) for k, v in value.items()}
        elif hasattr(value, "model_dump"):
            # Pydantic model
            return value.model_dump()
        else:
            # Try to convert to string representation
            return str(value)

    def _get_current_values(self, exclude_fields: Optional[list[str]] = None) -> dict[str, Any]:
        """Get current parameter values."""
        if exclude_fields is None:
            exclude_fields = []

        current_values = {}

        # Get parameters
        for field_name, field_value in self.__dict__.items():
            if field_name in exclude_fields:
                continue
            if isinstance(field_value, FixedParam):
                current_values[field_name] = self._serialize_default_value(field_value.value)
            else:
                current_values[field_name] = self._serialize_default_value(field_value)

        return current_values

    @classmethod
    def to_json_string(cls, indent: Optional[int] = 2, exclude_fields: Optional[list[str]] = None) -> str:
        """
        Generate a JSON string representation of the model schema.

        Args:
            indent: Number of spaces for indentation (None for compact)
            exclude_fields: List of field names to exclude from the schema

        Returns:
            JSON string representation
        """
        schema = cls.to_json_schema(exclude_fields=exclude_fields)
        return json.dumps(schema, indent=indent, ensure_ascii=False)


class FixedParam(BaseModel, Generic[T]):
    """Pydantic model for marking fixed parameters that don't change during optimization."""

    value: T = Field(..., description="The fixed value that doesn't change during optimization")

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, item):
        """Forward indexing to the value."""
        return self.value[item]

    def __getattr__(self, name: str):
        """First look for the attribute in the instance, then forward to value."""
        # Check if the attribute exists in the instance's __dict__ or as a property
        if name in self.__dict__ or hasattr(type(self), name):
            return object.__getattribute__(self, name)
        if name.startswith("_") or name in ("value", "model_fields", "model_config"):
            # Don't forward private attributes or Pydantic internals
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.value, name)

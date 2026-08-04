"""
Base classes for lineshapes in hadron physics.

Provides abstract base class that all lineshapes must implement using Pydantic.
"""
from abc import ABC, abstractmethod
from typing import Annotated, Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticUndefined

from .config import config
from .schema_base import FixedParam, JsonSchemaMixin, Numerical, T  # noqa: F401
from .threshold import BarrierFactor, BlattWeisskopfBarrier, ConstantThreshold

# Discriminated union so model_validate() can reconstruct the right ThresholdFunction
# subclass from a plain dict (e.g. after model_dump()) - a bare ThresholdFunction
# annotation is abstract and gives pydantic no way to pick a concrete variant.
AnyThresholdFunction = Annotated[Union[BlattWeisskopfBarrier, BarrierFactor, ConstantThreshold], Field(discriminator="kind")]


class LineshapeBase(BaseModel):
    """Base Pydantic model for all lineshapes."""

    s: Optional[FixedParam[Union[float, Any]]] = Field(
        default_factory=lambda: FixedParam(value=None),
        exclude=True,
        description="Mandelstam variable s (mass squared) or array of s values",
    )

    @field_validator("s", mode="before")
    @classmethod
    def ensure_s_is_array(cls, v):
        if v is None:
            return FixedParam(value=None)
        if isinstance(v, FixedParam) and v.value is None:
            return v
        # If v is already a FixedParam, extract its value for checking
        value = v.value if isinstance(v, FixedParam) else v
        # Check if value is an iterable (but not a string or bytes)
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            # Convert to backend array
            arr = config.backend.array(value)
            # If v is a FixedParam, return a new FixedParam with arr
            if isinstance(v, FixedParam):
                return FixedParam(value=arr)
            else:
                return arr
        return v

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def auto_wrap_fixed_params(cls, values):
        """Automatically wrap values in FixedParam for FixedParam fields."""
        if not isinstance(values, dict):
            return values

        # Get the model fields
        model_fields = cls.model_fields

        for field_name, field_info in model_fields.items():
            if field_name in values:
                field_type = field_info.annotation

                # Determine if the field expects a FixedParam, including Optional[FixedParam]
                expects_fixed = False
                origin = get_origin(field_type)
                if origin is Union:
                    for arg in get_args(field_type):
                        arg_origin = get_origin(arg)
                        if (isinstance(arg, type) and issubclass(arg, FixedParam)) or arg_origin is FixedParam:
                            expects_fixed = True
                            break
                elif isinstance(field_type, type) and issubclass(field_type, FixedParam):
                    expects_fixed = True

                if expects_fixed:
                    value = values[field_name]
                    # If the value is not already a FixedParam and not None, wrap it
                    if value is not None and not isinstance(value, FixedParam):
                        if isinstance(value, dict) and "value" in value:
                            value = value["value"]
                        if value is not None:
                            values[field_name] = FixedParam(value=value)

        return values

    def _parse_args_and_kwargs(self, args, kwargs):
        """Parse positional and keyword arguments."""
        if args:
            if len(args) > len(self.parameter_order):
                raise ValueError(
                    f"Too many positional arguments. Expected at most {len(self.parameter_order)}, got {len(args)}"
                )

            for i, value in enumerate(args):
                param_name = self.parameter_order[i]
                if param_name in kwargs:
                    raise ValueError(f"Parameter '{param_name}' provided both positionally and as keyword argument")
                kwargs[param_name] = value
        return args, kwargs


class Lineshape(LineshapeBase, JsonSchemaMixin, ABC):
    """
    Abstract base class for all lineshapes using Pydantic.

    All lineshapes must implement a __call__ method that takes the mass
    as the first parameter and returns the lineshape value.

    Supports parameter override at call time for optimization.
    """

    threshold_behaviour: AnyThresholdFunction = Field(
        default_factory=BlattWeisskopfBarrier,
        description="Outer barrier-factor strategy applied near the decay channel threshold",
    )

    @property
    @abstractmethod
    def _own_parameter_order(self) -> list[str]:
        """
        Return the order of this lineshape's own parameters for positional arguments.

        Returns:
            List of parameter names in the order they should be provided positionally
        """

    @property
    def parameter_order(self) -> list[str]:
        """
        Return the full order of parameters for positional arguments, including the
        parameters contributed by `threshold_behaviour`.

        Returns:
            List of parameter names in the order they should be provided positionally
        """
        return self._own_parameter_order + self.threshold_behaviour.parameter_order

    def get_fixed_parameters(self) -> dict[str, Any]:
        """Get the fixed parameters that don't change during optimization."""
        fixed_params = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, FixedParam):
                fixed_params[field_name] = field_value.value
        return fixed_params

    def get_optimization_parameters(self) -> dict[str, Any]:
        """Get the default optimization parameters."""
        opt_params = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == "threshold_behaviour":
                continue
            if not isinstance(field_value, FixedParam):
                opt_params[field_name] = field_value
        opt_params.update(self.threshold_behaviour.get_parameters())
        return opt_params

    def parameters(self) -> dict[str, Any]:
        """
        Get parameters in the order specified by parameter_order with their actual values.

        Returns:
            Dictionary with parameter names as keys and their actual instance values as values,
            ordered according to parameter_order
        """
        opt_params = self.get_optimization_parameters()
        return {param_name: opt_params[param_name] for param_name in self.parameter_order if param_name in opt_params}

    def _threshold_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Extract the subset of `params` that belong to `threshold_behaviour`, for passing to its `__call__`.

        Uses all of `threshold_behaviour`'s fields (not just the ones exposed by its
        `parameter_order`), so already-resolved values (e.g. a `q0` a subclass computed
        from `pole_mass` for its own width calculation) are reused rather than recomputed.
        """
        return {name: params[name] for name in self.threshold_behaviour.get_parameters()}

    def _get_parameters(self, *args, **kwargs) -> dict[str, Any]:
        """
        Get parameters with overrides from call arguments.

        Args:
            *args: Positional arguments in the order specified by parameter_order
            **kwargs: Keyword arguments

        Returns:
            Dictionary of parameter names to values

        Raises:
            ValueError: If a parameter is provided both positionally and as keyword
        """
        # Start with optimization parameters
        params = self.get_optimization_parameters().copy()
        args, kwargs = self._parse_args_and_kwargs(args, kwargs)

        # Apply keyword arguments
        for param_name, value in kwargs.items():
            params[param_name] = value

        return params

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[float, Any]:
        """
        Evaluate the lineshape at the s values provided during construction.

        Args:
            *args: Positional parameter overrides
            **kwargs: Keyword parameter overrides

        Returns:
            Lineshape value(s) at the s values from construction
        """

    @classmethod
    def to_json_schema(cls, exclude_fields: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Generate a JSON schema representation of the lineshape for frontend use.

        This excludes the 's' parameter as it will not be set in the frontend.

        Args:
            exclude_fields: Additional field names to exclude (s is always excluded)

        Returns:
            Dictionary containing the lineshape structure, parameters, and metadata
        """
        if exclude_fields is None:
            exclude_fields = []

        # Always exclude 's' for lineshapes; threshold_behaviour is merged in separately below
        exclude_fields = list(exclude_fields) + ["s", "threshold_behaviour"]

        # Use the mixin's base implementation
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

        # Merge in the parameters contributed by threshold_behaviour, so its own
        # optimization/fixed parameters (e.g. r, q0) appear flattened alongside
        # this lineshape's own, exactly like any other call-time parameter.
        threshold_field = model_fields.get("threshold_behaviour")
        if threshold_field is not None:
            threshold_cls = type(threshold_field.get_default(call_default_factory=True))
            # "kind" is the pydantic discriminator used for serialization, not a physics parameter.
            threshold_schema = threshold_cls.to_json_schema(exclude_fields=["kind"])
            regular_params.update(threshold_schema["parameters"])
            fixed_params.update(threshold_schema["fixed_parameters"])

        # Build the complete schema
        schema = {
            "model_type": class_name,
            "description": class_doc.strip(),
            "fixed_parameters": fixed_params,
            "parameters": regular_params,
        }

        # Customize the schema for lineshapes
        schema["lineshape_type"] = schema.pop("model_type")
        schema["optimization_parameters"] = schema.pop("parameters")

        return schema

"""
Base classes for lineshapes in hadron physics.

Provides abstract base class that all lineshapes must implement using Pydantic.
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Dict, List, Optional, TypeVar, Generic, get_origin, get_args
from pydantic import BaseModel, Field, model_validator, field_validator
from .config import config
import json

# Template type for marking fixed parameters
T = TypeVar('T')
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
        if name.startswith('_') or name in ('value', 'model_fields', 'model_config'):
            # Don't forward private attributes or Pydantic internals
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.value, name)

class LineshapeBase(BaseModel):
    """Base Pydantic model for all lineshapes."""
    s: FixedParam[Union[float, Any]] = Field(..., description="Mandelstam variable s (mass squared) or array of s values")
    
    @field_validator('s', mode='before')
    @classmethod
    def ensure_s_is_array(cls, v):
        # If v is already a FixedParam, extract its value for checking
        value = v.value if isinstance(v, FixedParam) else v
        # Check if value is an iterable (but not a string or bytes)
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
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
    
    @model_validator(mode='before')
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
                
                # Check if this is a FixedParam field
                if isinstance(field_type, type) and issubclass(field_type, FixedParam):
                    value = values[field_name]
                    
                    # If the value is not already a FixedParam, wrap it
                    if not isinstance(value, FixedParam):
                        if isinstance(value, dict) and 'value' in value:
                            value = value['value']
                        values[field_name] = FixedParam(value=value)
        
        return values


class Lineshape(LineshapeBase, ABC):
    """
    Abstract base class for all lineshapes using Pydantic.
    
    All lineshapes must implement a __call__ method that takes the mass
    as the first parameter and returns the lineshape value.
    
    Supports parameter override at call time for optimization.
    """
    
    @property
    @abstractmethod
    def parameter_order(self) -> List[str]:
        """
        Return the order of parameters for positional arguments.
        
        Returns:
            List of parameter names in the order they should be provided positionally
        """
        pass
    
    def get_fixed_parameters(self) -> Dict[str, Any]:
        """Get the fixed parameters that don't change during optimization."""
        fixed_params = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, FixedParam):
                fixed_params[field_name] = field_value.value
        return fixed_params
    
    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get the default optimization parameters."""
        opt_params = {}
        for field_name, field_value in self.__dict__.items():
            if not isinstance(field_value, FixedParam):
                opt_params[field_name] = field_value
        return opt_params
    
    def _get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
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
        
        # Apply positional arguments
        if args:
            if len(args) > len(self.parameter_order):
                raise ValueError(f"Too many positional arguments. Expected at most {len(self.parameter_order)}, got {len(args)}")
            
            for i, value in enumerate(args):
                param_name = self.parameter_order[i]
                if param_name in kwargs:
                    raise ValueError(f"Parameter '{param_name}' provided both positionally and as keyword argument")
                params[param_name] = value
        
        # Apply keyword arguments
        for param_name, value in kwargs.items():
            if param_name in params and param_name in [self.parameter_order[i] for i in range(len(args))]:
                raise ValueError(f"Parameter '{param_name}' provided both positionally and as keyword argument")
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
        pass
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate a JSON schema representation of the lineshape for frontend use.
        
        This excludes the 's' parameter as it will not be set in the frontend.
        
        Returns:
            Dictionary containing the lineshape structure, parameters, and metadata
        """
        # Get the class name and description
        class_name = self.__class__.__name__
        class_doc = self.__class__.__doc__ or ""
        
        # Get model fields information
        model_fields = self.model_fields
        
        # Separate fixed and optimization parameters
        fixed_params = {}
        optimization_params = {}
        
        for field_name, field_info in model_fields.items():
            # Skip the 's' parameter as requested
            if field_name == 's':
                continue
                
            # Extract field information
            field_type = field_info.annotation
            field_description = field_info.description or ""
            field_default = field_info.default if field_info.default is not ... else None
            
            # Determine if this is a FixedParam field
            is_fixed_param = False
            inner_type = None
            
            # Check for FixedParam types
            if hasattr(field_type, '__origin__') and field_type.__origin__ is not None:
                origin = get_origin(field_type)
                if origin is FixedParam:
                    is_fixed_param = True
                    args = get_args(field_type)
                    inner_type = args[0] if args else Any
            elif isinstance(field_type, type) and issubclass(field_type, FixedParam):
                is_fixed_param = True
                inner_type = Any
            
            # Convert type to JSON-serializable format
            type_info = self._type_to_json_info(inner_type if is_fixed_param else field_type)
            
            # Create parameter info
            param_info = {
                "type": type_info["type"],
                "description": field_description,
                "default": self._serialize_default_value(field_default),
                "constraints": type_info.get("constraints", {}),
                "items": type_info.get("items"),  # For arrays/lists
                "properties": type_info.get("properties"),  # For objects
            }
            
            # Remove None values to keep JSON clean
            param_info = {k: v for k, v in param_info.items() if v is not None}
            
            # Add to appropriate category
            if is_fixed_param:
                fixed_params[field_name] = param_info
            else:
                optimization_params[field_name] = param_info
        
        # Get parameter order for optimization parameters
        param_order = [p for p in self.parameter_order if p != 's']
        
        # Build the complete schema
        schema = {
            "lineshape_type": class_name,
            "description": class_doc.strip(),
            "fixed_parameters": fixed_params,
            "optimization_parameters": optimization_params,
            "parameter_order": param_order,
            "current_values": self._get_current_values(),
        }
        
        return schema
    
    def _type_to_json_info(self, type_hint) -> Dict[str, Any]:
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
        elif type_hint is list or type_hint is List:
            return {"type": "array"}
        elif type_hint is dict or type_hint is Dict:
            return {"type": "object"}
        
        # Handle generic types
        origin = get_origin(type_hint)
        args = get_args(type_hint)
        
        if origin is list or origin is List:
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": self._type_to_json_info(item_type)
            }
        elif origin is dict or origin is Dict:
            return {"type": "object"}
        elif origin is Union:
            # Handle Union types (like Optional)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                return self._type_to_json_info(non_none_types[0])
            else:
                return {
                    "type": "union",
                    "anyOf": [self._type_to_json_info(arg) for arg in non_none_types]
                }
        
        # Handle custom classes
        if hasattr(type_hint, '__name__'):
            return {"type": "object", "class": type_hint.__name__}
        
        # Fallback
        return {"type": "any"}
    
    def _serialize_default_value(self, value):
        """Serialize default values to JSON-compatible format."""
        if value is None or value is ...:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, list):
            return [self._serialize_default_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_default_value(v) for k, v in value.items()}
        elif hasattr(value, 'model_dump'):
            # Pydantic model
            return value.model_dump()
        else:
            # Try to convert to string representation
            return str(value)
    
    def _get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values (excluding 's')."""
        current_values = {}
        
        # Get fixed parameters
        for field_name, field_value in self.__dict__.items():
            if field_name == 's':
                continue
            if isinstance(field_value, FixedParam):
                current_values[field_name] = self._serialize_default_value(field_value.value)
            else:
                current_values[field_name] = self._serialize_default_value(field_value)
        
        return current_values
    
    def to_json_string(self, indent: Optional[int] = 2) -> str:
        """
        Generate a JSON string representation of the lineshape schema.
        
        Args:
            indent: Number of spaces for indentation (None for compact)
            
        Returns:
            JSON string representation
        """
        schema = self.to_json_schema()
        return json.dumps(schema, indent=indent, ensure_ascii=False)

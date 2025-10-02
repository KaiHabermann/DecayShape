"""
Base classes for lineshapes in hadron physics.

Provides abstract base class that all lineshapes must implement using Pydantic.
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Dict, List, Optional, TypeVar, Generic, get_origin, get_args
from pydantic import BaseModel, Field, model_validator, field_validator
from .config import config

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
        """Forward all attribute access to the value."""
        if name.startswith('_') or name in ('value', 'model_fields', 'model_config'):
            # Don't forward private attributes or Pydantic internals
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.value, name)

class LineshapeBase(BaseModel):
    """Base Pydantic model for all lineshapes."""
    s: FixedParam[Union[float, Any]] = Field(..., description="Mandelstam variable s (mass squared) or array of s values")
    
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
    
    # def model_dump(self) -> Dict[str, Any]:
    #     """Serialize the lineshape to a dictionary."""
    #     return super().model_dump()
    
    # @classmethod
    # def model_validate(cls, data: Dict[str, Any]):
    #     """Deserialize a lineshape from a dictionary."""
    #     return super().model_validate(data)

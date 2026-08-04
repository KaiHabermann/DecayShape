"""
Threshold/barrier-factor behaviour for lineshapes.

A ThresholdFunction is a pluggable strategy for the outer barrier factor that
multiplies a lineshape's amplitude near a decay channel's threshold. Every
Lineshape holds one via its `threshold_behaviour` field; the ThresholdFunction's
own optimization parameters (e.g. `r`, `q0`) are merged into the owning
lineshape's call signature (see `Lineshape.parameter_order`).
"""
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from .config import config
from .particles import Channel
from .schema_base import JsonSchemaMixin
from .utils import angular_momentum_barrier_factor, blatt_weiskopf_form_factor


class ThresholdFunction(BaseModel, JsonSchemaMixin, ABC):
    """Abstract base class for pluggable outer barrier-factor behaviour."""

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def parameter_order(self) -> list[str]:
        """Names of this threshold function's own parameters, in positional order."""

    @abstractmethod
    def __call__(
        self, q: Union[float, Any], channel: Channel, L: int, s0: Optional[float] = None, **kwargs
    ) -> Union[float, Any]:
        """
        Evaluate the threshold/barrier factor.

        Args:
            q: Breakup momentum at the evaluation point(s)
            channel: Decay channel the barrier factor is evaluated for
            L: Angular momentum (un-doubled integer)
            s0: Reference Mandelstam s used to derive a default q0 when not overridden
            **kwargs: Overrides for this threshold function's own parameters

        Returns:
            Threshold/barrier factor value(s)
        """

    def get_parameters(self) -> dict[str, Any]:
        """
        Current stored values of all of this threshold function's fields.

        Note this returns every field, not just the ones exposed by `parameter_order` -
        a field like `q0` that isn't user-set still needs to flow through as `None` so
        callers can tell "unset, compute a default" apart from "explicitly overridden".
        `kind` (the pydantic discriminator used for serialization) is excluded - it is
        not a physics parameter.
        """
        return {name: getattr(self, name) for name in type(self).model_fields if name != "kind"}


class BlattWeisskopfBarrier(ThresholdFunction):
    """Blatt-Weisskopf form factor times angular-momentum barrier factor. The default threshold behaviour."""

    kind: Literal["blatt_weisskopf_barrier"] = "blatt_weisskopf_barrier"
    r: float = Field(1e-3, description="Hadron radius parameter for the Blatt-Weisskopf form factor")
    q0: Optional[float] = Field(None, description="Reference momentum for the barrier factor")

    @property
    def parameter_order(self) -> list[str]:
        params = ["r"]
        if self.q0 is not None:
            params.append("q0")
        return params

    def __call__(
        self,
        q: Union[float, Any],
        channel: Channel,
        L: int,
        s0: Optional[float] = None,
        r: Optional[float] = None,
        q0: Optional[float] = None,
        **kwargs,
    ) -> Union[float, Any]:
        r = self.r if r is None else r
        q0 = self.q0 if q0 is None else q0
        if q0 is None:
            q0 = channel.momentum(s0)
        return blatt_weiskopf_form_factor(q, r, L) * angular_momentum_barrier_factor(q, q0, L)


class BarrierFactor(ThresholdFunction):
    """Angular-momentum barrier factor only, without the Blatt-Weisskopf form factor."""

    kind: Literal["barrier_factor"] = "barrier_factor"
    q0: Optional[float] = Field(None, description="Reference momentum for the barrier factor")

    @property
    def parameter_order(self) -> list[str]:
        return ["q0"] if self.q0 is not None else []

    def __call__(
        self,
        q: Union[float, Any],
        channel: Channel,
        L: int,
        s0: Optional[float] = None,
        q0: Optional[float] = None,
        **kwargs,
    ) -> Union[float, Any]:
        q0 = self.q0 if q0 is None else q0
        if q0 is None:
            q0 = channel.momentum(s0)
        return angular_momentum_barrier_factor(q, q0, L)


class ConstantThreshold(ThresholdFunction):
    """No threshold behaviour: always returns 1."""

    kind: Literal["constant_threshold"] = "constant_threshold"

    @property
    def parameter_order(self) -> list[str]:
        return []

    def __call__(
        self, q: Union[float, Any], channel: Channel, L: int, s0: Optional[float] = None, **kwargs
    ) -> Union[float, Any]:
        return config.backend.ones_like(q)

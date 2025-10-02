"""
Advanced K-matrix implementation using poles and channels.

This module provides a more sophisticated K-matrix implementation
that uses the particle/pole/channel structure for complex multi-channel
resonance analysis.
"""

from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from .base import Lineshape, FixedParam
from .particles import Particle, Channel, CommonParticles
from decayshape import config


class KMatrixAdvanced(Lineshape):
    """
    Advanced K-matrix lineshape using poles and channels.
    
    This implementation allows for multiple poles and channels,
    making it suitable for complex coupled-channel analysis.
    """
    
    # Fixed parameters (channels and particles)
    channels: FixedParam[List[Channel]] = Field(..., description="List of decay channels")
    output_channel: FixedParam[int] = Field(
        default=FixedParam[int](value=0),
        description="Which channel of the F-vector to return (0-indexed)"
    )
    
    # Optimization parameters (poles and couplings)
    pole_masses: List[float] = Field(..., description="List of pole masses")
    production_couplings: List[float] = Field(default_factory=list, description="Production couplings from initial state to each pole (length = n_poles)")
    decay_couplings: List[float] = Field(default_factory=list, description="Decay couplings from each pole to each channel (length = n_poles × n_channels)")
    r: float = Field(default=1.0, description="Hadron radius parameter")
    L: int = Field(default=0, description="Angular momentum of the decay")
    q0: float = Field(default=None, description="Reference momentum")
    
    class Config:
        arbitrary_types_allowed = True
    
    @model_validator(mode='after')
    def validate_and_fill_couplings(self):
        """Validate coupling lengths and fill with defaults if empty."""
        n_poles = len(self.pole_masses)
        n_channels = len(self.channels.value)
        
        # Validate output_channel
        if self.output_channel.value < 0 or self.output_channel.value >= n_channels:
            raise ValueError(f"output_channel must be between 0 and {n_channels-1}, got {self.output_channel.value}")
        
        # Fill production couplings with defaults if empty
        if not self.production_couplings:
            self.production_couplings = [1.0] * n_poles
        elif len(self.production_couplings) != n_poles:
            raise ValueError(f"production_couplings must have length {n_poles}, got {len(self.production_couplings)}")
        
        # Fill decay couplings with defaults if empty
        if not self.decay_couplings:
            self.decay_couplings = [1.0] * (n_poles * n_channels)
        elif len(self.decay_couplings) != n_poles * n_channels:
            raise ValueError(f"decay_couplings must have length {n_poles * n_channels}, got {len(self.decay_couplings)}")
        
        return self
    
    @property
    def parameter_order(self) -> List[str]:
        """Return the order of parameters for positional arguments."""
        n_poles = len(self.pole_masses)
        n_channels = len(self.channels.value)
        
        # Build flat parameter order
        params = []
        
        # Add pole masses
        for i in range(n_poles):
            params.append(f"pole_mass_{i}")
        
        # Add production couplings
        for i in range(n_poles):
            params.append(f"production_coupling_{i}")
        
        # Add decay couplings (pole_index * n_channels + channel_index)
        for pole_idx in range(n_poles):
            for channel_idx in range(n_channels):
                params.append(f"decay_coupling_{pole_idx}_{channel_idx}")
        
        # Add other parameters
        params.extend(["r", "L", "q0"])
        
        return params
    
    def model_post_init(self, __context):
        """Post-initialization to set q0 if not provided."""
        if self.q0 is None:
            # Use the first pole mass as reference
            if self.pole_masses:
                self.q0 = self.pole_masses[0] / 2.0
            else:
                self.q0 = 0.5
    
    def _get_parameters(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get parameters with overrides, handling flat parameter structure.
        """
        # Start with optimization parameters
        params = self.get_optimization_parameters().copy()
        
        # Convert flat parameters back to lists for internal use
        n_poles = len(self.pole_masses)
        n_channels = len(self.channels.value)
        
        # Handle pole masses
        pole_masses = []
        for i in range(n_poles):
            param_name = f"pole_mass_{i}"
            if param_name in kwargs:
                pole_masses.append(kwargs[param_name])
            else:
                pole_masses.append(self.pole_masses[i])
        params["pole_masses"] = pole_masses
        
        # Handle production couplings
        production_couplings = []
        for i in range(n_poles):
            param_name = f"production_coupling_{i}"
            if param_name in kwargs:
                production_couplings.append(kwargs[param_name])
            else:
                production_couplings.append(self.production_couplings[i])
        params["production_couplings"] = production_couplings
        
        # Handle decay couplings
        decay_couplings = []
        for pole_idx in range(n_poles):
            for channel_idx in range(n_channels):
                param_name = f"decay_coupling_{pole_idx}_{channel_idx}"
                flat_idx = pole_idx * n_channels + channel_idx
                if param_name in kwargs:
                    decay_couplings.append(kwargs[param_name])
                else:
                    decay_couplings.append(self.decay_couplings[flat_idx])
        params["decay_couplings"] = decay_couplings
        
        # Handle other parameters
        for param_name in ["r", "L", "q0"]:
            if param_name in kwargs:
                params[param_name] = kwargs[param_name]
        
        return params
    
    def function(self, s, *args, **kwargs) -> Union[float, Any]:
        """
        Evaluate the advanced K-matrix lineshape.
        
        Implements the K-matrix formalism as described in AmpForm documentation:
        1. Build full T-matrix for all channels
        2. Build P-vector from production couplings
        3. Build F-vector from T-matrix and P-vector
        4. Return first entry of F-vector
        """
        # Get parameters with overrides
        params = self._get_parameters(*args, **kwargs)
        
        np = config.backend  # Get backend dynamically
        n_poles = len(params["pole_masses"])
        n_channels = len(self.channels.value)
        
        # Step 1: Build the full T-matrix
        T_matrix = self._build_t_matrix(params, s, n_poles, n_channels)
        
        # Step 2: Build the P-vector
        P_vector = self._build_p_vector(params, s, n_poles, n_channels)
        
        # Step 3: Build the F-vector
        F_vector = self._build_f_vector(T_matrix, P_vector, s, n_channels)
        
        # Step 4: Return the specified channel of the F-vector
        output_idx = self.output_channel.value
        if n_channels == 1:
            # Single channel: F_vector is already 1D
            return F_vector
        else:
            # Multi-channel: F_vector is 2D, return specified channel
            return F_vector[output_idx, :]

    def __call__(self, *args, **kwargs) -> Union[float, Any]:
        return self.function(self.s.value, *args, **kwargs)
    
    def _build_t_matrix(self, params: Dict[str, Any], s: Union[float, Any], 
                       n_poles: int, n_channels: int) -> Union[float, Any]:
        """
        Build the full T-matrix for all channels.
        
        T = K * (I - i*K*rho)^(-1)
        where K is the K-matrix and rho is the phase space factor.
        """
        np = config.backend  # Get backend dynamically
        s_len = config.backend.shape(s)
        if len(s_len) == 0:
            s_len = 1
        else:
            s_len,  = s_len
        
        # Calculate phase space factors for each channel (vectorized)
        rho_list = []
        for channel in self.channels.value:
            val = channel.phase_space_factor(s)
            rho_list.append(val)
        rho = np.stack(rho_list, axis=0)
        
        # Build K-matrix: K_ij = sum_R (g_Ri * g_Rj) / (m_R^2 - s)
        # Vectorized approach: compute all pole contributions at once
        K = np.zeros((n_channels, n_channels, s_len), dtype=complex)
        
        # Convert decay couplings to array for vectorized operations
        g_matrix = np.array(params["decay_couplings"]).reshape(n_poles, n_channels)
        pole_masses = np.array(params["pole_masses"])
        
        # Vectorized computation over all poles and s values
        for R in range(n_poles):
            m_R = pole_masses[R]
            denominator = m_R**2 - s
            
            # Handle the case where s equals a pole mass (add small epsilon)
            epsilon = 1e-10
            denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
            
            # Get decay couplings for this pole
            g_R = g_matrix[R]  # Shape: (n_channels,)
            
            # Vectorized outer product: g_R[:, None] * g_R[None, :] gives (n_channels, n_channels)
            # Then broadcast over s dimension: / denominator[None, None, :] 
            K += g_R[:, None, None] * g_R[None, :, None] / denominator[None, None, :]
        
        # Calculate T-matrix: T = K * (I - i*K*rho)^(-1)
        if n_channels == 1:
            # Single channel: T = K / (1 - i*K*rho)
            T_matrix = K[0, 0] / (1 - 1j * K[0, 0] * rho[0])
        else:
            # Multi-channel case: T = K * (I - i*K*rho)^(-1)
            # Fully vectorized calculation for all s values at once
            I = np.eye(n_channels)
            
            # Use np.diag to create diagonal matrices for all s values (n_channels, n_channels, s_len)
            rho_diag = np.stack([np.diag(rho[:, i]) for i in range(s_len)], axis=-1)
            
            # Compute denominator matrices for all s values
            # I[:, :, None] broadcasts identity to (n_channels, n_channels, s_len)
            # Use np.einsum for proper matrix multiplication over the last dimension
            K_rho_product = np.einsum('ijl,jkl->ikl', K, rho_diag)  # (n_channels, n_channels, s_len)
            denominator_matrices = I[:, :, None] - 1j * K_rho_product
            
            # Vectorized matrix inversion for all s values
            try:
                # Try to use vectorized solve: K @ inv(denominator_matrices)
                # This is equivalent to solving: denominator_matrices @ T = K for T
                T_matrix = np.linalg.solve(
                    np.transpose(denominator_matrices, (2, 0, 1)),  # (s_len, n_channels, n_channels)
                    np.transpose(K, (2, 0, 1))  # (s_len, n_channels, n_channels)
                )
                T_matrix = np.transpose(T_matrix, (1, 2, 0))  # Back to (n_channels, n_channels, s_len)
            except np.linalg.LinAlgError:
                # Fallback to element-wise inversion if vectorized approach fails
                T_matrix = np.zeros((n_channels, n_channels, s_len), dtype=complex)
                for s_idx in range(s_len):
                    K_s = K[:, :, s_idx]
                    denominator_matrix = denominator_matrices[:, :, s_idx]
                    
                    det = np.linalg.det(denominator_matrix)
                    if np.abs(det) < 1e-12:
                        T_matrix[:, :, s_idx] = K_s @ np.linalg.pinv(denominator_matrix)
                    else:
                        T_matrix[:, :, s_idx] = K_s @ np.linalg.inv(denominator_matrix)
        
        return T_matrix
    
    def _build_p_vector(self, params: Dict[str, Any], s: Union[float, Any], 
                       n_poles: int, n_channels: int) -> Union[float, Any]:
        """
        Build the P-vector from production couplings.
        
        P_i = sum_R (beta_R * g_Ri) / (m_R^2 - s)
        where beta_R are the production couplings and g_Ri are the decay couplings.
        """
        np = config.backend  # Get backend dynamically
        s_len = len(s)
        
        # Convert to arrays for vectorized operations
        g_matrix = np.array(params["decay_couplings"]).reshape(n_poles, n_channels)  # (n_poles, n_channels)
        beta_array = np.array(params["production_couplings"])  # (n_poles,)
        pole_masses = np.array(params["pole_masses"])  # (n_poles,)
        
        # Vectorized computation over all poles
        # Shape manipulations for broadcasting:
        # pole_masses: (n_poles,) -> (n_poles, 1) for broadcasting with s
        # s: (s_len,) -> (1, s_len) for broadcasting with pole_masses
        # Result denominator: (n_poles, s_len)
        denominators = pole_masses[:, None]**2 - s[None, :]  # (n_poles, s_len)
        
        # Handle the case where s equals a pole mass (add small epsilon)
        epsilon = 1e-10
        denominators = np.where(np.abs(denominators) < epsilon, epsilon, denominators)
        
        # Vectorized P-vector computation:
        # beta_array: (n_poles,) -> (n_poles, 1, 1) for broadcasting
        # g_matrix: (n_poles, n_channels) -> (n_poles, n_channels, 1) for broadcasting with s
        # denominators: (n_poles, s_len) -> (n_poles, 1, s_len) for broadcasting with channels
        # Result: (n_poles, n_channels, s_len) -> sum over poles -> (n_channels, s_len)
        
        contributions = (beta_array[:, None, None] * 
                        g_matrix[:, :, None] / 
                        denominators[:, None, :])  # (n_poles, n_channels, s_len)
        
        # Sum over all poles to get final P-vector
        P_vector = np.sum(contributions, axis=0)  # (n_channels, s_len)
        
        return P_vector
    
    def _build_f_vector(self, T_matrix: Union[float, Any], P_vector: Union[float, Any], 
                       s: Union[float, Any], n_channels: int) -> Union[float, Any]:
        """
        Build the F-vector from T-matrix and P-vector.
        
        F = (I - i*T*rho)^(-1) * P
        where T is the T-matrix, rho is the phase space factor, and P is the P-vector.
        """
        np = config.backend  # Get backend dynamically
        s_len = len(s)
        
        # Calculate phase space factors for each channel (vectorized)
        rho_list = []
        for channel in self.channels.value:
            val = channel.phase_space_factor(s)
            rho_list.append(val)
        rho = np.stack(rho_list, axis=0)
        
        # Build F-vector: F = (I - i*T*rho)^(-1) * P
        if n_channels == 1:
            # Single channel: F = P / (1 - i*T*rho)
            F_vector = P_vector[0] / (1 - 1j * T_matrix * rho[0])
        else:
            # Multi-channel case: F = (I - i*T*rho)^(-1) * P
            # Fully vectorized calculation for all s values at once
            I = np.eye(n_channels)
            
            # Create diagonal matrices for all s values: shape (n_channels, n_channels, s_len)
            rho_diag = np.stack([np.diag(rho[:, i]) for i in range(s_len)], axis=-1)
            
            # Compute denominator matrices for all s values
            # I[:, :, None] broadcasts identity to (n_channels, n_channels, s_len)
            # Use np.einsum for proper matrix multiplication over the last dimension
            T_rho_product = np.einsum('ijl,jkl->ikl', T_matrix, rho_diag)  # (n_channels, n_channels, s_len)
            denominator_matrices = I[:, :, None] - 1j * T_rho_product
            
            # Vectorized matrix solve for all s values
            try:
                # Solve: denominator_matrices @ F = P for F
                # Transpose to (s_len, n_channels, n_channels) for batch solve
                F_matrix = np.linalg.solve(
                    np.transpose(denominator_matrices, (2, 0, 1)),  # (s_len, n_channels, n_channels)
                    np.transpose(P_vector, (1, 0))[:, :, None]  # (s_len, n_channels, 1)
                )
                F_vector = np.transpose(F_matrix[:, :, 0], (1, 0))  # Back to (n_channels, s_len)
            except np.linalg.LinAlgError:
                # Fallback to element-wise inversion if vectorized approach fails
                F_vector = np.zeros((n_channels, s_len), dtype=complex)
                for s_idx in range(s_len):
                    T_s = T_matrix[:, :, s_idx]
                    P_s = P_vector[:, s_idx]
                    rho_s = rho[:, s_idx]
                    
                    I = np.eye(n_channels)
                    rho_diag_s = np.diag(rho_s)
                    denominator_matrix = I - 1j * T_s @ rho_diag_s
                    
                    # Check if the matrix is invertible
                    det = np.linalg.det(denominator_matrix)
                    if np.abs(det) < 1e-12:
                        # If determinant is too small, use pseudo-inverse
                        F_vector[:, s_idx] = np.linalg.pinv(denominator_matrix) @ P_s
                    else:
                        F_vector[:, s_idx] = np.linalg.inv(denominator_matrix) @ P_s
        
        return F_vector
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about all channels."""
        info = {}
        for i, channel in enumerate(self.channels.value):
            info[f"channel_{i}"] = {
                "particles": [channel.particle1.value, channel.particle2.value],
                "threshold": channel.threshold,
                "total_mass": channel.total_mass
            }
        return info
    
    def get_pole_info(self) -> Dict[str, Any]:
        """Get information about all poles."""
        info = {}
        n_channels = len(self.channels.value)
        for i, pole_mass in enumerate(self.pole_masses):
            info[f"pole_{i}"] = {
                "mass": pole_mass,
                "production_coupling": self.production_couplings[i],
                "decay_couplings": self.decay_couplings[i*n_channels:(i+1)*n_channels]
            }
        return info


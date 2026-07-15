"""
LiDAR data acquisition and processing module.

Handles:
- Data acquisition from Webots LiDAR sensor
- Invalid value filtering and interpolation
- Data normalization
- Optional noise injection for domain randomization
- Optional dropout for robustness
"""

from typing import Optional
import numpy as np
from controller import Lidar  # type: ignore
import config


class LidarManager:
    """
    Manages LiDAR sensor data acquisition and preprocessing.
    
    Provides methods for:
    - Acquiring raw LiDAR measurements
    - Cleaning invalid/missing values
    - Normalizing distances
    - Injecting noise for domain randomization
    - Applying dropout for robustness training
    """

    def __init__(self, lidar_device: Lidar, enable_noise: bool = False, 
                 enable_dropout: bool = False, noise_std: float = 0.02,
                 dropout_prob: float = 0.05):
        """
        Initialize LiDAR manager.
        
        Args:
            lidar_device: Webots Lidar device
            enable_noise: Whether to inject random noise
            enable_dropout: Whether to apply random dropout
            noise_std: Standard deviation of noise as fraction of range
            dropout_prob: Probability of dropout per measurement
        """
        self.lidar = lidar_device
        self.enable_noise = enable_noise
        self.enable_dropout = enable_dropout
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.acquisition_problems = 0

    def get_raw_lidar_mm(self) -> np.ndarray:
        """
        Acquire raw LiDAR measurements and extract sector.
        
        The Webots LiDAR has 361 measurements (-180° to +180°).
        We extract the sector [-100°, +100°] which corresponds to indices [80, 281).
        
        Returns:
            Normalized sector array of shape (LIDAR_SECTOR_SIZE,) in millimeters
        """
        # Get full 361-measurement range image from Webots
        raw = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)
        
        # Extract sector [-100° to +100°] → 201 values (indices 80-280)
        sector = raw[80:281]
        
        return sector

    def _clean_invalid_values(self, sector: np.ndarray) -> np.ndarray:
        """
        Replace invalid measurements (0 or infinity) with NaN for interpolation.
        
        Args:
            sector: Raw sector measurements
            
        Returns:
            Sector with NaN values where measurements are invalid
        """
        sector = np.where((sector == 0) | np.isinf(sector), np.nan, sector)
        return sector

    def _interpolate_missing_values(self, sector: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values using forward and backward fill.
        
        Uses forward fill (left-to-right) with backward fill (right-to-left) fallback
        to handle edge cases.
        
        Args:
            sector: Sector with NaN values
            
        Returns:
            Interpolated sector array
        """
        # Forward fill (left to right)
        left = sector.copy()
        mask = np.isnan(left)
        idx = np.where(~mask, np.arange(len(left)), 0)
        np.maximum.accumulate(idx, out=idx)
        left = left[idx]
        
        # Backward fill (right to left)
        right = sector[::-1].copy()
        mask = np.isnan(right)
        idx = np.where(~mask, np.arange(len(right)), 0)
        np.maximum.accumulate(idx, out=idx)
        right = right[idx][::-1]
        
        # Combine: prefer forward fill, fallback to backward fill
        sector_filled = np.where(np.isnan(left), right, left)
        
        return sector_filled
    
    def get_lidar_cleaned(self) -> np.ndarray:
        """
        Acquire and clean LiDAR measurements.
        
        Returns:
            Cleaned sector array of shape (LIDAR_SECTOR_SIZE,) in millimeters
        """
        sector = self.get_raw_lidar_mm()
        sector = self._clean_invalid_values(sector)
        sector = self._interpolate_missing_values(sector)
        return sector
        

    def _apply_noise(self, sector: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to LiDAR measurements (domain randomization).
        
        Args:
            sector: Cleaned sector measurements
            
        Returns:
            Noisy sector array
        """
        if not self.enable_noise:
            return sector
        
        noise = np.random.normal(
            0, 
            self.noise_std * config.LIDAR_RANGE_MM, 
            size=sector.shape
        )
        noisy_sector = sector + noise
        # Clip to valid range
        noisy_sector = np.clip(noisy_sector, 0, config.LIDAR_RANGE_MM * 1.5)
        return noisy_sector

    def _apply_dropout(self, sector: np.ndarray) -> np.ndarray:
        """
        Apply random dropout to simulate sensor failures.
        
        Args:
            sector: Noisy sector measurements
            
        Returns:
            Sector with dropout applied
        """
        if not self.enable_dropout:
            return sector
        
        dropout_mask = np.random.random(sector.shape) < self.dropout_prob
        # Replace dropped values with neighboring values
        sector_copy = sector.copy()
        for i in np.where(dropout_mask)[0]:
            # Use neighbor average or edge value
            if i == 0:
                sector_copy[i] = sector[i+1]
            elif i == len(sector) - 1:
                sector_copy[i] = sector[i-1]
            else:
                sector_copy[i] = (sector[i-1] + sector[i+1]) / 2.0
        return sector_copy

    def get_lidar_normalized(self) -> np.ndarray:
        """
        Acquire, clean, and normalize LiDAR measurements.
        
        Process:
        1. Get raw measurements from Webots
        2. Clean invalid values (0, inf)
        3. Interpolate missing values
        4. Optionally apply noise
        5. Optionally apply dropout
        6. Normalize to [0, 1]
        
        Returns:
            Normalized LiDAR sector array of shape (LIDAR_SECTOR_SIZE,)
        """
        # Get raw measurements in millimeters
        sector = self.get_raw_lidar_mm()
        
        # Convert to millimeters (Webots returns in meters for this sensor)
        sector = sector * 1000.0
        
        # Clean invalid values
        sector = self._clean_invalid_values(sector)
        
        # Interpolate missing values
        sector = self._interpolate_missing_values(sector)
        
        # Apply domain randomization (if enabled)
        sector = self._apply_noise(sector)
        sector = self._apply_dropout(sector)
        
        # Normalize to [0, 1]
        normalized = (sector / config.LIDAR_RANGE_MM).astype(np.float32)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized

    def get_lidar_with_retry(self, max_retries: int = config.MAX_LIDAR_RETRY) -> np.ndarray:
        """
        Acquire LiDAR data with retry logic for acquisition failures.
        
        Validates that center measurement is non-zero (sanity check).
        Retries up to max_retries times if validation fails.
        
        Args:
            max_retries: Maximum retry attempts
            
        Returns:
            Valid normalized LiDAR data
        """
        retry_count = 0
        lidar_data = self.get_lidar_cleaned()
        
        # Sanity check: center should not be zero
        while lidar_data[config.LIDAR_SECTOR_SIZE // 2] == 0 and retry_count < max_retries:
            self.acquisition_problems += 1
            print(f" LiDAR acquisition problem #{self.acquisition_problems}")
            retry_count += 1
            lidar_data = self.get_lidar_cleaned()
        
        lidar_data = self.get_lidar_normalized()
        return lidar_data

    def reset_problem_counter(self) -> None:
        """Reset the acquisition problems counter."""
        self.acquisition_problems = 0

    def get_problem_count(self) -> int:
        """Get current acquisition problems count."""
        return self.acquisition_problems

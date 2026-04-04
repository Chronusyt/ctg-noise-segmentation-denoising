"""
Signal Quality Assessment Module

This module detects unreliable segments in FHR (Fetal Heart Rate) signals.

CONSERVATIVE approach - only flag obvious artifacts:
1. Signal dropout (values <= 0 or >= 255)
2. Boundary saturation (<= 50 or >= 220)
3. Extreme jumps that are clearly physiologically impossible

The output is a binary mask where:
- 0 = reliable signal
- 1 = unreliable signal (artifacts, dropout)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SignalQualityConfig:
    """
    Configuration parameters for signal quality assessment.
    
    Conservative defaults to minimize false positives.
    
    Attributes:
        lower_bound: Lower physiological FHR bound (bpm)
        upper_bound: Upper physiological FHR bound (bpm)
        max_jump: Maximum plausible FHR change per sample (bpm)
                  Very conservative - only flag extreme jumps
        merge_gap_seconds: Maximum gap (seconds) between regions to merge (0 = disabled)
    """
    lower_bound: int = 50
    upper_bound: int = 220
    max_jump: float = 30.0  # only flag >30 bpm jumps
    merge_gap_seconds: float = 30.0  # Merge regions within 15 seconds of each other


class SignalQualityAssessor:
    """
    Assess FHR signal quality and generate reliability mask.
    
    CONSERVATIVE detection - only flags obvious artifacts:
    1. Boundary saturation (<=50 or >=220)
    2. Extreme jumps (>30 bpm between samples)
    
    Post-processing:
    - Merge nearby artifact regions (configurable gap threshold)
    """
    
    def __init__(self, config: Optional[SignalQualityConfig] = None, sample_rate: float = 4.0):
        """
        Initialize the signal quality assessor.
        
        Args:
            config: Configuration parameters (uses defaults if None)
            sample_rate: Sampling rate in Hz (default 4.0)
        """
        self.config = config if config is not None else SignalQualityConfig()
        self.sample_rate = sample_rate
    
    def assess(self, fhr: np.ndarray) -> np.ndarray:
        """
        Assess signal quality and return reliability mask.
        
        CONSERVATIVE approach - only flag clear artifacts.
        
        Args:
            fhr: FHR signal (numpy array)
            
        Returns:
            Binary mask (0=reliable, 1=unreliable) same length as input
        """
        # Initialize mask (0 = reliable)
        mask = np.zeros(len(fhr), dtype=np.uint8)
        
        # 1. Mark boundary saturation (<=50 or >=220)
        mask |= self._detect_boundary_saturation(fhr)
        
        # 2. Mark extreme jumps (>30 bpm) - clearly physiologically impossible
        mask |= self._detect_extreme_jumps(fhr)
        
        # 3. Merge nearby artifact regions if configured
        if self.config.merge_gap_seconds > 0:
            mask = self._merge_nearby_regions(mask)
        
        return mask
    
    def _merge_nearby_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Merge artifact regions that are close to each other.
        
        If two artifact regions are separated by a gap smaller than 
        merge_gap_seconds, they are merged into one continuous region.
        
        Args:
            mask: Binary mask (0=reliable, 1=unreliable)
            
        Returns:
            Binary mask with merged regions
        """
        # Convert gap threshold to samples
        gap_samples = int(self.config.merge_gap_seconds * self.sample_rate)
        
        if gap_samples < 1:
            return mask
        
        # Find artifact regions
        mask = mask.copy()
        padded = np.concatenate([[0], mask, [0]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) < 2:
            return mask
        
        # Merge regions that are close together
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - ends[i]
            if gap <= gap_samples:
                # Fill the gap
                mask[ends[i]:starts[i + 1]] = 1
        
        return mask
    
    def _detect_boundary_saturation(self, fhr: np.ndarray) -> np.ndarray:
        """
        Detect segments at or beyond boundary saturation thresholds.
        
        Args:
            fhr: FHR signal
            
        Returns:
            Binary mask marking saturated segments
        """
        mask = np.zeros(len(fhr), dtype=np.uint8)
        
        lower = self.config.lower_bound
        upper = self.config.upper_bound
        mask[(fhr <= lower) | (fhr >= upper)] = 1
        
        return mask
    
    def _detect_extreme_jumps(self, fhr: np.ndarray) -> np.ndarray:
        """
        Detect extreme jumps (>30 bpm between consecutive samples).
        
        Only flags clearly physiologically impossible jumps.
        Does NOT flag transitions to/from dropout values.
        
        Args:
            fhr: FHR signal
            
        Returns:
            Binary mask marking extreme jump points
        """
        mask = np.zeros(len(fhr), dtype=np.uint8)
        
        if len(fhr) < 2:
            return mask
        
        fhr_int = fhr.astype(np.int32)
        
        for i in range(1, len(fhr)):
            # Skip if either value is dropout (<=0 or >=255)
            if fhr[i] <= 0 or fhr[i] >= 255 or fhr[i-1] <= 0 or fhr[i-1] >= 255:
                continue
            
            # Skip if either value is at saturation boundary
            if fhr[i] <= self.config.lower_bound or fhr[i] >= self.config.upper_bound or fhr[i-1] <= self.config.lower_bound or fhr[i-1] >= self.config.upper_bound:
                continue
            
            jump = abs(fhr_int[i] - fhr_int[i-1])
            
            # Only mark extreme jumps
            if jump > self.config.max_jump:
                mask[i] = 1
                mask[i-1] = 1
        
        return mask
    
    
    
    def get_statistics(self, fhr: np.ndarray, mask: np.ndarray) -> dict:
        """
        Compute statistics about signal quality.
        
        Args:
            fhr: FHR signal
            mask: Reliability mask (0=reliable, 1=unreliable)
            
        Returns:
            Dictionary with quality statistics
        """
        total_samples = len(fhr)
        unreliable_samples = np.sum(mask)
        reliable_samples = total_samples - unreliable_samples
        
        return {
            'total_samples': int(total_samples),
            'reliable_samples': int(reliable_samples),
            'unreliable_samples': int(unreliable_samples),
            'reliability_percent': float(reliable_samples / total_samples * 100),
            'duration_minutes': float(total_samples / self.sample_rate / 60),
            'unreliable_minutes': float(unreliable_samples / self.sample_rate / 60),
        }


def assess_signal_quality(fhr: np.ndarray, 
                         config: Optional[SignalQualityConfig] = None,
                         sample_rate: float = 4.0) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to assess FHR signal quality.
    
    Args:
        fhr: FHR signal (numpy array)
        config: Optional configuration parameters
        sample_rate: Sampling rate in Hz (default 4.0)
        
    Returns:
        Tuple of (mask, statistics)
        - mask: Binary array (0=reliable, 1=unreliable)
        - statistics: Dictionary with quality metrics
    """
    assessor = SignalQualityAssessor(config, sample_rate=sample_rate)
    mask = assessor.assess(fhr)
    stats = assessor.get_statistics(fhr, mask)
    return mask, stats

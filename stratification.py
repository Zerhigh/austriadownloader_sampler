from pydantic import BaseModel, field_validator
from typing import Dict, Union


class Stratification(BaseModel):
    """
    A class for stratified sampling based on land cover distribution.

    Attributes:
        clc_distribution (Dict[int, float]): Land cover class frequencies.
        window_threshold (Dict[int, float]): Minimum required values for valid sampling.
        num_samples (Union[int, str]): Number of samples ('max' for all available).
    """

    clc_distribution: Dict[int, float]
    window_threshold: Dict[int, float]
    num_samples: Union[int, str]

    @property
    def clc_filtered(self) -> Dict[int, float]:
        """
        Filters out land cover classes with zero frequency.

        Returns:
            Dict[int, float]: A dictionary containing only land cover classes with a frequency > 0.
        """
        return {k: v for k, v in self.clc_distribution.items() if v > 0}

    @field_validator("clc_distribution")
    @classmethod
    def validate_clc_distribution(cls, value: Dict[int, float]) -> Dict[int, float]:
        """Ensures clc_distribution has integer keys and float values."""
        if not all(isinstance(k, int) and isinstance(v, (int, float)) for k, v in value.items()):
            raise ValueError("clc_distribution must have integer keys and numeric (int or float) values.")

        dists = sum(list(value.values()))
        if dists != 1:
            raise ValueError(f'Error: Distribution levels are != 1: {dists}')

        return value

    @field_validator("window_threshold")
    @classmethod
    def validate_window_threshold(cls, value: Dict[int, float]) -> Dict[int, float]:
        """Ensures window_threshold has integer keys and float values."""
        if not all(isinstance(k, int) and isinstance(v, (int, float)) for k, v in value.items()):
            raise ValueError("window_threshold must have integer keys and numeric (int or float) values.")
        return value

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, value: Union[int, str]) -> Union[int, str]:
        """Validates num_samples to be either a positive integer or 'max'."""
        if not (value == "max" or (isinstance(value, int) and value > 0)):
            raise ValueError("num_samples must be 'max' or a positive integer.")
        return value



from pydantic import BaseModel, field_validator
from typing import Tuple


class ImageConfig(BaseModel):
    """
    A class for configuring image properties related to raster processing.

    Attributes:
        pixel_size (float): The size of each pixel in meters.
        shape (Tuple[int, int, int]): The shape of the image as (bands, height, width).
        raster_size (int): A predefined constant for raster window calculations.
    """

    pixel_size: float
    shape: Tuple[int, int, int]
    raster_size: int = 100  # Corine defined raster size

    @property
    def window_size(self) -> int:
        """
        Calculates the window size based on pixel size and image shape.

        Returns:
            int: The computed window size.
        """
        width = (self.pixel_size * self.shape[1]) // 2
        # Reduce window size by 1 to account for sampled pixel width (from 0 to 100m in one direction)
        return int(width // self.raster_size) - 1

    @field_validator("pixel_size")
    @classmethod
    def validate_pixel_size(cls, value: float) -> float:
        """Ensures pixel_size is a positive float."""
        if value <= 0:
            raise ValueError("pixel_size must be a positive float.")
        return value

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Validates that shape is a tuple of three positive integers."""
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or not all(isinstance(dim, int) and dim > 0 for dim in value)
        ):
            raise ValueError("shape must be a tuple of three positive integers (bands, height, width).")
        return value

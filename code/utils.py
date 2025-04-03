import time
from contextlib import contextmanager
from typing import Any, Dict, Generator

import rasterio
from loguru import logger
from pystac import Item


@contextmanager
def timer(task_name: str) -> Generator[None, None, None]:
    """
    Context manager to time a code block.

    Args:
        task_name: Name of the task being timed

    Example:
        with timer("Processing images"):
            # Do some work
            process_images()
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{task_name} completed in {elapsed:.2f} seconds")


def get_epsg_code(image: Item) -> int:
    # Approach 1: Check the properties directly
    if "proj:epsg" in image.properties:
        epsg_code = image.properties["proj:epsg"]
        logger.info(f"EPSG code from properties: {epsg_code}")

    # Check if "projection" appears in any of the stac_extensions strings
    has_projection = any("projection" in ext for ext in image.stac_extensions)
    if has_projection:
        from pystac.extensions.projection import ProjectionExtension

        proj_ext = ProjectionExtension.ext(image)
        epsg_code = proj_ext.epsg
        if epsg_code is None:
            logger.error("EPSG code is None")
        else:
            logger.info(f"EPSG code from projection extension: {epsg_code}")

    # 3. MGRS-30UWC indicates UTM zone 30 North (EPSG:32630)
    if "grid:code" in image.properties:
        mgrs_code = image.properties["grid:code"]
        logger.info(f"MGRS code: {mgrs_code}")

        # Extract UTM zone from MGRS code (format: "MGRS-30UWC")
        if mgrs_code.startswith("MGRS-"):
            utm_zone = mgrs_code[5:7]  # Get "30" from "MGRS-30UWC"
            hemisphere = mgrs_code[7:8]  # Get "U" from "MGRS-30UWC"

            # "U" indicates Northern hemisphere in MGRS
            if hemisphere in "NPQRSTUVZ":  # Northern hemisphere bands
                epsg_code = int(f"326{utm_zone}")  # Northern UTM zones: 326xx
            else:  # Southern hemisphere
                epsg_code = int(f"327{utm_zone}")  # Southern UTM zones: 327xx

            logger.info(f"Derived EPSG code from MGRS: {epsg_code}")

    return epsg_code


def is_cog(file_path: str) -> Dict[str, Any]:
    """
    Check if a file has Cloud Optimized GeoTIFF (COG) characteristics.

    Args:
        file_path: Path to the raster file

    Returns:
        Dictionary with COG validation results
    """
    result = {
        "is_cog": False,
        "has_overviews": False,
        "is_tiled": False,
        "has_internal_tiling": False,
        "errors": [],
    }

    try:
        with rasterio.open(file_path) as src:
            # Check for internal tiling
            profile = src.profile
            result["is_tiled"] = profile.get("tiled", False)

            # Check for block size (internal tiling)
            if profile.get("blockxsize") and profile.get("blockysize"):
                result["has_internal_tiling"] = True

            # Check for overviews
            result["has_overviews"] = len(src.overviews(1)) > 0

            # Basic COG criteria: has overviews and is internally tiled
            result["is_cog"] = result["has_overviews"] and result["has_internal_tiling"]

            # Additional info
            result["driver"] = profile.get("driver")
            result["profile"] = profile

    except Exception as e:
        result["errors"].append(str(e))

    return result


def log_raster_info(file_path: str) -> None:
    """
    Print detailed information about a raster file.

    Args:
        file_path: Path to the raster file
    """
    with rasterio.open(file_path) as src:
        profile = src.profile
        cog_status = is_cog(file_path)

        logger.info(f"Raster Information for: {file_path}")
        logger.info(f"Driver: {profile['driver']}")
        logger.info(f"Dimensions: {src.width} x {src.height} pixels")
        logger.info(f"Bands: {src.count}")
        logger.info(f"CRS: {src.crs}")
        logger.info(f"Data Type: {src.dtypes[0]}")
        logger.info(f"No Data Value: {src.nodata}")
        logger.info(f"Internal Tiling: {profile.get('tiled', False)}")
        logger.info(f"Block Size: {src.block_shapes}")

        # Print overview information
        overview_factors = [src.overviews(i + 1) for i in range(src.count)]
        logger.info(f"Overviews: {overview_factors[0] if overview_factors else 'None'}")

        # Is it a COG?
        logger.info(f"Is Cloud Optimized GeoTIFF: {cog_status['is_cog']}")
        if not cog_status["is_cog"]:
            missing = []
            if not cog_status["has_overviews"]:
                missing.append("overviews")
            if not cog_status["has_internal_tiling"]:
                missing.append("internal tiling")
            if missing:
                logger.info(f"Missing COG characteristics: {', '.join(missing)}")


class Spinner:
    """
    A simple spinner class to show progress during long-running operations.

    Example:
        with Spinner("Processing data"):
            # Do some long operation
            time.sleep(5)
    """

    def __init__(self, message: str = "Working", delay: float = 0.1) -> None:
        """
        Initialize the spinner.

        Args:
            message: Message to display alongside the spinner
            delay: Time between spinner updates in seconds
        """
        self.message = message
        self.delay = delay
        self.spinner_generator = self._create_spinner()
        self.running = False
        self.spinner_thread = None

    def _create_spinner(self) -> Generator[str, None, None]:
        """Create a generator for spinner characters."""
        while True:
            for char in "|/-\\":
                yield f"{char}"

    def __enter__(self) -> "Spinner":
        import sys
        import threading

        self.running = True

        def spin() -> None:
            while self.running:
                sys.stdout.write(f"\r{self.message} {next(self.spinner_generator)} ")
                sys.stdout.flush()
                time.sleep(self.delay)

        self.spinner_thread = threading.Thread(target=spin)
        self.spinner_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        import sys

        sys.stdout.write(f"\r{self.message} Done!      \n")

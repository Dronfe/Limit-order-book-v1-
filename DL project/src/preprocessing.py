"""Preprocessing utilities for Chandrayaan-2 DFSAR and OHRC products.

This module reduces DFSAR speckle, co-registers OHRC optical imagery onto the
DFSAR grid, and stacks the aligned layers into a multi-channel tensor suitable
for the dual-stream ice-detection network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

try:
    from osgeo import gdal, gdalconst, osr
except ImportError:  # pragma: no cover - optional at import time for unit tests
    gdal = None
    gdalconst = None
    osr = None

logger = logging.getLogger(__name__)

DFSAR_CHANNELS: tuple[str, ...] = ("HH", "HV", "VH", "VV")
DEFAULT_LEE_WINDOW = 7
DEFAULT_LEE_LOOKS = 1.0


def _to_float32(array: np.ndarray) -> np.ndarray:
    """Cast imagery to float32 and replace non-finite values with zeros."""
    data = np.asarray(array, dtype=np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _odd_window(window_size: int) -> int:
    """Ensure the Lee filter kernel is a positive odd integer."""
    if window_size < 3:
        raise ValueError("window_size must be >= 3")
    if window_size % 2 == 0:
        window_size += 1
        logger.warning("Lee filter window must be odd; using %d", window_size)
    return window_size


def lee_filter(
    image: np.ndarray,
    window_size: int = DEFAULT_LEE_WINDOW,
    n_looks: float = DEFAULT_LEE_LOOKS,
) -> np.ndarray:
    """Apply the Lee speckle filter to a single DFSAR intensity channel.

    The Lee filter estimates a locally linear MMSE solution under a
    multiplicative noise model:

        I_hat = mu + W * (I - mu)
        W = 1 - (Cu^2 / Ci^2)

    where ``Cu`` is the noise variation coefficient (``1 / sqrt(n_looks)``)
    and ``Ci`` is the local coefficient of variation.

    Args:
        image: 2-D array of radar backscatter (linear intensity or amplitude^2).
        window_size: Odd square kernel size used for local statistics.
        n_looks: Equivalent number of looks used to set the noise CV.

    Returns:
        Speckle-reduced image with the same shape as ``image``.
    """
    if image.ndim != 2:
        raise ValueError(f"Lee filter expects a 2-D array, got shape {image.shape}")

    window_size = _odd_window(window_size)
    src = _to_float32(image)
    kernel = (window_size, window_size)

    local_mean = cv2.blur(src, kernel)
    local_mean_sq = cv2.blur(src * src, kernel)
    local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0.0)

    cu2 = 1.0 / max(float(n_looks), 1e-6)
    # Avoid division by zero in homogeneous / dark pixels.
    ci2 = local_var / np.maximum(local_mean * local_mean, 1e-12)
    weight = 1.0 - (cu2 / np.maximum(ci2, 1e-12))
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (src - local_mean)
    return np.clip(filtered, 0.0, None).astype(np.float32)


def lee_filter_polarimetric(
    channels: Mapping[str, np.ndarray] | np.ndarray,
    window_size: int = DEFAULT_LEE_WINDOW,
    n_looks: float = DEFAULT_LEE_LOOKS,
    channel_order: Sequence[str] = DFSAR_CHANNELS,
) -> np.ndarray:
    """Apply the Lee filter independently to each DFSAR polarimetric channel.

    Args:
        channels: Mapping of polarisation name to 2-D array, or an array of
            shape ``(C, H, W)`` ordered as ``channel_order``.
        window_size: Odd square kernel size.
        n_looks: Equivalent number of looks.
        channel_order: Names used when ``channels`` is a mapping.

    Returns:
        Array of shape ``(C, H, W)`` with speckle-reduced polarimetric bands.
    """
    if isinstance(channels, np.ndarray):
        if channels.ndim != 3:
            raise ValueError(f"Expected (C, H, W) array, got {channels.shape}")
        stack = [_to_float32(band) for band in channels]
    else:
        stack = []
        for name in channel_order:
            if name not in channels:
                raise KeyError(f"Missing DFSAR channel '{name}'")
            stack.append(_to_float32(channels[name]))

    filtered = [lee_filter(band, window_size=window_size, n_looks=n_looks) for band in stack]
    return np.stack(filtered, axis=0)


def _percentile_normalize(
    array: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Linearly stretch an image to [0, 1] using robust percentiles."""
    data = _to_float32(array)
    sample = data[valid_mask] if valid_mask is not None else data
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    lo, hi = np.percentile(sample, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = (data - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def _read_raster(path: str | Path) -> tuple[np.ndarray, rasterio.Affine, str | None]:
    """Read a raster as float32 with its affine transform and CRS."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        return data, src.transform, src.crs


def _gdal_reproject_to_reference(
    src_path: str | Path,
    ref_path: str | Path,
    resampling: int | None = None,
) -> np.ndarray:
    """Resample ``src_path`` onto the grid of ``ref_path`` using GDAL Warp.

    This path is preferred when both datasets are on disk because GDAL handles
    CRS transforms, geotransform alignment, and nodata in a single warp.
    """
    if gdal is None:
        raise ImportError("GDAL (osgeo) is required for disk-based co-registration")

    if resampling is None:
        resampling = gdalconst.GRA_Bilinear

    src_ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    ref_ds = gdal.Open(str(ref_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"Unable to open OHRC raster: {src_path}")
    if ref_ds is None:
        raise FileNotFoundError(f"Unable to open DFSAR raster: {ref_path}")

    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    width, height = ref_ds.RasterXSize, ref_ds.RasterYSize
    bands = src_ds.RasterCount
    dtype = gdal.GDT_Float32

    mem_driver = gdal.GetDriverByName("MEM")
    dst_ds = mem_driver.Create("", width, height, bands, dtype)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)

    src_wkt = src_ds.GetProjection()
    if src_wkt and osr is not None:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_wkt)
        src_wkt = src_srs.ExportToWkt()
    dst_wkt = proj
    if dst_wkt and osr is not None:
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(dst_wkt)
        dst_wkt = dst_srs.ExportToWkt()

    gdal.ReprojectImage(src_ds, dst_ds, src_wkt, dst_wkt, resampling)

    warped = dst_ds.ReadAsArray().astype(np.float32)
    src_ds = None
    ref_ds = None
    dst_ds = None
    if warped.ndim == 2:
        warped = warped[np.newaxis, ...]
    return warped


def coregister_ohrc_to_dfsar(
    ohrc_path: str | Path,
    dfsar_path: str | Path,
    resampling: Resampling = Resampling.bilinear,
    p_low: float = 1.0,
    p_high: float = 99.0,
    use_gdal: bool = True,
) -> np.ndarray:
    """Load OHRC imagery, resample it onto the DFSAR grid, and normalize it.

    Spatial co-registration uses the DFSAR geotransform, CRS, width, and
    height as the target grid so that optical and radar pixels correspond
    one-to-one.

    Args:
        ohrc_path: Path to the OHRC GeoTIFF (or GDAL-readable) product.
        dfsar_path: Path to a DFSAR GeoTIFF that defines the reference grid.
            Any polarimetric channel may be used as the geometric reference.
        resampling: Rasterio resampling kernel (used in the rasterio fallback).
        p_low: Lower percentile for robust [0, 1] normalization.
        p_high: Upper percentile for robust [0, 1] normalization.
        use_gdal: If True, warp with GDAL; otherwise use rasterio.warp.

    Returns:
        Normalized OHRC array of shape ``(C, H, W)`` matching the DFSAR grid.
    """
    ohrc_path = Path(ohrc_path)
    dfsar_path = Path(dfsar_path)

    with rasterio.open(dfsar_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height, dst_width = ref.height, ref.width
        dst_nodata = ref.nodata

    if use_gdal and gdal is not None:
        warped = _gdal_reproject_to_reference(ohrc_path, dfsar_path)
    else:
        with rasterio.open(ohrc_path) as src:
            warped = np.zeros((src.count, dst_height, dst_width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=warped,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata if dst_nodata is not None else 0.0,
            )

    valid = np.isfinite(warped) & (warped != 0)
    normalized_bands = [
        _percentile_normalize(band, p_low=p_low, p_high=p_high, valid_mask=valid[i])
        for i, band in enumerate(warped)
    ]
    return np.stack(normalized_bands, axis=0)


def normalize_dfsar(
    radar: np.ndarray,
    db_scale: bool = True,
    clip_db: tuple[float, float] = (-35.0, 5.0),
) -> np.ndarray:
    """Convert linear DFSAR intensity to a network-friendly [0, 1] range.

    Args:
        radar: Array of shape ``(C, H, W)`` in linear power (or already dB).
        db_scale: If True, convert linear intensity to dB before stretching.
        clip_db: dB clip range used when ``db_scale`` is True.

    Returns:
        Normalized radar cube of shape ``(C, H, W)``.
    """
    data = _to_float32(radar)
    if db_scale:
        data = 10.0 * np.log10(np.maximum(data, 1e-12))
        lo, hi = clip_db
        data = np.clip(data, lo, hi)
        data = (data - lo) / max(hi - lo, 1e-6)
    else:
        bands = [_percentile_normalize(band) for band in data]
        data = np.stack(bands, axis=0)
    return data.astype(np.float32)


def stack_radar_optical(
    radar: np.ndarray,
    optical: np.ndarray,
    normalize_radar: bool = True,
    db_scale: bool = True,
) -> np.ndarray:
    """Stack preprocessed DFSAR and OHRC layers into one multi-channel cube.

    Channel order is ``[HH, HV, VH, VV, OHRC_1, ..., OHRC_C]``. The returned
    array is ready to be split back into radar / optical streams or saved as
    a training patch.

    Args:
        radar: Speckle-filtered DFSAR array ``(4, H, W)``.
        optical: Co-registered OHRC array ``(C, H, W)`` or ``(H, W)``.
        normalize_radar: Apply ``normalize_dfsar`` before stacking.
        db_scale: Passed through to ``normalize_dfsar``.

    Returns:
        Float32 array of shape ``(4 + C, H, W)``.
    """
    radar = _to_float32(radar)
    optical = _to_float32(optical)

    if radar.ndim != 3:
        raise ValueError(f"radar must be (C, H, W), got {radar.shape}")
    if optical.ndim == 2:
        optical = optical[np.newaxis, ...]
    if optical.ndim != 3:
        raise ValueError(f"optical must be (H, W) or (C, H, W), got {optical.shape}")
    if radar.shape[-2:] != optical.shape[-2:]:
        raise ValueError(
            "Spatial mismatch between radar and optical: "
            f"{radar.shape[-2:]} vs {optical.shape[-2:]}"
        )

    if normalize_radar:
        radar = normalize_dfsar(radar, db_scale=db_scale)

    stacked = np.concatenate([radar, optical], axis=0)
    return stacked.astype(np.float32, copy=False)


def preprocess_scene(
    dfsar_channels: Mapping[str, np.ndarray] | np.ndarray | Mapping[str, str | Path],
    ohrc_path: str | Path,
    dfsar_reference_path: str | Path,
    lee_window: int = DEFAULT_LEE_WINDOW,
    n_looks: float = DEFAULT_LEE_LOOKS,
) -> np.ndarray:
    """End-to-end preprocessing of one co-located DFSAR / OHRC scene.

    Args:
        dfsar_channels: In-memory polarimetric cube, mapping of arrays, or
            mapping of file paths for HH/HV/VH/VV.
        ohrc_path: Path to the OHRC raster.
        dfsar_reference_path: Path to a DFSAR raster that defines the grid.
        lee_window: Lee filter kernel size.
        n_looks: Equivalent number of looks.

    Returns:
        Stacked multi-channel array ``(4 + C_ohrc, H, W)``.
    """
    if isinstance(dfsar_channels, np.ndarray):
        radar = lee_filter_polarimetric(dfsar_channels, window_size=lee_window, n_looks=n_looks)
    else:
        loaded: dict[str, np.ndarray] = {}
        for name, value in dfsar_channels.items():
            if isinstance(value, (str, Path)):
                data, _, _ = _read_raster(value)
                loaded[name] = data[0] if data.ndim == 3 else data
            else:
                loaded[name] = np.asarray(value)
        radar = lee_filter_polarimetric(loaded, window_size=lee_window, n_looks=n_looks)

    optical = coregister_ohrc_to_dfsar(ohrc_path, dfsar_reference_path)
    return stack_radar_optical(radar, optical)


def crop_to_overlap(
    src_path: str | Path,
    ref_path: str | Path,
) -> Window:
    """Return the window of ``ref_path`` that overlaps ``src_path`` in WGS84.

    Useful when DFSAR and OHRC footprints only partially intersect.
    """
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        left, bottom, right, top = transform_bounds(src.crs, ref.crs, *src.bounds)
        return from_bounds(left, bottom, right, top, transform=ref.transform)

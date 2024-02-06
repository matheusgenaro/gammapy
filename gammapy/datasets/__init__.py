# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.registry import Registry
from .core import Dataset, Datasets
from .flux_points import FluxPointsDataset
from .io import OGIPDatasetReader, OGIPDatasetWriter
from .map import MapDataset, MapDatasetOnOff, MapDatasetBASiL, create_map_dataset_geoms
from .simulate import MapDatasetEventSampler, ObservationEventSampler
from .spectrum import SpectrumDataset, SpectrumDatasetOnOff, SpectrumDatasetOnOffBASiL

DATASET_REGISTRY = Registry(
    [
        MapDataset,
        MapDatasetOnOff,
        MapDatasetBASiL,
        SpectrumDataset,
        SpectrumDatasetOnOff,
        SpectrumDatasetOnOffBASiL,
        FluxPointsDataset,
    ]
)

"""Registry of dataset classes in Gammapy."""

__all__ = [
    "create_map_dataset_geoms",
    "Dataset",
    "DATASET_REGISTRY",
    "Datasets",
    "FluxPointsDataset",
    "MapDataset",
    "MapDatasetEventSampler",
    "MapDatasetOnOff",
    "MapDatasetBASiL",
    "ObservationEventSampler",
    "OGIPDatasetWriter",
    "OGIPDatasetReader",
    "SpectrumDataset",
    "SpectrumDatasetOnOff",
    "SpectrumDatasetOnOffBASiL",
]

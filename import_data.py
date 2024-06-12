import numpy as np
import xarray as xr
from glob import glob
from typing import List, Optional
import matplotlib.pyplot as plt


class ImportData:
    S2_SCALE_FACTOR = 1e-4
    S2_BANDS = [
        "aot",
        "blue",
        "coastal",
        "green",
        "nir",
        "nir08",
        "nir09",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "swir16",
        "swir22",
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.patches = self.load_patches()
        self.masked_patch = self.process_patches()

    def load_patches(self) -> List[xr.Dataset]:
        patches = []
        for patch_s3_key in glob(f"{self.filepath}/*"):
            patch = xr.open_zarr(patch_s3_key, consolidated=False)
            patches.append(patch)
        return patches

    def scale_sentinel_bands(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset[self.S2_BANDS] = (dataset[self.S2_BANDS] * self.S2_SCALE_FACTOR).astype(np.float32)
        return dataset

    def get_scl_mask(self, dataset: xr.Dataset, scl_classes: List[int]) -> np.array:
        return dataset["scl"].isin(scl_classes).values

    def apply_mask(self, dataset: xr.Dataset, mask: np.array, flatten_time_dim: bool) -> xr.Dataset:
        if flatten_time_dim:
            return dataset.where(mask).min(dim="time", skipna=True)
        else:
            return dataset.where(mask)

    def process_patches(self) -> xr.Dataset:
        for patch in self.patches:
            scaled_patch = self.scale_sentinel_bands(patch)
            scl_mask = self.get_scl_mask(scaled_patch, scl_classes=[4, 5])
            masked_patch = self.apply_mask(scaled_patch, scl_mask, flatten_time_dim=False)

            indexes = ["ndvi", "sr"]
            for index in indexes:
                masked_patch[index] = self.calculate_spectral_index(masked_patch, index)

        return masked_patch

    def arvi(self, dataset: xr.Dataset) -> xr.DataArray:
        return (dataset.nir - (2 * dataset.red - dataset.blue)) / (
                dataset.nir + (2 * dataset.red - dataset.blue)
        )

    def evi(self, dataset: xr.Dataset) -> xr.DataArray:
        return (
                (dataset.nir - dataset.red)
                / (dataset.nir + 6 * dataset.red - 7.5 * dataset.blue + 1)
        ) * 2.5

    def ndvi(self, dataset: xr.Dataset) -> xr.DataArray:
        return (dataset.nir - dataset.red) / (dataset.nir + dataset.red)

    def savi(self, dataset: xr.Dataset, soil_brightness_correction: Optional[float] = 0.5) -> xr.DataArray:
        return ((dataset.nir - dataset.red) * (1 + soil_brightness_correction)) / (
                dataset.nir + dataset.red + soil_brightness_correction
        )

    def dvi(self, dataset: xr.Dataset) -> xr.DataArray:
        return dataset.nir08 - dataset.red

    def sr(self, dataset: xr.Dataset) -> xr.DataArray:
        return dataset.nir08 / dataset.red

    def gci(self, dataset: xr.Dataset) -> xr.DataArray:
        return (dataset.nir08) / (dataset.green) - 1

    def msavi2(self, dataset: xr.Dataset) -> xr.DataArray:
        return (2 * dataset.nir08 + 1) - np.sqrt(
            np.square(2 * dataset.nir08 + 1) - 8 * (dataset.nir08 - dataset.red)
        ) / 2

    def ndwi(self, dataset: xr.Dataset) -> xr.DataArray:
        return (dataset.green - dataset.nir08) / (dataset.green + dataset.nir08)

    def ndmi(self, dataset: xr.Dataset) -> xr.DataArray:
        return (dataset.nir08 - dataset.swir16) / (dataset.nir08 + dataset.swir16)

    def calculate_spectral_index(self, dataset: xr.Dataset, index: str, **kwargs) -> xr.DataArray:
        if index == "arvi":
            return self.arvi(dataset)
        elif index == "evi":
            return self.evi(dataset)
        elif index == "ndvi":
            return self.ndvi(dataset)
        elif index == "savi":
            return self.savi(dataset, **kwargs)
        elif index == "dvi":
            return self.dvi(dataset)
        elif index == "sr":
            return self.sr(dataset)
        elif index == "gci":
            return self.gci(dataset)
        elif index == "msavi2":
            return self.msavi2(dataset)
        elif index == "ndwi":
            return self.ndwi(dataset)
        elif index == "ndmi":
            return self.ndmi(dataset)
        else:
            raise ValueError(f"Index {index} not supported")

    def sort_timesteps(self):

        """This will sort all the timesteps from one to n depending on which ones have the LEAST Nan values"""
        list_values = []

        for i in range(len(self.masked_patch['time'])):
            arr = self.masked_patch.variables['ndvi'][i].values
            Nan_mask = np.isnan(arr)
            non_nan_mask = ~Nan_mask
            count_value = np.sum(non_nan_mask)
            list_values.append([i, count_value])

        sorted_list = sorted(list_values, key=lambda x: x[1], reverse=True)

        print("The index in time which corresponds to the best data ")
        return sorted_list

    def time_for_index(self,index):

        """THis will literally print the time for that particular timestep """

        print(self.masked_patch['time'][index])

    def plot_NDVI(self, index):
        """"This will plot the NDVI data"""

        single_time_patch = self.masked_patch.isel(time=index)
        NDVI_arr = single_time_patch.variables['ndvi']
        ARR = NDVI_arr.values

        xvalues = single_time_patch.coords['x']
        yvalues = single_time_patch.coords['y']

        plt.imshow(ARR, extent=(xvalues.min(), xvalues.max(), yvalues.min(), yvalues.max()), origin='lower')
        plt.colorbar()
        plt.xlabel('Lat')
        plt.ylabel('Lon')
        plt.show()

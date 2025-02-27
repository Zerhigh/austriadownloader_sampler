import os.path

import geopandas as gpd
import numpy as np
import rasterio
import pandas as pd
import shapely
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point


class Config:
    def __init__(self, pixel_size=1.6, shape=(4, 1024, 1024)):
        self.pixel_size = pixel_size
        self.shape = shape


class Sampler:
    def __init__(self, sample_path, outpath_cropped_corine, corine_path, config, verbose=False):
        """
        Initialize the RasterReader with a given file path.
        """
        self.sample_path = sample_path
        self.corine_path = corine_path
        self.outpath_cropped_corine = outpath_cropped_corine
        self.config = config
        self.verbose = verbose

        self.lambert = 'EPSG:31287'
        self.corine = None
        self.corine_landcover = {111: "Continuous urban fabric",
                                 112: "Discontinuous urban fabric",
                                 121: "Industrial or commercial units",
                                 122: "Road and rail networks and associated land",
                                 123: "Port areas",
                                 124: "Airports",
                                 131: "Mineral extraction sites",
                                 132: "Dump sites",
                                 133: "Construction sites",
                                 141: "Green urban areas",
                                 142: "Sport and leisure facilities",
                                 211: "Non-irrigated arable land",
                                 212: "Permanently irrigated land",
                                 213: "Rice fields",
                                 221: "Vineyards",
                                 222: "Fruit trees and berry plantations",
                                 223: "Olive groves",
                                 231: "Pastures",
                                 241: "Annual crops associated with permanent crops",
                                 242: "Complex cultivation patterns",
                                 243: "Land principally occupied by agriculture, with significant areas of natural vegetation",
                                 244: "Agro-forestry areas",
                                 311: "Broad-leaved forest",
                                 312: "Coniferous forest",
                                 313: "Mixed forest",
                                 321: "Natural grasslands",
                                 322: "Moors and heathland",
                                 323: "Sclerophyllous vegetation",
                                 324: "Transitional woodland-shrub",
                                 331: "Beaches, dunes, sands",
                                 332: "Bare rocks",
                                 333: "Sparsely vegetated areas",
                                 334: "Burnt areas",
                                 335: "Glaciers and perpetual snow",
                                 411: "Inland marshes",
                                 412: "Peat bogs",
                                 421: "Salt marshes",
                                 422: "Salines",
                                 423: "Intertidal flats",
                                 511: "Water courses",
                                 512: "Water bodies",
                                 521: "Coastal lagoons",
                                 522: "Estuaries",
                                 523: "Sea and ocean"}

    @property
    def border_austria(self):
        fp_austria = 'data/matched_metadata.gpkg'
        fp = gpd.read_file(fp_austria)
        border = shapely.union_all(fp.geometry)
        return gpd.GeoDataFrame(geometry=[border], crs=fp.crs)

    def transform_corine(self, src, outpath):
        try:
            # Compute new transform and dimensions
            transform, width, height = calculate_default_transform(
                src.crs, self.lambert, src.width, src.height, *src.bounds
            )

            # Define metadata for the output raster
            new_meta = src.meta.copy()
            new_meta.update({
                "crs": self.lambert,
                "transform": transform,
                "width": width,
                "height": height
            })

            # Open output raster and reproject
            with rasterio.open(outpath, "w", **new_meta) as dst:
                reproject(
                    source=src.read(1),  # Read single band
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=self.lambert,
                    resampling=Resampling.nearest  # Change to bilinear/cubic if needed
                )
            return outpath
        except Exception as e:
            print(f"Error transforming raster to {self.lambert}: {e}")

    def crop(self, src, gdf):
        try:
            # Transform the geometry to match the raster's CRS - not necessary as i transform the raster
            gdf = gdf.to_crs(src.crs)

            # Clip the raster using the geometry
            image, out_transform = mask(src, [gdf.geometry[0]], crop=True)

            # Copy metadata and update
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": image.shape[1],
                "width": image.shape[2],
                "transform": out_transform
            })
            return image, meta
        except Exception as e:
            print(f"Error clipping raster: {e}")

    def process_corine(self):
        # croP first, then trasnform
        cropped_corine = 'data/corine/cropped.tif'

        # Open the raster file
        with rasterio.open(self.corine_path) as src:
            out_image, out_meta = self.crop(src=src, gdf=self.border_austria)

            # Save the clipped raster
            with rasterio.open(cropped_corine, "w", **out_meta) as dest:
                dest.write(out_image)

        # crop with austrian bordders extent
        with rasterio.open(cropped_corine) as src:
            # save as file
            self.transform_corine(src=src, outpath=self.outpath_cropped_corine)

        return

    def load_corine(self):
        """
        Open the raster file using rasterio.
        """
        if os.path.exists(self.outpath_cropped_corine):
            if self.verbose:
                print(f'loading corine from: {self.outpath_cropped_corine}')
            self.corine = rasterio.open(self.outpath_cropped_corine)
        else:
            if self.verbose:
                print('creating corine cropped image')
            self.process_corine()
            self.corine = rasterio.open(self.outpath_cropped_corine)

    def even_sample(self, gdf, config):
        """Generate an evenly spaced grid of centroids over a given geometry.

        Args:
            geometry (shapely.geometry.Polygon): Target polygon geometry.
            cell_size (float): Size of each grid cell.

        Returns:
            geopandas.GeoDataFrame: Grid of centroid points within the geometry.
        """
        # Get bounding box
        # iterate over all provided aois
        cell_size = config.pixel_size * config.shape[1]
        point_list = []

        minx, miny, maxx, maxy = gdf.iloc[0].geometry.bounds

        # Create a grid of points
        x_coords = np.arange(minx, maxx, cell_size) + (cell_size / 2)
        y_coords = np.arange(miny, maxy, cell_size) + (cell_size / 2)

        return [Point(x, y) for x in x_coords for y in y_coords]

    def filter_samples(self, gdf):
        # filter noData values
        points_corine_filtered = gdf.drop(gdf[gdf['value'] == self.corine.nodata].index)
        if len(points_corine_filtered) != len(gdf) and self.verbose:
            print(f'Dropped {len(gdf) - len(points_corine_filtered)} entries due to NoData values.')

        # assign text landcover
        points_corine_filtered['clc_str'] = points_corine_filtered.apply(
            lambda row: self.corine_landcover[row['value']], axis=1)

        # fulfill stratification

        return points_corine_filtered

    def generate_sample(self, num_points, aoi=None, sample_method='random', to_sample_crs='EPSG:4326', rng=1414):
        if self.corine is None:
            print('Corine Dataset not loaded... loading now')
            self.load_corine()
        # if aoi is None:
        #     aoi = self.border_austria
        #     if sample_method == 'random':
        #         sample = aoi.sample_points(size=num_points, rng=rng)
        #     else:
        #         sample = self.even_sample(geometry=aoi, config=self.config)
        #     points = gpd.GeoDataFrame(geometry=[p for p in sample[0].geoms], crs=aoi.crs)
        # else:
        #     if sample_method == 'random':
        #         sample = aoi.sample_points(size=num_points, rng=rng)
        #     else:
        #         sample = self.even_sample(geometry=aoi, config=self.config)
        #     # switched coords from wgs84 to european system
        #     points = gpd.GeoDataFrame(geometry=[shapely.Point(p.y, p.x) for p in sample[0].geoms], crs=aoi.crs)

        if aoi is None:
            aoi = self.border_austria
        if sample_method == 'random':
            sample = aoi.sample_points(size=num_points, rng=rng)
            points = gpd.GeoDataFrame(geometry=[p for p in sample[0].geoms], crs=aoi.crs)
        else:
            sample = self.even_sample(gdf=aoi, config=self.config)
            points = gpd.GeoDataFrame(geometry=sample, crs=aoi.crs)

        # convert series to corine crs to query landcover
        assert points.crs == self.lambert and self.corine.crs == self.lambert

        coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
        points['value'] = [x[0] for x in self.corine.sample(coord_list)]

        points_corine_filtered = self.filter_samples(gdf=points)
        if points_corine_filtered.crs != to_sample_crs:
            points_corine_filtered.to_crs(crs=to_sample_crs, inplace=True)

        if self.verbose:
            print(f'Generated {len(points_corine_filtered)} sample points in given AOI.')
        points_corine_filtered.to_file(f'output/sample_{sample_method}.gpkg', driver='GPKG')

        return


config = Config(pixel_size=1.6, shape=(4, 1024, 1024))
sampler = Sampler(sample_path='',
                  corine_path="data/corine/CLC2018ACC_V2018_20.tif",
                  outpath_cropped_corine="data/corine/corine_transformed.tif",
                  verbose=True,
                  config=config)

sampler.load_corine()


# aoi_bbox = shapely.box(47.9290294921047, 16.073601498323555, 47.95607947801313, 16.133786632464748)
# aoi = gpd.GeoDataFrame(geometry=[aoi_bbox], crs='EPSG:4236')

aoi_bbox = shapely.box(616294, 472362.1, 631819, 490055)
aoi = gpd.GeoDataFrame(geometry=[aoi_bbox], crs='EPSG:31287')

sampler.generate_sample(num_points=50, aoi=aoi, sample_method='even')
sampler.generate_sample(num_points=50, aoi=aoi, sample_method='random')

pass
# t.crop_full_corine()

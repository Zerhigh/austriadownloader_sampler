import os.path
import time

import geopandas as gpd
import numpy as np
import rasterio
import pandas as pd
import shapely
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point
from tqdm import tqdm

from imageconfig import ImageConfig
from stratification import Stratification


class Sampler:
    def __init__(self, sample_path, outpath_cropped_corine, corine_path, aggregate, rng, config, stratification, verbose=False):
        """
        Initialize the RasterReader with a given file path.
        """
        self.sample_path = sample_path
        self.corine_path = corine_path
        self.outpath_cropped_corine = outpath_cropped_corine
        self.aggregate = aggregate
        self.rng = rng
        self.config = config
        self.stratification = stratification
        self.verbose = verbose

        self.lambert = 'EPSG:31287'
        self.wgs = 'EPSG:4326'
        self.corine = None
        self.clc_agg_int = {111: 1,  # "urban"
                        112: 1,  # "urban"
                        121: 1,  # "urban"
                        122: 1,  # "urban"
                        123: 1,  # "urban"
                        124: 1,  # "urban"
                        131: 1,  # "urban"
                        132: 1,  # "urban"
                        133: 1,  # "urban"
                        141: 1,  # "urban"
                        142: 1,  # "urban"
                        511: 2,  # "water"
                        512: 2,  # "water"
                        521: 2,  # "water"
                        522: 2,  # "water"
                        523: 2,  # "water"
                        211: 3,  # "agricultural"
                        212: 3,  # "agricultural"
                        213: 3,  # "agricultural"
                        221: 3,  # "agricultural"
                        222: 3,  # "agricultural"
                        223: 3,  # "agricultural"
                        231: 3,  # "agricultural"
                        241: 3,  # "agricultural"
                        242: 3,  # "agricultural"
                        243: 3,  # "agricultural"
                        244: 3,  # "agricultural"
                        311: 4,  # "forests"
                        312: 4,  # "forests"
                        313: 4,  # "forests"
                        321: 5,  # "other"
                        322: 5,  # "other"
                        323: 5,  # "other"
                        324: 5,  # "other"
                        331: 5,  # "other"
                        332: 6,  # "bare_rock"
                        333: 5,  # "other"
                        334: 5,  # "other"
                        335: 7,  # "glacier"
                        411: 5,  # "other"
                        412: 5,  # "other"
                        421: 5,  # "other"
                        422: 5,  # "other"
                        423: 5  # "other"
                    }
        self.clc_keys = [1, 2, 3, 4, 5, 6, 7]

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

    def crop(self, src, gdf, aggregate=False):
        try:
            # Transform the geometry to match the raster's CRS - not necessary as i transform the raster
            gdf = gdf.to_crs(src.crs)

            # Clip the raster using the geometry
            image, out_transform = mask(src, [gdf.geometry[0]], crop=True)

            # convert the large number of corine classes to a reduced aggregated list of classes
            if aggregate:
                if self.verbose:
                    print('Aggregating corine classes..')
                unique_vals = np.unique(image)
                for val in unique_vals:
                    # filter out no data value
                    if val >= 0:
                        image[image == val] = self.clc_agg_int[val]

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
            out_image, out_meta = self.crop(src=src, gdf=self.border_austria, aggregate=self.aggregate)

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

    def filter_window(self, pt, window_size=None):
        if window_size is None:
            window_size = 4
        # Convert geographical point to pixel coordinates
        col, row = ~self.corine.transform * (pt.x, pt.y)

        window = ((row - window_size, row + window_size + 1),
                  (col - window_size, col + window_size + 1))

        # Read the surrounding pixels
        surrounding_pixels = self.corine.read(1, window=window)
        vals, counts = np.unique(surrounding_pixels, return_counts=True)

        return surrounding_pixels, vals, counts

    def filter_samples(self, gdf, window_size=None):
        # filter noData values
        points_corine_filtered = gdf.drop(gdf[gdf['corine'] == self.corine.nodata].index)
        if self.verbose and len(points_corine_filtered) != len(gdf):
            print(f'Dropped {len(gdf) - len(points_corine_filtered)} entries due to NoData values.')

        if self.verbose:
            print('Filtering points..')

        # iterate over each sampled point and check if subclasses are included in this extracted window
        for i, row in tqdm(points_corine_filtered.iterrows()):
            _, vals, counts = self.filter_window(pt=row.geometry, window_size=window_size)
            csum = np.sum(counts)
            if self.corine.nodata in vals:
                pass
            # calculate the percentage of each class in this window, if it exceeds the treshhold include it a attribute
            dist = {v: c/csum for v, c in zip(vals, counts) if v != self.corine.nodata}

            # go over distributions if its over the threshold change its flag to true
            for k, v in dist.items():
                if v >= self.stratification.window_threshold[k]:
                    points_corine_filtered.at[i, f'{k}'] = 1
                    # save space in dataframe
                    points_corine_filtered.at[i, f'{k}_dist'] = int(round(v, 2)*100)

        return points_corine_filtered

    def select_samples(self, df):
        # select smaple size
        num_samples = 0
        if self.stratification.num_samples == 'max':
            num_samples = len(df)
        else:
            num_samples = self.stratification.num_samples

        df['class'] = None
        df = df.sample(frac=1, random_state=self.rng)

        meta_df = pd.DataFrame({
            'target_percentage': self.stratification.clc_filtered.values()
        }, index=self.stratification.clc_filtered.keys())
        meta_df['class'] = meta_df.index.astype(str)

        filtered_dfs = {}

        # Iterate over each class and its percentage in sorted_strat
        for clc, row in meta_df.sort_values(by='target_percentage', ascending=False).iterrows():
            # Filter the dataframe based on the class column
            df_filtered = df[df[f'{clc}_dist'] > 0]

            # Calculate the percentage
            percentage = len(df_filtered) / len(df)
            # Calculate the percentage difference
            percentage_diff = percentage - row.target_percentage
            filtered_len = len(df_filtered)

            # calculate the minimum sample size
            if percentage_diff < 0:
                if filtered_len <= 0:
                    meta_df.loc[clc, 'min_sample'] = None
                    print(f'Warning: Class {clc} is not represented at all, will not consider it and continue with stratification.')
                else:
                    meta_df.loc[clc, 'min_sample'] = int(filtered_len / row.target_percentage)
            # I have enough samples, just gibe me all of them
            else:
                meta_df.loc[clc, 'min_sample'] = num_samples

            meta_df.loc[clc, 'percentage'] = percentage
            meta_df.loc[clc, 'percentage_diff'] = percentage_diff
            meta_df.loc[clc, 'length'] = filtered_len
            # svae Seperated dataframe in dict with common key
            filtered_dfs[clc] = df_filtered

        min_sample = min(meta_df.min_sample)

        for clc, row in meta_df.iterrows():
            # calculate number of samples of adapted maximal number
            if pd.notna(row.min_sample):
                meta_df.loc[clc, 'assigned_samples'] = row.target_percentage * min_sample
            else:
                meta_df.loc[clc, 'assigned_samples'] = 0

        class_dist_cols = [col for col in df.columns if col.endswith('_dist')]
        class_labels = [col.split('_dist')[0] for col in class_dist_cols]

        # Add a column for the best class prediction
        df['best_class'] = df[class_dist_cols].idxmax(axis=1).str.replace('_dist', '')

        quota = {str(k): v.assigned_samples for k, v in meta_df.iterrows()}
        class_counts = {key: 0 for key in quota}

        # Assign classes while respecting quotas
        for idx, row in df.iterrows():
            best_class = row['best_class']

            # Only assign if quota isn't full
            if best_class in quota and class_counts[best_class] < quota[best_class]:
                df.at[idx, 'class'] = best_class
                class_counts[best_class] += 1

        # Assign remaining unassigned rows to any class with available slots
        for idx, row in df.iterrows():
            if pd.isna(row['class']):
                # Check if row is still unassigned: sorte on distribution
                sorted_rows = row[class_dist_cols].sort_values(ascending=False)
                for index, value in sorted_rows.items():
                    if value > 0:
                        cl = index.replace('_dist', '')
                        if class_counts[cl] < quota[cl]:
                            df.at[idx, 'class'] = cl
                            class_counts[cl] += 1
                            break  # Stop once assigned

        df = df.dropna(subset=['class'])
        class_counts = df['class'].value_counts()
        meta_df['class_counts'] = meta_df['class'].map(class_counts)
        return df, meta_df

    def generate_download_file(self, gdf):
        if gdf.crs != self.wgs:
            gdf = gdf.to_crs(self.wgs)
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x

        columns_to_keep = ['lat', 'lon', 'corine', 'class']  # Replace with your column names

        df = gdf.loc[:, columns_to_keep]
        df.reset_index(drop=True, inplace=True)
        df['id'] = df.index
        return df

    def generate_sample(self, num_points, aoi=None, sample_method='random', to_sample_crs='EPSG:4326'):
        print('Generating samples..')
        if self.corine is None:
            print('Corine Dataset not loaded... loading now')
            self.load_corine()

        if aoi is None:
            aoi = self.border_austria
        if sample_method == 'random':
            sample = aoi.sample_points(size=num_points, rng=self.rng)
            points = gpd.GeoDataFrame(geometry=[p for p in sample[0].geoms], crs=aoi.crs)
        else:
            sample = self.even_sample(gdf=aoi, config=self.config)
            points = gpd.GeoDataFrame(geometry=sample, crs=aoi.crs)

        # convert series to corine crs to query landcover
        assert points.crs == self.lambert and self.corine.crs == self.lambert

        coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
        points['corine'] = [x[0] for x in self.corine.sample(coord_list)]

        # add class-tagging columns to df
        for col in self.clc_keys:
            points[f'{col}'] = 0
            points[f'{col}_dist'] = 0

        # assign attributes to sampeld points basd on window operation
        points_corine_filtered = self.filter_samples(gdf=points, window_size=self.config.window_size) #self.config.window_size

        # select saple of points from datarame
        selected, meta = self.select_samples(df=points_corine_filtered)
        print(meta)

        if self.verbose:
            print(f'Generated {len(selected)} sample points in given AOI.')
        selected.to_file(f'output/{self.sample_path}.gpkg', driver='GPKG')
        meta.to_csv(f'output/{self.sample_path}_metadata.csv')

        out_pd = self.generate_download_file(selected)
        out_pd.to_csv(f'output/sample_{sample_method}_download.csv', index=False)

        return


# 1: urban
# 2: water
# 3: agricultural
# 4: forest
# 5: other
# 6: bare_rock
# 7: glacier
config = ImageConfig(pixel_size=2.5, shape=(4, 512, 512))
strat = Stratification(clc_distribution={1: 0.45, 2: 0.01, 3: 0.25, 4: 0.25, 5: 0.02, 6: 0.01, 7: 0.01},
                       window_threshold={1: 0.05, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1},
                       num_samples='max')

sampler = Sampler(sample_path='verification',
                  corine_path="data/corine/CLC2018ACC_V2018_20.tif",
                  outpath_cropped_corine="data/corine/corine_transformed_aggregated.tif",
                  aggregate=True,
                  verbose=True,
                  config=config,
                  rng=1441,
                  stratification=strat)

sampler.load_corine()


# greatEr vienna
aoi_bbox_ = shapely.box(567222, 445325, 671295, 545245)
# Vienna
aoi_bbox = shapely.box(616294, 472362.1, 638000, 491000)
vienna = gpd.GeoDataFrame(geometry=[aoi_bbox], crs='EPSG:31287')
vienna_greater = gpd.GeoDataFrame(geometry=[aoi_bbox_], crs='EPSG:31287')

t1 = time.time()
sampler.generate_sample(num_points=53000, aoi=None, sample_method='even')
print('sampling for whole Austria [s]: ', round(time.time() - t1, 2))

pass

clc_agg = {111: "urban",
                        112: "urban",
                        121: "urban",
                        122: "urban",
                        123: "urban",
                        124: "urban",
                        131: "urban",
                        132: "urban",
                        133: "urban",
                        141: "urban",
                        142: "urban",
                        511: "water",
                        512: "water",
                        521: "water",
                        522: "water",
                        523: "water",
                        211: "agricultural",
                        212: "agricultural",
                        213: "agricultural",
                        221: "agricultural",
                        222: "agricultural",
                        223: "agricultural",
                        231: "agricultural",
                        241: "agricultural",
                        242: "agricultural",
                        243: "agricultural",
                        244: "agricultural",
                        311: "forest",
                        312: "forest",
                        313: "forest",
                        321: "other",
                        322: "other",
                        323: "other",
                        324: "other",
                        331: "other",
                        332: "bare_rock",
                        333: "other",
                        334: "other",
                        335: "glacier",
                        411: "other",
                        412: "other",
                        421: "other",
                        422: "other",
                        423: "other"}
corine_landcover = {111: "Continuous urban fabric",
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
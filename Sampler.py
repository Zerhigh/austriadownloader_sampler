import os.path

import rasterio
import numpy as np
import geopandas as gpd
import shapely
from rasterio.mask import mask


class Sampler:
    def __init__(self, sample_path, outpath_cropped_corine, corine_path, verbose=False):
        """
        Initialize the RasterReader with a given file path.
        """
        self.sample_path = sample_path
        self.corine_path = corine_path
        self.outpath_cropped_corine = outpath_cropped_corine
        self.verbose = verbose
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
        fp_austria = r'C:\Users\PC\Coding\GeoQuery\austriadownloader\austria_data\matched_metadata.gpkg'
        fp = gpd.read_file(fp_austria)
        border = shapely.union_all(fp.geometry)
        return gpd.GeoDataFrame(geometry=[border], crs=fp.crs)

    def load_corine(self):
        """
        Open the raster file using rasterio.
        """
        try:
            if os.path.exists(self.outpath_cropped_corine):
                if self.verbose:
                    print(f'loading corine from: {self.outpath_cropped_corine}')
                self.corine = rasterio.open(self.outpath_cropped_corine)
            else:
                if self.verbose:
                    print('creating corine cropped image')
                self.crop_full_corine()
                self.corine = rasterio.open(self.outpath_cropped_corine)
        except Exception as e:
            print(f"Error opening corine: {e}")

    def crop(self, src, gdf):
        try:
            # Transform the geometry to match the raster's CRS
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

    def crop_full_corine(self):
        if os.path.exists(self.outpath_cropped_corine):
            print('cropped file already exists')
        else:
            try:
                # Open the raster file
                with rasterio.open(self.corine_path) as src:
                    out_image, out_meta = self.crop(src=src, gdf=self.border_austria)

                    # Save the clipped raster
                    with rasterio.open(self.outpath_cropped_corine, "w", **out_meta) as dest:
                        dest.write(out_image)

                if self.verbose:
                    print(f"Clipped raster saved to: {self.outpath_cropped_corine}")

            except Exception as e:
                print(f"Error clipping full raster: {e}")

        return

    def get_corine_value(self, x, y):
        """
        Get the pixel value at the given (x, y) coordinate.
        :param x: X coordinate in pixel space
        :param y: Y coordinate in pixel space
        :return: Pixel value(s) at the given location
        """
        if self.corine:
            return self.corine.read(window=((y, y + 1), (x, x + 1))).squeeze()
        else:
            raise ValueError("Dataset is not open.")

    def generate_sample(self, num_points, aoi=None, to_sample_crs='EPSG:4326', rng=1414):
        if self.corine is None:
            print('Corine Dataset not loaded... loading now')
            self.load_corine()
        if aoi is None:
            aoi = self.border_austria
            sample = aoi.sample_points(size=num_points, rng=rng)
            points = gpd.GeoDataFrame(geometry=[p for p in sample[0].geoms], crs=aoi.crs)
        else:
            sample = aoi.sample_points(size=num_points, rng=rng)
            # switched coords from wgs84 to european system
            points = gpd.GeoDataFrame(geometry=[shapely.Point(p.y, p.x) for p in sample[0].geoms], crs=aoi.crs)

        # convert series to corine crs to query landcover
        points_corine = points.to_crs(crs=self.corine.crs)
        coord_list = [(x, y) for x, y in zip(points_corine["geometry"].x, points_corine["geometry"].y)]
        points_corine['value'] = [x[0] for x in self.corine.sample(coord_list)]

        # filter noData values
        points_corine_filtered = points_corine.drop(points_corine[points_corine['value'] == self.corine.nodata].index)
        if len(points_corine_filtered) != len(points_corine) and self.verbose:
            print(f'Dropped {len(points_corine) - len(points_corine_filtered)} entries due to NoData values.')

        # assign text landcover
        points_corine_filtered['clc_str'] = points_corine_filtered.apply(lambda row: self.corine_landcover[row['value']], axis=1)

        if points_corine_filtered.crs != to_sample_crs:
            points_corine_filtered.to_crs(crs=to_sample_crs, inplace=True)

        if self.verbose:
            print(f'Generated {len(points_corine_filtered)} sample points in given AOI.')
        points_corine_filtered.to_file('sample.gpkg', driver='GPKG')

        return


sampler = Sampler(sample_path='',
                  corine_path=r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\corine\eea_r_3035_100_m_clc-2018-acc_p_2017-2018_v01_r00\eea_r_3035_100_m_clc-2018-acc_p_2017-2018_v01_r00\CLC2018ACC_V2018_20.tif",
                  outpath_cropped_corine=r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\corine\corine_croppped.tif",
                  verbose=True)

sampler.load_corine()

aoi_bbox = shapely.box(47.9290294921047, 16.073601498323555, 47.95607947801313, 16.133786632464748)
aoi = gpd.GeoDataFrame(geometry=[aoi_bbox], crs='EPSG:4326')

sampler.generate_sample(num_points=40000, aoi=None)

pass
# t.crop_full_corine()

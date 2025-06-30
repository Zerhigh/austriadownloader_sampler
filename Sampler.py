import time
import shapely
import geopandas as gpd
import numpy as np

from datetime import datetime
from shapely.geometry import Point
from pathlib import Path

from imageconfig import ImageConfig


class Sampler:
    def __init__(self, rng, config, verbose=False):
        """
        Initialize the RasterReader with a given file path.
        """
        self.rng = rng
        self.config = config
        self.verbose = verbose

        self.lambert = 'EPSG:31287'
        self.wgs = 'EPSG:4326'

    @property
    def border_austria_matched_metadata(self):
        fp_austria = 'data/matched_metadata.gpkg'
        fp = gpd.read_file(fp_austria)
        border = shapely.union_all(fp.geometry)
        return gpd.GeoDataFrame(geometry=[border], crs=fp.crs)

    @property
    def border_austria(self):
        fp_austria = Path('data/oesterreich_border/oesterreich.shp')
        fp = gpd.read_file(fp_austria)
        return fp

    def convert_date(self, date_str: str) -> datetime:
        """
        Convert a German date string to a datetime object.

        :param date_str: The input date string in 'dd-Mon-yy' format with German month names.
        :return: A datetime object.
        :raises ValueError: If the date string is incorrectly formatted.
        """
        try:
            day, month, year = date_str.split("-")
            german_to_english_months = {
                "Jan": "Jan", "Feb": "Feb", "Mär": "Mar", "Apr": "Apr", "Mai": "May",
                "Jun": "Jun", "Jul": "Jul", "Aug": "Aug", "Sep": "Sep", "Okt": "Oct",
                "Nov": "Nov", "Dez": "Dec"
            }

            month = german_to_english_months.get(month, month)
            return datetime.strptime(f"{day}-{month}-{year}", "%d-%b-%y")
        except Exception as e:
            raise ValueError(f"Error parsing date {date_str}: {e}")

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

        geom = gdf.iloc[0].geometry
        minx, miny, maxx, maxy = geom.bounds

        point_list = []

        # Create a grid of points
        x_coords = np.arange(minx, maxx, cell_size) + (cell_size / 2)
        y_coords = np.arange(miny, maxy, cell_size) + (cell_size / 2)

        points = [Point(x, y) for x in x_coords for y in y_coords if geom.intersects(Point(x, y))]
        if self.verbose:
            print(f'Filtered {len(x_coords)*len(y_coords) - len(points)} points with intersection of boundary')
        return points

    def generate_download_file(self, gdf, keep_geom=True):
        if gdf.crs != self.wgs:
            gdf = gdf.to_crs(self.wgs)
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x

        columns_to_keep = ['geometry', 'lat', 'lon', 'query_day', 'ARCHIVNR', 'Operat', 'vector_url', 'RGB_raster', 'NIR_raster', 'prevTime', 'Date', 'beginLifeS', 'endLifeSpa']  # Replace with your column names
        if not keep_geom:
            columns_to_keep.remove('geometry')

        df = gdf.loc[:, columns_to_keep]
        df.reset_index(drop=True, inplace=True)
        df['id'] = df.index.astype(str)
        return df

    def add_date(self, gdf):
        fp_austria = 'data/matched_metadata.gpkg'
        orthophoto_meta = gpd.read_file(fp_austria)
        orthophoto_meta['beginLifeS'] = orthophoto_meta['beginLifeS'].apply(self.convert_date)
        orthophoto_meta['endLifeSpa'] = orthophoto_meta['endLifeSpa'].apply(self.convert_date)
        orthophoto_meta['query_day'] = orthophoto_meta['beginLifeS'] + (orthophoto_meta['endLifeSpa'] - orthophoto_meta['beginLifeS']) / 2

        intersection = gpd.sjoin(gdf, orthophoto_meta, how='left', predicate='intersects')

        return intersection

    def generate_sample(self, sample_path, num_points, aoi=None, sample_method='random'):
        print('Generating samples..')
        if aoi is None:
            aoi = self.border_austria
        if sample_method == 'random':
            sample = aoi.sample_points(size=num_points, rng=self.rng)
            points = gpd.GeoDataFrame(geometry=[p for p in sample[0].geoms], crs=aoi.crs)
        else:
            sample = self.even_sample(gdf=aoi, config=self.config)
            points = gpd.GeoDataFrame(geometry=sample, crs=aoi.crs)

        selected_wmeta = self.add_date(points)

        if self.verbose:
            print(f'Generated {len(selected_wmeta)} sample points in given AOI.')

        selected_wmeta.to_file(f'output/{sample_path}.gpkg', driver='GPKG')

        # used to download gee Sentinel2 Data
        selected_reduced = self.generate_download_file(selected_wmeta)
        selected_reduced.to_file(f'output/{sample_path}_s2download.gpkg', driver='GPKG')

        out_pd = self.generate_download_file(selected_wmeta, keep_geom=False)
        out_pd.to_csv(f'output/{sample_path}.csv', index=False)
        return


# greatEr vienna
aoi_bbox_ = shapely.box(567222, 445325, 671295, 545245)
vienna_greater = gpd.GeoDataFrame(geometry=[aoi_bbox_], crs='EPSG:31287')

# Vienna
aoi_bbox = shapely.box(616294, 472362.1, 638000, 491000)
vienna = gpd.GeoDataFrame(geometry=[aoi_bbox], crs='EPSG:31287')

# demo
demo_ = gpd.GeoDataFrame(geometry=[shapely.box(592406, 420561, 601846, 428517)], crs='EPSG:31287')

sampler = Sampler(verbose=True,
                  config=ImageConfig(pixel_size=2.5, shape=(4, 512, 512)),
                  rng=1441)

t1 = time.time()
sampler.generate_sample(sample_path='demo_full', num_points=53000, aoi=None, sample_method='even')
print('sampling for whole Austria [s]: ', round(time.time() - t1, 2))

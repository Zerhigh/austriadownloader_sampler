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
        points = []

        for i, row in gdf.iterrows():
            geom = row['geometry']

            minx, miny, maxx, maxy = geom.bounds

            # Create a grid of points
            x_coords = np.arange(minx, maxx, cell_size) + (cell_size / 2)
            y_coords = np.arange(miny, maxy, cell_size) + (cell_size / 2)
            pass

            for x in x_coords:
                for y in y_coords:
                    if geom.intersects(Point(x, y)):
                        points.append(Point(x, y))

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

    def generate_sample(self, output_path, num_points, aoi=None, sample_method='random'):
        if self.verbose:
            print('Generating samples..')
            print(f'Sampling method: {sample_method}')

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

        output_path_full = f'output/{output_path}'
        if not Path('output').exists():
            Path('output').mkdir(exist_ok=True)
        selected_wmeta.to_file(f'{output_path_full}.gpkg', driver='GPKG')

        # used to download gee Sentinel2 Data
        selected_reduced = self.generate_download_file(selected_wmeta)
        selected_reduced.to_file(f'{output_path_full}_s2download.gpkg', driver='GPKG')

        out_pd = self.generate_download_file(selected_wmeta, keep_geom=False)
        out_pd.to_csv(f'{output_path_full}.csv', index=False)
        return


if __name__ == "__main__":
    # Greater vienna area
    vienna_greater = gpd.GeoDataFrame(geometry=[shapely.box(567222, 445325, 671295, 545245)],
                                      crs='EPSG:31287')

    # Vienna
    vienna = gpd.GeoDataFrame(geometry=[shapely.box(616294, 472362.1, 638000, 491000)],
                              crs='EPSG:31287')

    # Demo Area
    demo_ = gpd.GeoDataFrame(geometry=[shapely.box(592406, 420561, 601846, 428517)],
                             crs='EPSG:31287')

    # lakes requests
    lakes = gpd.GeoDataFrame(geometry=[shapely.box(643286, 422503, 663798, 454725),
                                       shapely.box(440517, 295426, 475534, 313339),
                                       shapely.box(392604, 430935, 423143, 453110)
                                       ],
                             crs='EPSG:31287')

    # Define sampler class
    sampler = Sampler(verbose=True,
                      config=ImageConfig(pixel_size=20, shape=(3, 512, 512)),
                      rng=1441)

    t1 = time.time()
    sampler.generate_sample(output_path='demo',
                            num_points=53000,
                            aoi=None,
                            sample_method='even')
    print('sampling for whole Austria [s]: ', round(time.time() - t1, 2))


# austriadownlaoder_sampler

This Python project generates geospatial sample points (random or evenly spaced) within Austria with a user-defined area of interest (AOI). This sampling process is designed to streamline the further downloading process of the [austriadownloader](https://github.com/Zerhigh/austriadownloader) package.

Modify the `__main__` block in `Sampler.py` to define your AOI and sampling parameters. Also select your `ImageConfig` parameters `shape` and `pizel_size` to reduce sampling overlap or undersampling.

### Output

* `output/demo_full.csv`: CSV version for tabular processing (as input for the [austriadownloader](https://github.com/Zerhigh/austriadownloader) package)
* `output/demo_full.gpkg`: Sample points with geospatial extension
* `output/demo_full_s2download.gpkg`: Sample points with additional metadata for Sentinel-2 tile download


### Notes

Input metadata is already available at:

* `data/matched_metadata.gpkg`: Links for downloading data
* `data/oesterreich_border/oesterreich.shp`: Austria boundary shape 

### Requirements

* Python 3.10+
* `geopandas`
* `shapely`
* `numpy`
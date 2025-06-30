
# austriadownlaoder_sampler

This Python project generates geospatial sample points (random or evenly spaced) within Austria with a user-defined area of interest (AOI). This sampling process is designed to streamline for the further downlaoding process of the [austriadownloader](https://github.com/Zerhigh/austriadownloader) package.

### Requirements

* Python 3.10+
* `geopandas`
* `shapely`
* `numpy`

Modify the `__main__` block to define your AOI and sampling parameters.

### Output

* `output/demo_full.gpkg`: Sample points with geospatial extension
* `output/demo_full_s2download.gpkg`: Sample points with additional metadata for Sentinel-2 tile download
* `output/demo_full.csv`: CSV version for tabular processing

### Notes

Input data is already available at

* `data/matched_metadata.gpkg`: Links for downloading data
* `data/oesterreich_border/oesterreich.shp`: Austria boundary shape 

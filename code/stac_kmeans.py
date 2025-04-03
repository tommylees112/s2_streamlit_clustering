import os
import sys
import tempfile
from pathlib import Path
from pprint import pformat
from typing import Any, List, Optional, Tuple

import folium
import folium.plugins
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import rasterio
from dotenv import load_dotenv
from loguru import logger
from matplotlib import cm
from pystac import Item
from pystac_client import Client
from rasterio import warp
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).parent))

from streamlit_folium import folium_static
from utils import Spinner, get_epsg_code, log_raster_info, timer

load_dotenv()

# Set env params
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
os.environ["AWS_HTTPS"] = "YES"
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"


# Function to create RGB composite from image array
def create_rgb_composite(image_array):
    """
    Create an RGB composite image from Sentinel-2 bands.

    Sentinel-2 bands used:
    - Red: Band 4 (665 nm) - Index 2 in image_array
    - Green: Band 3 (560 nm) - Index 1 in image_array
    - Blue: Band 2 (490 nm) - Index 0 in image_array

    The function normalizes each band to enhance visualization.
    """
    # Extract RGB bands (bands 2, 3, 4 correspond to indices 0, 1, 2)
    r = image_array[2]  # Red - Band 4
    g = image_array[1]  # Green - Band 3
    b = image_array[0]  # Blue - Band 2

    # Normalize each band to 0-255 range for visualization
    def normalize(band):
        min_val = np.percentile(band, 2)  # 2nd percentile to avoid outliers
        max_val = np.percentile(band, 98)  # 98th percentile to avoid outliers
        norm = np.clip((band - min_val) / (max_val - min_val), 0, 1) * 255
        return norm.astype(np.uint8)

    r_norm = normalize(r)
    g_norm = normalize(g)
    b_norm = normalize(b)

    # Stack the bands to create RGB image (height, width, channels)
    rgb = np.stack([r_norm, g_norm, b_norm], axis=2)

    return rgb


# Get the least cloudy image, the clipping window around a given a point, and its transform object
def get_less_cloudy_image(
    coords: List[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 10,
    npixels: int = 100,
    plotimage: bool = False,
    asset_key: str = "B08_10m",
) -> Tuple[Optional[Item], Optional[Window], Optional[rasterio.Affine]]:
    """
    Read the Sentinel-2 STAC catalogue and return the best image (i.e. less cloud coverage), at the
    specified location and using the given start and end dates.

    INPUT   -   coords:             List with [Lon, Lat] coordinates
            -   start_date:         Start date as a string with format "YYYY-MM-DD"
            -   end_date:           End date as a string with format "YYYY-MM-DD"
            -   max_cloud_cover:    Integer between 0-100
            -   npixels:            Number of pixels that defines the size of the image
            -   plotimage:          Plot the less cloudy image (True/False)
    OUTPUT  -   best_image:         Less cloudy image
            -   window:             Object for extracting the image subset
            -   transform:          Object for georeferencing the image subset
            -   bounds:             Lon/Lat bounds coordinates
    """
    # API
    # catalog_url = "https://earth-search.aws.element84.com/v1"
    catalog_url = "https://stac.dataspace.copernicus.eu/v1"

    # Initialise the connection to the STAC catalog (CDSE)
    catalog = Client.open(catalog_url)
    catalog.add_conforms_to("ITEM_SEARCH")
    logger.info(f"Catalog: {catalog}")
    logger.info(f"Catalog conforms to: {catalog.conforms_to}")
    logger.info(f"Catalog collections: {catalog.get_collections()}")
    logger.info(f"Catalog Type: {type(catalog)}/n{dir(catalog)}")

    # Build the search params: Image retrieval parameters
    intersects_dict = dict(type="Point", coordinates=coords)
    dates = start_date + "/" + end_date
    filters = {
        "eo:cloud_cover": dict(lt=max_cloud_cover),
    }

    # create search params
    search_params = {
        "collections": ["sentinel-2-l2a"],
        "intersects": intersects_dict,
        "datetime": dates,
        "query": filters,
    }

    logger.info(f"Search params:\n{pformat(search_params)}")

    # Search the Sentinel-2 COGs catalog
    search = catalog.search(**search_params)

    # Sort items by cloud cover and return the less cloudy image
    items = list(search.items())
    logger.info(f"Search returned {len(items)} items")

    if len(items) > 0:
        best_image = sorted(items, key=lambda item: item.properties["eo:cloud_cover"])[
            0
        ]
        logger.info(f"Best image: {best_image} [type: {type(best_image)}]")
        logger.info(f"Best image assets: {best_image.assets.keys()}")

        ## Get the projection of the image
        epsg_code = get_epsg_code(best_image)

        # Read band 8 and calculate the clipping window
        # img = rasterio.open(best_image.assets[asset_key].href)
        with rasterio.open(best_image.assets[asset_key].href) as img:
            # Extract raster parameters
            ncols, nrows = img.width, img.height
            sref = img.crs

            # Transform Lon/Lat to the image CRS
            EPSG = 4326
            sref_wgs84 = rasterio.crs.CRS.from_epsg(EPSG)
            x, y = warp.transform(sref_wgs84, sref, [coords[0]], [coords[1]])
            row, col = rasterio.transform.rowcol(img.transform, x[0], y[0])

            # Enforce npixels is less than the image size
            if (npixels * 2 > ncols) or (npixels * 2 > nrows):
                npixels = min(nrows / 4, ncols / 4)

            # Define rasterio window
            row_start = row - npixels
            row_stop = row + npixels
            col_start = col - npixels
            col_stop = col + npixels
            window = rasterio.windows.Window.from_slices(
                (row_start, row_stop), (col_start, col_stop)
            )

            # Create the transform object for the subset area
            transform = img.window_transform(window)

            # Plot the image band
            if False:
                plt.imshow(img, cmap="Greys_r")
                plt.scatter([col], [row], s=200, c="yellow", marker="+")
                plt.axis("off")

        # Return image, the clipping window, and its transform object
        return best_image, window, transform
    else:
        logger.error("No images found. Change search parameters")
        return None, None, None


# Read a subset of the image bands and create an array
def read_sentinel2(
    best_image: Item,
    window: Window,
    transform: rasterio.Affine,
    plotimage: bool = False,
) -> Tuple[numpy.ndarray, List[List[float]], rasterio.crs.CRS]:
    """
    Read a subset of the best satellite image (STAC item) and return a numpy array
    for the clustering process.

    INPUT   -   best_image:     COG image selected using the API (pystac item)
            -   window:         Rasterio's Window object that defines the clipping boundary
            -   plotimage:      Plot the clipped satellite image (True/False)

    OUTPUT  -   image_array:    3D numpy array with the clipped image
    """
    # Sentinel-2 10-m bands
    bands = [2, 3, 4, 8]

    # Create an empty array to store each band as a column
    arr = numpy.empty((0))

    # Loop through the bands
    for b in bands:
        # Create band name string
        band_name = f"B{str(b).zfill(2)}_10m"

        # Get the url to the COG
        try:
            asset_url = best_image.assets[band_name].href
        except KeyError:
            raise KeyError(
                f"Band {band_name} not found in the image assets. Try: {best_image.assets.keys()}"
            )

        # Check if asset is a COG (for the first band only)
        if b == bands[0]:
            logger.info(f"Checking COG characteristics for band {band_name}...")
            with rasterio.open(asset_url) as src:
                logger.info(f"Raster driver: {src.driver}")
                logger.info(f"Has overviews: {len(src.overviews(1)) > 0}")
                logger.info(f"Internal tiling: {src.profile.get('tiled', False)}")
                logger.info(f"Block size: {src.block_shapes}")

        # Open the COG image
        with rasterio.open(asset_url) as img:
            # Read only the pixels inside the window
            img_arr = img.read(1, window=window)

            # Extract windowed image bounds
            sref = img.crs
            sref_wgs84 = rasterio.crs.CRS.from_epsg(4326)
            bounds_utm = rasterio.transform.array_bounds(
                img_arr.shape[0], img_arr.shape[1], transform
            )
            image_bounds = warp.transform(
                sref,
                sref_wgs84,
                [bounds_utm[0], bounds_utm[2]],
                [bounds_utm[1], bounds_utm[3]],
            )

            # Reshape and append each band
            arr = numpy.append(arr, img_arr.flatten())

    # Reshape array
    image_array = arr.reshape(len(bands), img_arr.shape[0], img_arr.shape[1])

    if plotimage:
        plt.imshow(img_arr, cmap="Greys_r")
        plt.axis("off")

    return image_array, image_bounds, sref


# Perform a K-Means clustering
def get_clusters(
    image_array: numpy.ndarray, nclusters: int = 5, plotimage: bool = False
) -> numpy.ndarray:
    """
    Using the satellite image subset array, calculate the specified
    number of clusters and return a numpy array.

    INPUT   -   image_array:        Satellite image stored as a 3D array
            -   nclusters:          Number of clusters
            -   plotimage:          Plot the clustered image (True/False)
    OUTPUT  -   clusters_array:     Clusters stored as a 2D array
    """
    # Reshape the array
    arr_kmeans = image_array.reshape([image_array.shape[0], -1]).T

    # Normalize the data
    scaled = MinMaxScaler().fit_transform(arr_kmeans)

    # Calculate clusters
    with Spinner(f"Calculating {nclusters} clusters"):
        clusters = KMeans(n_clusters=nclusters).fit(scaled)

    # Convert the clusters to a raster image
    rowcol = int(clusters.labels_.shape[0] ** 0.5)
    clusters_array = clusters.labels_.reshape(rowcol, rowcol)

    if plotimage:
        # Plot the clusters
        plt.imshow(clusters_array, cmap="tab20")
        plt.axis("off")

    return clusters_array


# Create dataframe to visualize area by clusters
def get_areas(clusters_array: numpy.ndarray) -> pandas.DataFrame:
    """
    Calculate the area (in hectares) covered by each cluster, and
    return a pandas dataframe.

    INPUT   -   clusters_array:     Clusters stored as a 2D array
    OUTPUT  -   df_areas:           Dataframe with the areas in hectares
    """
    # Get summary table
    clusters, counts = numpy.unique(clusters_array, return_counts=True)
    # Transform area from m2 to hectares. Pixel size: 10x10m
    counts = [npixels * 100 / 10000.0 for npixels in counts]
    # Create dataframe
    df_areas = pandas.DataFrame(
        list(zip(clusters, counts)), columns=["Clusters", "Area (ha)"]
    )

    return df_areas


# folium map
def show_folium_map(
    clusters_array: numpy.ndarray,
    lon: float,
    lat: float,
    image_bounds: List[List[float]],
    rgb_composite: numpy.ndarray = None,
) -> Any:
    """
    Create a folium map using as background Esri's satellite
    tiles, and on top showing the clusters.

    INPUT   -   clusters_array:     Clusters stored as a 2D array
            -   lon:                Longitude (user's input)
            -   lat:                Latitude (user's input)
            -   bounds:             List of the clusters bounds
            -   rgb_composite:      Optional RGB composite image (height, width, 3)
    OUTPUT  -   folium_map:         folium object
    """
    # Create folium map
    map_bounds = [
        [image_bounds[1][0], image_bounds[0][0]],
        [image_bounds[1][1], image_bounds[0][1]],
    ]

    m = folium.Map(location=[lat, lon], tiles=None)

    # Add tile layer to map first
    tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    attribution = "Tiles &copy; Esri Imagery"
    folium.TileLayer(
        tiles=tiles, attr=attribution, name="Esri Imagery", control=False
    ).add_to(m)

    # Create FeatureGroup for RGB composite if provided and add it first
    if rgb_composite is not None:
        fg_rgb = folium.FeatureGroup(name="RGB True Color")
        folium.raster_layers.ImageOverlay(
            image=rgb_composite, bounds=map_bounds, opacity=0.9
        ).add_to(fg_rgb)
        fg_rgb.add_to(m)  # Add RGB group to map

    # Create FeatureGroup for clusters and add it second
    fg_clusters = folium.FeatureGroup(name="Image Clusters")
    folium.raster_layers.ImageOverlay(
        image=clusters_array, colormap=cm.tab10, bounds=map_bounds, opacity=0.7
    ).add_to(fg_clusters)
    fg_clusters.add_to(m)  # Add Cluster group to map

    # Add LayerControl after adding layers
    folium.LayerControl(collapsed=False).add_to(m)

    # Add full screen button
    folium.plugins.Fullscreen().add_to(m)

    # Fit map bounds
    m.fit_bounds(map_bounds)

    # Render Folium map
    # Uncomment the import line at the top of this file
    # from streamlit_folium import folium_static
    folium_map = folium_static(m, width=1000)

    # Return the map object directly instead of using folium_static
    return m


# Export clusters to GeoTIFF
def export_clusters(
    clusters_array: numpy.ndarray, sref: rasterio.crs.CRS, transform: rasterio.Affine
) -> str:
    """
    Export the clusters array to GeoTIFF.

    INPUT   -   clusters_array:     Numpy array with the clusters
            -   sref:               Spatial reference of the original image
    OUTPUT  -   GeoTIFF:            File named 'clusters.tif'
    """

    # Export clusters
    profile = {
        "driver": "GTiff",
        "height": clusters_array.shape[0],
        "width": clusters_array.shape[1],
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": sref,
        "transform": transform,
    }

    # Create a temprary file TIFF to store the clusters
    tmpfile = tempfile.NamedTemporaryFile("w")
    fn = tmpfile.name + ".tif"
    with rasterio.open(fn, "w", **profile) as dataset:
        dataset.write(clusters_array, 1)
    return fn


if __name__ == "__main__":
    # Define the point of interest ! (npixels = number of pixels around that point)
    lon, lat = -2.130144, 51.632115

    with timer(f"{'-' * 10}\nGetting image"):
        best_image, window, transform = get_less_cloudy_image(
            coords=[lon, lat],
            start_date="2022-01-01",
            end_date="2024-01-31",
            max_cloud_cover=10,
            npixels=100,
            plotimage=True,
        )

    with timer(f"{'-' * 10}\nReading Sentinel-2 data"):
        image_array, image_bounds, sref = read_sentinel2(
            best_image, window, transform, plotimage=True
        )
    log_raster_info(best_image.assets["B08_10m"].href)

    with timer(f"{'-' * 10}\nClustering"):
        clusters_array = get_clusters(image_array=image_array, nclusters=5)

    # Create clusters GeoTIFF
    output_file = export_clusters(clusters_array, sref, transform)

    df_areas = get_areas(clusters_array)

    logger.info(f"Clustering complete. Output file: {output_file}")
    logger.info(f"Areas by cluster:\n{df_areas}")

    # Create RGB composite
    rgb_composite = create_rgb_composite(image_array)

    # Show the map
    show_folium_map(clusters_array, lon, lat, image_bounds, rgb_composite)

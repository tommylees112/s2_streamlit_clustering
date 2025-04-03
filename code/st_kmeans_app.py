from datetime import date, timedelta

import stac_kmeans as sk
import streamlit as st
from altair import Chart, Color, Scale

st.set_page_config(
    page_title="Sentinel-2 Image Clustering", page_icon=":satellite:", layout="wide"
)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Main title
st.title("🛰️ Sentinel-2 Image Clustering [Cloud Geospatial 101 Capstone]")

# Add data source information
st.info(
    "🛰️ **Space Explorer!** This app connects to the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) "
    "using fancy STAC technology to bring you stunning Sentinel-2 satellite imagery. 🌍\n\n"
    "✨ Simply pick your favorite spot on Earth, set a date range, and we'll hunt down the clearest image "
    "(goodbye clouds! ☁️) before transforming it into colorful clusters with K-means magic! 🧩"
)

# Streamlit parameters
st.sidebar.title("🎮 Control Panel: Pick Your Spot! 🌍")
with st.sidebar.form(key="params_form"):
    lon = st.number_input(
        label="Longitude",
        format="%.5f",
        step=0.000001,
        min_value=-180.0,
        max_value=180.0,
        value=-2.130144,  # -124.20160,  # -2.130144,
    )
    lat = st.number_input(
        label="Latitude",
        format="%.5f",
        step=0.000001,
        min_value=-90.0,
        max_value=90.0,
        value=51.632115,  # 49.17965,  # 51.632115
    )
    start_date = st.date_input(label="Start date", value=date(2021, 7, 15))
    min_value = start_date + timedelta(days=1)
    end_date = st.date_input(
        label="End date", value=date(2024, 8, 15), min_value=min_value
    )
    max_cloud_cover = st.slider(
        label="Maximum cloud cover % (0-50)",
        value=10,
        step=5,
        min_value=0,
        max_value=50,
    )
    npixels = (
        st.slider(
            label="Image width/height in kilometers (1-10)",
            value=1,
            step=1,
            min_value=1,
            max_value=10,
        )
        * 100
        / 2
    )
    nclusters = st.slider(
        label="Number of clusters (2-10)", value=4, step=1, min_value=2, max_value=10
    )
    sub1 = st.form_submit_button(label="🚀 Launch Clustering Mission!")

# Read the STAC catalog and retrieve the best satellite image
with st.spinner("Downloading satellite imagery..."):
    best_image, window, transform = sk.get_less_cloudy_image(
        coords=[lon, lat],
        start_date=str(start_date),
        end_date=str(end_date),
        max_cloud_cover=max_cloud_cover,
        npixels=npixels,
    )

if best_image:
    # Read image subset
    image_array, image_bounds, sref = sk.read_sentinel2(
        best_image=best_image, window=window, transform=transform
    )

    # K-means clustering
    with st.spinner("Image clustering in progress..."):
        # K-means clustering
        clusters_array = sk.get_clusters(image_array=image_array, nclusters=nclusters)

        # Create clusters GeoTIFF
        output_file = sk.export_clusters(clusters_array, sref, transform)
        # Download clusters GeoTIFF
        with open(output_file, "rb") as img:
            st.sidebar.download_button(
                "Download clusters", img, file_name="clusters.tif"
            )

        # Calculate clusters areas in hectares
        df_areas = sk.get_areas(clusters_array)

    # Create bar chart
    st.header("📊 Area(ha) covered by each cluster")

    # Define custom Altair parameters
    # The colour palette is the same one used for the map clusters
    c = (
        Chart(df_areas, width=1000)
        .mark_bar()
        .encode(
            x="Clusters:N",
            y="Area (ha):Q",
            color=Color("Clusters:N", scale=Scale(scheme="category10")),
            tooltip="Area (ha)",
        )
        .interactive()
    )

    # Show the chart
    st.altair_chart(c)

    # Show the map
    st.header("🗺️ Satellite image clusters")

    # Create a checkbox to toggle RGB layer visibility
    show_rgb = st.checkbox(
        "🌈 Show true color RGB image",
        value=True,
        help="Display the actual satellite image using Red, Green, and Blue bands",
    )

    # Create RGB composite from image_array if checkbox is checked
    rgb_composite = sk.create_rgb_composite(image_array)

    # Add both the clusters and RGB composite to the map
    m = sk.show_folium_map(
        clusters_array=clusters_array,
        lon=lon,
        lat=lat,
        image_bounds=image_bounds,
        rgb_composite=rgb_composite,  # Pass the RGB composite to the function
    )

else:
    st.warning(
        "😮 Oops! No satellite images found with those parameters. Try adjusting your date range or increasing the cloud cover tolerance! 🌤️"
    )

# Display best_image metadata as JSON
if best_image:
    st.header("🔍 Satellite Image Metadata")
    with st.expander(f"✨ View Best Image Metadata (lowest cloud cover)\n{best_image}"):
        # Convert PySTAC Item to dict and display as JSON
        best_image_dict = best_image.to_dict()
        st.json(best_image_dict)

st.sidebar.info(
    "Code available at [Github](https://www.github.com/julionovoa/st-kmeans)"
)

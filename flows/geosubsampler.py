# This file is your entry point:
# - add you Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('geosubsampler.py')
#   and everything is gonna be ok.
#
# Remember: everything is gonna be ok in the end: if it's not ok, it's not the end.
# Alternatively, ask for help at https://github.com/deeplime-io/onecode/issues

from onecode import Logger, file_input, file_output, checkbox, number_input, text_input
from .calcs.StructuralOrientationSubSampler import StructuralOrientationSubSampler
from .calcs.StructuralPolygonSubSampler import StructuralPolygonSubSampler
from .calcs.PolygonTriangulator import PolygonTriangulator
from .calcs.FaultLineMerger import FaultLineMerger
from .calcs.subsampleFaults import subsampleFaults
import geopandas as gpd
import matplotlib.pyplot as plt
import zipfile
import tempfile
from pathlib import Path


def run():
    Logger.info(
        """
        #####################################################################
        ###> Hello from GeoSubSampler!
        #####################################################################
        """
    )

    Par = read_input_parameters()
    run_subsampler(Par)


class InputParameters:
    """
    A class to contains all input parameters.
    """

    def __init__(self):
        # -------------------------------
        # Section 'FilePaths'.
        # -------------------------------
        #

        self.do_points = checkbox(
            key="do_points", value=True, label="Structure Points SubSampling"
        )

        self.structures_input_path = file_input(
            key="my_structures",
            value="uploads/structure_points.zip",
            label="Select a zipped point shapefile",
            types=[("Shapefiles", ".zip .ZIP")],
            make_path=True,  # will create the uploads folder if doesn't exist)
        )

        self.structures_output_path = file_output(
            key="structures_output_path",
            value="output/structures_SubSampled.zip",
            make_path=True,  # will create the output folder if doesn't exist
        )

        self.dipField = text_input(
            key="dipField",
            value="Dip",
            label="Dip info Field Name",
            max_chars=30,
            placeholder="Choose a field name from your structure point file ",
        )

        self.dipDirField = text_input(
            key="dipDirField",
            value="Strike",
            label="Dip Direction or Strike Field Name",
            max_chars=30,
            placeholder="Choose a field name from your structure point file ",
        )

        self.pointGridSize = number_input(
            key="pointGridSize",
            value=10,
            label="Orientations GridSize (km)",
            min=0.1,
            max=1000,
        )

        self.do_faults = checkbox(
            key="do_faults", value=True, label="Faults Polyline SubSampling"
        )

        self.faults_input_path = file_input(
            key="my_faults",
            value="uploads/faults_polylines.zip",
            label="Select a zipped polyline shapefile",
            types=[("Shapefiles", ".zip .ZIP")],
            make_path=True,  # will create the uploads folder if doesn't exist)
        )

        self.faults_output_path = file_output(
            key="faults_output_path",
            value="output/faults_SubSampled.zip",
            make_path=True,  # will create the output folder if doesn't exist
        )

        self.faultMinLength = number_input(
            key="faultMinLength",
            value=10,
            label="Fault Min Length (km)",
            min=0.1,
            max=1000,
        )
        self.do_geology = checkbox(
            key="do_geology", value=True, label="Geology Polygon SubSampling"
        )

        self.geology_input_path = file_input(
            key="my_geology",
            value="uploads/geology_polygons.zip",
            label="Select a zipped polygon shapefile",
            types=[("Shapefiles", ".zip .ZIP")],
            make_path=True,  # will create the uploads folder if doesn't exist)
        )

        self.geology_output_path = file_output(
            key="geology_output_path",
            value="output/geology_SubSampled.zip",
            make_path=True,  # will create the output folder if doesn't exist
        )

        self.priority1Field = text_input(
            key="priority1Field",
            value="UNITNAME",
            label="Polygon Priority 1 Field Name",
            max_chars=30,
            placeholder="Choose a field name from your geology polygon file ",
        )

        self.priority2Field = text_input(
            key="priority2Field",
            value="GROUP_",
            label="Polygon Priority 2 Field Name",
            max_chars=30,
            placeholder="Choose a field name from your geology polygon file ",
        )

        self.priority3Field = text_input(
            key="priority3Field",
            value="SUPERGROUP",
            label="Polygon Priority 3 Field Name",
            max_chars=30,
            placeholder="Choose a field name from your geology polygon file ",
        )

        self.priority4Field = text_input(
            key="priority4Field",
            value="CRATON",
            label="Polygon Priority 4 Field Name",
            max_chars=30,
            placeholder="Choose a field name from your geology polygon file ",
        )

        self.priority5Field = text_input(
            key="priority5Field",
            value="OROGEN",
            label="Polygon Priority 5  Field Name",
            max_chars=30,
            placeholder="Choose a field name from your geology polygon file ",
        )

        self.do_dykes = checkbox(
            key="do_dykes", value=True, label="Handle dyke polygons separately"
        )

        self.dykeField = text_input(
            key="dykeField",
            value="CODE",
            label="Dyke Field Name",
            max_chars=30,
            placeholder="Choose a dyke property field name from your geology polygon file ",
        )

        self.dykeCodes = text_input(
            key="dykeCodes",
            value="P_-_wx-o, P_-_wz-ow, P_-_wz-om, P_-_ww-o, P_-_wz-og, P_-_wt-o",
            label="Codes in chosen field that identify dykes",
            max_chars=200,
            placeholder="Comma separated list of dyke codes",
        )

        self.geologyMinDiam = number_input(
            key="geologyMinDiam",
            value=50,
            label="Geology Polygon Min Diam (km)",
            min=0.1,
            max=1000,
        )

        self.distance_threshold = number_input(
            key="distance_threshold",
            value=1,
            label="Geology Polygon node mismatch tolerance (projection units)",
            min=0.01,
            max=100000,
        )

        self.filePointsOrig = file_output(
            key="filePointsOrig",
            value="output/filePointsOrig.png",
            label="PointsOrig",
            tags=["png"],
            make_path=True,
        )
        self.filePointsSub = file_output(
            key="filePointsSub",
            value="output/filePointsSub.png",
            label="PointsSub",
            tags=["png"],
            make_path=True,
        )
        self.fileFaultsOrig = file_output(
            key="fileFaultsOrig",
            value="output/fileFaultsOrig.png",
            label="FaultsOrig",
            tags=["png"],
            make_path=True,
        )
        self.fileFaultsSub = file_output(
            key="fileFaultsSub",
            value="output/fileFaultsSub.png",
            label="FaultsSub",
            tags=["png"],
            make_path=True,
        )
        self.fileGeologyOrig = file_output(
            key="fileGeologyOrig",
            value="output/fileGeologyOrig.png",
            label="GeologyOrig",
            tags=["png"],
            make_path=True,
        )
        self.fileGeologySub = file_output(
            key="fileGeologySub",
            value="output/fileGeologySub.png",
            label="GeologySub",
            tags=["png"],
            make_path=True,
        )


def read_input_parameters():
    return InputParameters()


def plot_results(gdf, path, color, gtype, field, title="Subsampled"):
    """
    Plot the results of the subsampling process.
    """
    Logger.info("Plotting results...")

    # Load the sampled data

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    if gtype == "points":
        plot_normal_vectors(gdf, azimuth_col=field, vector_length=10000.0)

        # gdf.plot(ax=ax, color=color, markersize=5, label="Structures")
    elif gtype == "faults":
        gdf.plot(ax=ax, color=color, linewidth=1.5, label="Faults")
    else:
        gdf.plot(ax=ax, column="UNITNAME", alpha=0.5, label="Geology")

    ax.set_title(f"{title} {gtype}")
    plt.savefig(path)


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def plot_normal_vectors(
    gdf, azimuth_col="azimuth", vector_length=0.001, figsize=(10, 10), arrow_props=None
):
    """
    Plot points as vectors normal (perpendicular) to the azimuth direction.

    Parameters:
    gdf: GeoDataFrame with Point geometries and azimuth column
    azimuth_col: name of the azimuth column (in degrees)
    vector_length: length of the vectors in map units
    figsize: figure size
    arrow_props: dictionary of arrow styling properties
    """

    # Default arrow properties
    if arrow_props is None:
        arrow_props = {
            "arrowstyle": "-",
            "color": "red",
            "linewidth": 1.5,
            "alpha": 0.7,
        }

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the base points
    # gdf.plot(ax=ax, color="blue", markersize=50, alpha=0.6)

    # Extract coordinates
    x_coords = gdf.geometry.x
    y_coords = gdf.geometry.y
    azimuths = gdf[azimuth_col]

    # Convert azimuth to normal direction
    # Azimuth is typically measured clockwise from North
    # Normal vectors are perpendicular (add 90 degrees)
    normal_angles = azimuths

    # Convert to radians for trigonometry
    normal_radians = np.radians(normal_angles)

    # Calculate vector end points
    dx = vector_length * np.sin(normal_radians)
    dy = vector_length * np.cos(normal_radians)

    # Add vectors to the plot
    for i in range(len(gdf)):
        start = (x_coords.iloc[i], y_coords.iloc[i])
        end = (x_coords.iloc[i] + dx.iloc[i], y_coords.iloc[i] + dy.iloc[i])

        arrow = FancyArrowPatch(start, end, **arrow_props)
        ax.add_patch(arrow)

    # Set axis limits based on your data bounds
    buffer = vector_length * 2
    ax.set_xlim(x_coords.min() - buffer, x_coords.max() + buffer)
    ax.set_ylim(y_coords.min() - buffer, y_coords.max() + buffer)

    # Set equal aspect ratio and add labels
    ax.set_aspect("equal")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Strike Direction")

    # Add a grid for better visualization
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def write_zipped_shapefile(gdf, output_zip_path, layer_name="data"):
    """
    Write a GeoDataFrame to a zipped shapefile with all associated files

    Parameters:
    gdf (GeoDataFrame): The geodataframe to write
    output_zip_path (str): Path for the output zip file
    layer_name (str): Name for the shapefile (default: "data")
    """

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shapefile_path = temp_path / f"{layer_name}.shp"

        # Write the shapefile to temporary directory
        gdf.to_file(shapefile_path, driver="ESRI Shapefile")

        # Get all shapefile-related files
        shapefile_files = list(temp_path.glob(f"{layer_name}.*"))

        # Create the zip file
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in shapefile_files:
                # Add each file to zip with just the filename (no path)
                zipf.write(file_path, file_path.name)

        print(f"Zipped shapefile created: {output_zip_path}")
        print(f"Files included: {[f.name for f in shapefile_files]}")


def checkUnits(gdf, length=1):
    crs = gdf.crs
    is_geographic = crs.is_geographic if hasattr(crs, "is_geographic") else False

    if is_geographic:
        lengthScale = length * 110000
    else:
        lengthScale = length
    return lengthScale


def run_subsampler(Par):
    """
    Run the subsampling process based on the input parameters.
    """

    if Par.do_points:
        Logger.info("Running StructuralOrientationSubSampler...")
        gdf = gpd.read_file(Par.structures_input_path)
        sampler = StructuralOrientationSubSampler(
            gdf, dip_col="Dip", strike_col="Strike", dip_convention="strike"
        )

        pointGridSize = checkUnits(gdf, Par.pointGridSize)

        sampled = sampler.gridCellAveraging(grid_size=pointGridSize * 1000)

        write_zipped_shapefile(
            sampled, Par.structures_output_path, layer_name="structures_SubSampled"
        )
        plot_results(
            gdf,
            Par.filePointsOrig,
            "red",
            gtype="points",
            field=Par.dipDirField,
            title="Original",
        )
        plot_results(
            sampled,
            Par.filePointsSub,
            "red",
            gtype="points",
            field=Par.dipDirField,
            title="Subsampled",
        )

    if Par.do_faults:
        Logger.info("Running FaultLineMerger...")
        gdf = gpd.read_file(Par.faults_input_path)

        faultMinLength = checkUnits(gdf, Par.faultMinLength)

        sampler = subsampleFaults(faultMinLength * 1000)
        sampled = sampler.filter_geodataframe(gdf)

        write_zipped_shapefile(
            sampled, Par.faults_output_path, layer_name="faults_SubSampled"
        )
        plot_results(
            gdf,
            Par.fileFaultsOrig,
            "black",
            gtype="faults",
            field=Par.dipDirField,
            title="Original",
        )
        plot_results(
            sampled,
            Par.fileFaultsSub,
            "black",
            gtype="faults",
            field=Par.dipDirField,
            title="Subsampled",
        )

    if Par.do_geology:
        Logger.info("Running StructuralPolygonSubSampler...")
        gdf = gpd.read_file(Par.geology_input_path)

        geologyMinDiam = checkUnits(gdf, Par.geologyMinDiam)

        triangulator = PolygonTriangulator(
            gdf=gdf,
            id_column=Par.dykeField,
            min_area_threshold=geologyMinDiam * 1000000.0,
            distance_threshold=Par.distance_threshold,
            strat1=Par.priority1Field,
            strat2=Par.priority2Field,
            strat3=Par.priority3Field,
            strat4=Par.priority4Field,
            lithoname=Par.priority5Field,
        )
        target_ids = Par.dykeCodes.replace(" ", "").split(",")
        dyked = triangulator.triangulate_polygons(target_ids=target_ids)

        sampler = StructuralPolygonSubSampler(dyked)
        sampled = sampler.clean_small_polygons_and_holes_new(
            dyked,
            min_area_threshold=geologyMinDiam * 1000000.0,
            distance_threshold=Par.distance_threshold,
            strat1=Par.priority1Field,
            strat2=Par.priority2Field,
            strat3=Par.priority3Field,
            strat4=Par.priority4Field,
            lithoname=Par.priority5Field,
        )

        plot_results(
            gdf,
            Par.fileGeologyOrig,
            "gray",
            gtype="geology",
            field=Par.dipDirField,
            title="Original",
        )
        write_zipped_shapefile(
            sampled, Par.geology_output_path, layer_name="geology_SubSampled"
        )
        plot_results(
            sampled,
            Par.fileGeologySub,
            "gray",
            gtype="geology",
            field=Par.dipDirField,
            title="Subsampled",
        )

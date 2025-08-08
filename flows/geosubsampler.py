# This file is your entry point:
# - add you Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('geosubsampler.py')
#   and everything is gonna be ok.
#
# Remember: everything is gonna be ok in the end: if it's not ok, it's not the end.
# Alternatively, ask for help at https://github.com/deeplime-io/onecode/issues

import onecode
from onecode import Logger, file_input, file_output, checkbox, number_input, text_input
from .calcs.StructuralOrientationSubSampler import StructuralOrientationSubSampler
from .calcs.StructuralPolygonSubSampler import StructuralPolygonSubSampler
from .calcs.PolygonTriangulator import PolygonTriangulator
from .calcs.FaultLineMerger import FaultLineMerger
from .calcs.subsampleFaults import subsampleFaults
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
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
        )

        self.faultMinLength = number_input(
            key="faultMinLength",
            value=10,
            label="Min Length (km)",
            min=0.1,
            max=1000,
            value=target_ids,
            label="Polygon Priority 5",
            max_chars=30,
            placeholder="Comma separated list of dyke codes",
        )

        self.geologyMinDiam = number_input(
            key="geologyMinDiam",
            value=50,
            label="Geology Polygon Min Diam (km)",
            min=0.1,
            max=1000,
        )

        self.filePointsOrig = file_output(
            key="filePointsOrig",
            value="output/filePointsOrig.png",
            label="filePointsOrig",
            tags=["png"],
            make_path=True,
        )
        self.filePointsSub = file_output(
            key="filePointsSub",
            value="output/filePointsSub.png",
            label="filePointsSub",
            tags=["png"],
            make_path=True,
        )
        self.fileFaultsOrig = file_output(
            key="fileFaultsOrig",
            value="output/fileFaultsOrig.png",
            label="fileFaultsOrig",
            tags=["png"],
            make_path=True,
        )
        self.fileFaultsSub = file_output(
            key="fileFaultsSub",
            value="output/fileFaultsSub.png",
            label="fileFaultsSub",
            tags=["png"],
            make_path=True,
        )
        self.fileGeologyOrig = file_output(
            key="fileGeologyOrig",
            value="output/fileGeologyOrig.png",
            label="fileGeologyOrig",
            tags=["png"],
            make_path=True,
        )
        self.fileGeologySub = file_output(
            key="fileGeologySub",
            value="output/fileGeologySub.png",
            label="fileGeologySub",
            tags=["png"],
            make_path=True,
        )


def read_input_parameters():
    return InputParameters()


def run_subsampler(Par):
    """
    Run the subsampling process based on the input parameters.
    """
    Logger.info(f"Par: {Par.geology_input_path}")

    if Par.do_points:
        Logger.info("Running StructuralOrientationSubSampler...")
        gdf = gpd.read_file(Par.structures_input_path)
        sampler = StructuralOrientationSubSampler(
            gdf, dip_col="Strike", strike_col="Dip", dip_convention="strike"
        )
        sampled = sampler.gridCellAveraging(grid_size=Par.pointGridSize * 1000)
        # sampled.to_file(Par.structures_output_path)
        write_zipped_shapefile(
            sampled, Par.structures_output_path, layer_name="structures_SubSampled"
        )
        plot_results(gdf, Par.filePointsOrig, "red", gtype="points", title="Original")
        plot_results(
            sampled, Par.filePointsSub, "red", gtype="points", title="Subsampled"
        )

    if Par.do_faults:
        Logger.info("Running FaultLineMerger...")
        gdf = gpd.read_file(Par.faults_input_path)
        sampler = subsampleFaults(Par.faultMinLength * 1000)
        sampled = sampler.filter_geodataframe(gdf)
        # sampled.to_file(Par.faults_output_path)
        write_zipped_shapefile(
            sampled, Par.faults_output_path, layer_name="faults_SubSampled"
        )
        plot_results(gdf, Par.fileFaultsOrig, "black", gtype="faults", title="Original")
        plot_results(
            sampled, Par.fileFaultsSub, "black", gtype="faults", title="Subsampled"
        )

    if Par.do_geology:
        Logger.info("Running StructuralPolygonSubSampler...")
        gdf = gpd.read_file(Par.geology_input_path)

        triangulator = PolygonTriangulator(
            gdf=gdf,
            id_column=Par.dykeField,
            min_area_threshold=Par.geologyMinDiam * 1000000.0,
            distance_threshold=1,
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
            min_area_threshold=Par.geologyMinDiam * 1000000.0,
            distance_threshold=1,
            strat1=Par.priority1Field,
            strat2=Par.priority2Field,
            strat3=Par.priority3Field,
            strat4=Par.priority4Field,
            lithoname=Par.priority5Field,
        )
        # sampled.to_file(Par.geology_output_path)
        plot_results(
            gdf, Par.fileGeologyOrig, "gray", gtype="geology", title="Original"
        )
        write_zipped_shapefile(
            sampled, Par.geology_output_path, layer_name="geology_SubSampled"
        )
        plot_results(
            sampled, Par.fileGeologySub, "gray", gtype="geology", title="Subsampled"
        )


def plot_results(gdf, path, color, gtype, title="Subsampled"):
    """
    Plot the results of the subsampling process.
    """
    Logger.info("Plotting results...")

    # Load the sampled data

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    if gtype == "points":
        gdf.plot(ax=ax, color=color, markersize=5, label="Structures")
    elif gtype == "faults":
        gdf.plot(ax=ax, color=color, linewidth=1.5, label="Faults")
    else:
        gdf.plot(ax=ax, column="UNITNAME", alpha=0.5, label="Geology")

    ax.set_title(f"{title} {gtype}")
    plt.savefig(path)


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

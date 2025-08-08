import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings
from typing import Union, Tuple, Optional
from math import acos, atan2, degrees


class StructuralOrientationSubSampler:
    """
    A comprehensive class for subsampling structural orientation point data using
    various methods described in geological modeling literature.

    This class implements six different subsampling methods:
    1. Decimation
    2. Stochastic subsampling
    3. Grid cell averaging
    4. Spherical statistics (Kent distribution)
    5. Outlier removal
    6. First-order subsampling (proximity to contacts)
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        dip_col: Optional[str] = None,
        strike_col: Optional[str] = None,
        dip_convention: Optional[str] = "dip_direction",
    ):
        """
        Initialize the subsampler with a GeoDataFrame of structural orientation points.

        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame containing structural orientation measurements.
            Must contain columns for strike/dip or dip direction/dip.
        """
        self.dip_col = dip_col
        self.strike_col = strike_col
        self.dip_convention = dip_convention

        self.gdf = gdf.copy()
        self._validate_input()
        self._prepare_vectors()

    def _validate_input(self):
        """Validate input GeoDataFrame structure."""
        required_geom = all(self.gdf.geometry.type == "Point")
        if not required_geom:
            raise ValueError("All geometries must be Point type")

    def strike_dip_vector(
        self, strike: Union[list, np.ndarray], dip: Union[list, np.ndarray]
    ) -> np.ndarray:
        """
        Calculates the strike-dip vector from the given strike and dip angles.

        Args:
            strike (Union[float, list, numpy.ndarray]): The strike angle(s) in degrees. Can be a single value or an array of values.
            dip (Union[float, list, numpy.ndarray]): The dip angle(s) in degrees. Can be a single value or an array of values.

        Returns:
            numpy.ndarray: The calculated strike-dip vector(s). Each row corresponds to a vector,
            and the columns correspond to the x, y, and z components of the vector.

        Note:
            This code is adapted from LoopStructural.
        """

        # Initialize a zero vector with the same length as the input strike and dip angles
        vec = np.zeros((len(strike), 3))

        # Convert the strike and dip angles from degrees to radians
        s_r = np.deg2rad(strike)
        d_r = np.deg2rad(dip)

        # Calculate the x, y, and z components of the strike-dip vector
        vec[:, 0] = np.sin(d_r) * np.cos(s_r)
        vec[:, 1] = -np.sin(d_r) * np.sin(s_r)
        vec[:, 2] = np.cos(d_r)

        # Normalize the strike-dip vector
        vec /= np.linalg.norm(vec, axis=1)[:, None]

        return vec

    def _prepare_vectors(self):
        """Convert strike/dip measurements to 3D unit vectors."""
        # Try to identify strike and dip columns

        self.gdf["vector"] = self.gdf.apply(
            lambda row: self._strike_dip_to_vector(
                row[self.strike_col], row[self.dip_col], self.dip_convention
            ),
            axis=1,
        )
        # print("vector",self.gdf['vector'])

    def _strike_dip_to_vector(
        self, strike, dip, dip_convention
    ) -> Tuple[float, float, float]:
        """Convert strike and dip to 3D unit vector (direction cosines)."""

        # Convert QVariant to float if needed
        if hasattr(strike, "value"):  # Check if it's a QVariant
            strike = strike.value()
        if hasattr(dip, "value"):  # Check if it's a QVariant
            dip = dip.value()

        # Handle None/null values
        if strike is None or dip is None:
            return (0, 0, 1)  # Default vector

        # Convert to float to ensure numeric type
        try:
            strike = float(strike)
            dip = float(dip)
        except (ValueError, TypeError):
            return (0, 0, 1)  # Default vector if conversion fails

        if dip_convention != "dip_direction":
            strike = np.mod(strike - 90.0, 360)

        vec = self.strike_dip_vector(np.array([strike]), np.array([dip]))
        # print("dd,d,i,j,k",strike,dip,vec[:,0], vec[:, 1], vec[:, 2])
        # return((vec[:, 0], vec[:, 1], vec[:, 2]))
        return (
            float(vec[0, 0]),
            float(vec[0, 1]),
            float(vec[0, 2]),
        )  # Returns tuple of scalars

    def _vector_to_strike_dip(
        self, vector: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """Convert 3D unit vector back to strike and dip."""
        l, m, n = vector

        # Calculate dip
        dip = degrees(acos(abs(n)))

        # Calculate strike
        if l != 0:
            strike = degrees(atan2(l, -m))
            if strike < 0:
                strike += 360
        else:
            strike = 90 if m < 0 else 270

        return strike, dip

    def decimation(self, n: int = 2) -> gpd.GeoDataFrame:
        """
        Decimate points by keeping every nth measurement.

        Parameters:
        -----------
        n : int
            Keep every nth point (e.g., n=2 keeps every second point)

        Returns:
        --------
        gpd.GeoDataFrame
            Subsampled GeoDataFrame
        """
        indices = range(0, len(self.gdf), n)
        return self.gdf.iloc[indices].reset_index(drop=True).drop("vector", axis=1)

    def stochastic_subsampling(
        self, retain_percentage: float = 50.0, random_state: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """
        Randomly subsample points to retain a specified percentage.

        Parameters:
        -----------
        retain_percentage : float
            Percentage of points to retain (0-100)
        random_state : int, optional
            Random seed for reproducibility

        Returns:
        --------
        gpd.GeoDataFrame
            Subsampled GeoDataFrame
        """
        if not 0 < retain_percentage <= 100:
            raise ValueError("retain_percentage must be between 0 and 100")

        n_retain = int(len(self.gdf) * retain_percentage / 100)

        if random_state:
            np.random.seed(random_state)

        indices = np.random.choice(len(self.gdf), size=n_retain, replace=False)
        return self.gdf.iloc[indices].reset_index(drop=True).drop("vector", axis=1)

    def dircos2ddd(self, l, m, n):
        dipdir = (360 + degrees(atan2(l, m))) % 360
        dip = 90 - degrees(np.arcsin(n))
        if dip > 90:
            dip = 180 - dip
            dipdir = (dipdir + 180) % 360
        return (dip, dipdir)

    def ddd2dircos(self, dip, dipdir):
        dip_rad = np.radians(dip)
        dipdir_rad = np.radians(dipdir)
        l = np.sin(dip_rad) * np.sin(dipdir_rad)
        m = np.sin(dip_rad) * np.cos(dipdir_rad)
        n = np.cos(dip_rad)
        return l, m, n

    def calc_vector(self, gpdataframe, dip, dipdir):
        df = pd.DataFrame(gpdataframe)
        data = []
        for index, row in df.iterrows():
            value = tuple(self.ddd2dircos(float(row[dip]), float(row[dipdir])))
            data.append(value)

        # Instead of concatenating, directly assign or update the existing vector column
        df["vector"] = data  # This will overwrite any existing vector column
        return df

    def calc_mean_vector(self, df):
        # Ensure we have a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Debug: Check the structure

        # Handle the vector column - it might be nested differently
        if "vector" in df.columns:
            vector_series = df["vector"]

            # Convert to list of vectors
            if isinstance(vector_series, pd.Series):
                vectors_list = vector_series.tolist()
            else:
                # If it somehow became a DataFrame
                vectors_list = vector_series.iloc[:, 0].tolist()
        else:
            raise ValueError("No 'vector' column found in DataFrame")

        # Extract l, m, n components

        df = df.copy()
        df.loc[:, "l"] = [v[0] for v in vectors_list]
        df.loc[:, "m"] = [v[1] for v in vectors_list]
        df.loc[:, "n"] = [v[2] for v in vectors_list]

        # Calculate sums and counts

        d = df["l"].sum()
        e = df["m"].sum()
        f = df["n"].sum()
        count = len(df.index)

        # resultant vector
        g = np.sqrt(float(d**2) + (e**2) + (f**2))

        # normalized r
        d2 = d / count
        e2 = e / count
        f2 = f / count
        g2 = np.sqrt(float(d2**2) + (e2**2) + (f2**2))

        # mean vector
        d3 = d2 / g2
        e3 = e2 / g2
        f3 = f2 / g2
        g3 = np.sqrt(float(d2**2) + (e2**2) + (f2**2))

        mean_vector = self.dircos2ddd(d3, e3, f3)
        return mean_vector

    def gridCellAveraging(self, grid_size: float = 1000.0) -> gpd.GeoDataFrame:
        """
        Average measurements within grid cells.

        Parameters:
        -----------
        grid_size : float
            Size of grid cells in map units (meters)

        Returns:
        --------
        gpd.GeoDataFrame
            Subsampled GeoDataFrame with averaged measurements
        """
        # Create grid
        bounds = self.gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        df = self.calc_vector(self.gdf, self.dip_col, self.strike_col)

        x = np.arange(minx, maxx, grid_size, dtype=np.int64)
        y = np.arange(miny, maxy, grid_size, dtype=np.int64)

        # Extract x and y coordinates once
        x_coords = df["geometry"].apply(lambda point: point.x)
        y_coords = df["geometry"].apply(lambda point: point.y)

        # Group by grid cells and average
        averaged_data = []

        for linex in x:
            for liney in y:
                # Use the extracted coordinates in your filter
                grid = df.loc[
                    (y_coords >= int(liney))
                    & (y_coords <= int(liney + grid_size))
                    & (x_coords >= int(linex))
                    & (x_coords <= int(linex + grid_size))
                ]
                if grid.empty:
                    continue
                else:
                    # Get average coordinates of the filtered data
                    centx = grid["geometry"].apply(lambda point: point.x).mean()
                    centy = grid["geometry"].apply(lambda point: point.y).mean()
                    # centx=linex+(grid_size/2)
                    # centy=liney+(grid_size/2)
                    mean_vector = self.calc_mean_vector(grid)
                    dip = mean_vector[0]
                    strike = mean_vector[1]

                # Create averaged record
                avg_record = {
                    "geometry": Point(centx, centy),
                    "vector": tuple(mean_vector),
                    self.strike_col: strike,
                    self.dip_col: dip,
                    "samples": len(grid),
                }
                averaged_data.append(avg_record)

        result_gdf = gpd.GeoDataFrame(averaged_data, crs=self.gdf.crs)

        columns_to_keep = ["geometry", self.strike_col, self.dip_col, "samples"]
        result_gdf = result_gdf[columns_to_keep]
        return result_gdf.reset_index(drop=True)

    def spherical_statistics_kent(self, grid_size: float = 1000.0) -> gpd.GeoDataFrame:
        """
        Apply spherical statistics using Kent distribution for grid cell averaging.

        Parameters:
        -----------
        grid_size : float
            Size of grid cells in map units (meters)

        Returns:
        --------
        gpd.GeoDataFrame
            Subsampled GeoDataFrame with Kent distribution statistics
        """
        """
        Average measurements within grid cells.
        
        Parameters:
        -----------
        grid_size : float
            Size of grid cells in map units (meters)
            
        Returns:
        --------
        gpd.GeoDataFrame
            Subsampled GeoDataFrame with averaged measurements
        """
        # Create grid
        bounds = self.gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        df = self.calc_vector(self.gdf, self.dip_col, self.strike_col)

        # Extract x and y coordinates once
        x_coords = df["geometry"].apply(lambda point: point.x)
        y_coords = df["geometry"].apply(lambda point: point.y)

        x = np.arange(minx, maxx, grid_size, dtype=np.int64)
        y = np.arange(miny, maxy, grid_size, dtype=np.int64)

        # Group by grid cells and average
        averaged_data = []

        for linex in x:
            for liney in y:
                grid = df.loc[
                    (y_coords >= int(liney))
                    & (y_coords <= int(liney + grid_size))
                    & (x_coords >= int(linex))
                    & (x_coords <= int(linex + grid_size))
                ]
                if grid.empty:
                    continue
                else:
                    centx = grid["geometry"].apply(lambda point: point.x).mean()
                    centy = grid["geometry"].apply(lambda point: point.y).mean()
                    # centx=linex+(grid_size/2)
                    # centy=liney+(grid_size/2)
                    dip2, dipdir2 = self.calc_mean_vector(grid)
                    count, kappa, beta = self.calc_kent(grid)

                # Create averaged record
                avg_record = {
                    "geometry": Point(centx, centy),
                    self.strike_col: dipdir2,
                    self.dip_col: dip2,
                    "kappa": kappa,
                    "beta": beta,
                    "samples": count,
                }
                averaged_data.append(avg_record)

        result_gdf = gpd.GeoDataFrame(averaged_data, crs=self.gdf.crs)

        columns_to_keep = [
            "geometry",
            self.strike_col,
            self.dip_col,
            "kappa",
            "beta",
            "samples",
        ]
        result_gdf = result_gdf[columns_to_keep]
        return result_gdf.reset_index(drop=True)

    def calc_kent(self, df):

        # Handle the vector column - it might be nested differently
        if "vector" in df.columns:
            vector_series = df["vector"]

            # Convert to list of vectors
            if isinstance(vector_series, pd.Series):
                vectors_list = vector_series.tolist()
            else:
                # If it somehow became a DataFrame
                vectors_list = vector_series.iloc[:, 0].tolist()
        else:
            raise ValueError("No 'vector' column found in DataFrame")

        df["l"] = [v[0] for v in vectors_list]
        df["m"] = [v[1] for v in vectors_list]
        df["n"] = [v[2] for v in vectors_list]

        d = df["l"].sum()
        e = df["m"].sum()
        f = df["n"].sum()
        count = len(df.index)
        # print(count)

        # resultant vector
        g = np.sqrt(float(d**2) + (e**2) + (f**2))

        # direction cosines (Leong and Carlile 1998)
        d2 = d / g
        e2 = e / g
        f2 = f / g
        # g2=sqrt(float(d2**2)+(e2**2)+(f2**2))

        # polar coordinates
        theta = acos(f2)
        if d2 == 0.0:
            phi = 0.0
        else:
            phi = np.arctan(e2 / d2)

        # compute kent distribution
        H = np.array(
            [
                [
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    -np.sin(theta),
                ],
                [-np.sin(phi), np.cos(phi), 0],
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta),
                ],
            ]
        )
        T = np.array(
            [
                [sum(df["l"] ** 2), sum(df["l"] * df["m"]), sum(df["l"] * df["n"])],
                [sum(df["l"] * df["m"]), sum(df["m"] ** 2), sum(df["m"] * df["n"])],
                [sum(df["l"] * df["n"]), sum(df["m"] * df["n"]), sum(df["n"] ** 2)],
            ]
        )
        B = H.transpose() * (T / count) * H
        sword = 0.5 * np.arctan((2 * B[0, 1]) / (B[0, 1] - B[1, 1]))

        # rotation matrix
        S = np.array(
            [
                [np.cos(sword), -np.sin(sword), 0],
                [np.sin(sword), np.cos(sword), 0],
                [0, 0, 1],
            ]
        )
        G = H * S
        V = G.transpose() * (T / count) * G
        Q = V[0, 0] - V[1, 1]

        R = g
        kappa = (1 / (2 - (2 * R) - Q)) + (1 / (2 - (2 * R) + Q))
        beta = 0.5 * ((1 / (2 - (2 * R) - Q)) - (1 / (2 - (2 * R) + Q)))

        return (count, kappa, beta)

    def outlier_removal(
        self, grid_size: float = 1000.0, outlier_threshold: float = 2.0
    ) -> gpd.GeoDataFrame:
        """
        Remove outliers using Kent distribution analysis within grid cells.

        Parameters:
        -----------
        grid_size : float
            Size of grid cells in map units (meters)
        outlier_threshold : float
            Threshold for outlier detection (higher = more permissive)

        Returns:
        --------
        gpd.GeoDataFrame
            Subsampled GeoDataFrame with outliers removed
        """
        # Create grid
        bounds = self.gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        df = self.calc_vector(self.gdf, self.dip_col, self.strike_col)

        # Extract x and y coordinates once
        x_coords = df["geometry"].apply(lambda point: point.x)
        y_coords = df["geometry"].apply(lambda point: point.y)

        x = np.arange(minx, maxx, grid_size, dtype=np.int64)
        y = np.arange(miny, maxy, grid_size, dtype=np.int64)

        # Group by grid cells and average
        averaged_data = []

        for linex in x:
            for liney in y:
                grid = df.loc[
                    (y_coords >= int(liney))
                    & (y_coords <= int(liney + grid_size))
                    & (x_coords >= int(linex))
                    & (x_coords <= int(linex + grid_size))
                ]
                # centx=linex+(grid_size/2)
                # centy=liney+(grid_size/2)
                if grid.empty:
                    continue
                elif len(grid) < 3:

                    centx = grid["geometry"].apply(lambda point: point.x).mean()
                    centy = grid["geometry"].apply(lambda point: point.y).mean()
                    dip_final, dipdir_final = self.calc_mean_vector(grid)
                    count, kappa, beta = self.calc_kent(grid)
                    removed_idx = -1

                else:
                    _, k0, _ = self.calc_kent(grid)

                    delta_kappas = []
                    for idx in grid.index:
                        temp = grid.drop(index=idx)
                        # dip_temp, dipdir_temp = self.calc_mean_vector(temp)
                        _, k_temp, _ = self.calc_kent(temp)
                        delta_kappas.append((idx, k_temp - k0))

                    removed_idx, max_delta = max(delta_kappas, key=lambda x: x[1])
                    final_grid = grid.drop(index=removed_idx)
                    centx = final_grid["geometry"].apply(lambda point: point.x).mean()
                    centy = final_grid["geometry"].apply(lambda point: point.y).mean()

                    # centx=linex+(grid_size/2)
                    # centy=liney+(grid_size/2)

                    dip_final, dipdir_final = self.calc_mean_vector(final_grid)
                    count, kappa, beta = self.calc_kent(final_grid)

                # Create averaged record
                avg_record = {
                    "geometry": Point(centx, centy),
                    self.strike_col: dipdir_final,
                    self.dip_col: dip_final,
                    "kappa": kappa,
                    "beta": beta,
                    "samples": count,
                    "removed_id": removed_idx,
                }
                averaged_data.append(avg_record)

        result_gdf = gpd.GeoDataFrame(averaged_data, crs=self.gdf.crs)

        columns_to_keep = [
            "geometry",
            self.strike_col,
            self.dip_col,
            "kappa",
            "beta",
            "samples",
            "removed_id",
        ]
        result_gdf = result_gdf[columns_to_keep]
        return result_gdf.reset_index(drop=True)

    def firstOrderSubsampling(
        self,
        contact_gdf: gpd.GeoDataFrame,
        distance_buffer: float = 500.0,
        angle_tolerance: float = 15.0,
    ) -> gpd.GeoDataFrame:
        pass

    def _calculate_line_strike(self, line_geom, point_geom) -> float:
        """Calculate strike of line geometry at nearest point."""
        # Get coordinates
        if hasattr(line_geom, "coords"):
            coords = list(line_geom.coords)
        else:
            coords = list(line_geom.exterior.coords)

        if len(coords) < 2:
            return 0.0

        # Find nearest segment
        min_dist = float("inf")
        nearest_segment = None

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]

            # Calculate distance from point to segment
            dist = Point(point_geom).distance(Point(p1))
            if dist < min_dist:
                min_dist = dist
                nearest_segment = (p1, p2)

        if nearest_segment:
            p1, p2 = nearest_segment
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            # Calculate strike (azimuth)
            strike = degrees(atan2(dx, dy))
            if strike < 0:
                strike += 360

            return strike

        return 0.0

    def get_subsample_summary(self, subsampled_gdf: gpd.GeoDataFrame) -> dict:
        """
        Get summary statistics for subsampled data.

        Parameters:
        -----------
        subsampled_gdf : gpd.GeoDataFrame
            Subsampled GeoDataFrame

        Returns:
        --------
        dict
            Summary statistics
        """
        original_count = len(self.gdf)
        subsampled_count = len(subsampled_gdf)

        summary = {
            "original_count": original_count,
            "subsampled_count": subsampled_count,
            "retention_percentage": (subsampled_count / original_count) * 100,
            "reduction_factor": (
                original_count / subsampled_count if subsampled_count > 0 else np.inf
            ),
        }

        return summary

"""
Structural Subsampling Engine — Orientation Data Spatial Reduction

Overview
--------
Reduces spatial clustering in bedding / foliation point datasets while
preserving the statistical character of the orientations.  Designed for
GSWA-style structural datasets where the raw shapefile may contain mixed
feature types (foliation, joints, etc.) and non-standard column names.

The main entry point is  run_subsampling_file().  It runs a two-stage pipeline:

  Stage 0 — Column standardisation and bedding filter
      1. Filters the raw shapefile to bedding-only records (optional).
      2. Renames source columns to the four standard names:
             DIP      – dip angle in degrees (0 = horizontal, 90 = vertical)
             DIP_DIR  – dip direction in degrees (0–360, clockwise from north)
             EASTING  – projected easting coordinate (metres)
             NORTHING – projected northing coordinate (metres)
      3. Converts strike to dip-direction if the source records strike.
      4. Writes the processed shapefile to  input_beddings_only/  so all
         subsequent methods read from one clean, consistently-named file.

  Stage 1 — Subsampling
      Applies the chosen algorithm(s) to the standardised bedding file.
      Six algorithms are available; they can be run individually or in
      combination by passing a list of method names to run_subsampling_file().

Seven subsampling algorithms
-----------------------------
1. decimation            – retain every nth point by digitisation order
2. stochastic            – random-fraction sampling with a fixed seed
3. gridcell_average      – grid-cell mean-vector averaging
4. spherical_kent        – grid-cell Kent distribution statistics (κ, β)
5. outlier_removal       – kappa-based single-outlier removal per cell
5b. outlier_carmichael   – iterative Δκ-break algorithm (Carmichael 2016)
6. firstorder            – proximity + angular alignment to contacts

Key parameters
--------------
bedding_file  : standardised bedding point shapefile path
geology_file  : geology polygon shapefile (used by Method 6 for contacts)
output_dir    : directory where subsampled shapefiles and CSVs are written
methods       : list of algorithm names to run (or 'all' to run every method)
grid_n        : grid cell size in metres for Methods 3–5b (default 1000 m)
decimation_n  : step size for Method 1 (default 5)
stoch_frac    : sampling fraction for Method 2 (default 0.5)
dist_buffer   : contact proximity buffer for Method 6 (default 500 m)
angle_tol     : strike alignment tolerance for Method 6 (default 15°)

References
----------
Carmichael, T. & Ailleres, L. (2016). Method and analysis for the upscaling
  of structural data. Journal of Structural Geology, 83, 121–133.
Leong, L.S. & Carlile, J.C. (1998). A method for estimating the Kent
  distribution parameters for orientation data. Mathematical Geology.
"""

import os
import math
from math import sin, cos, asin, atan, atan2, degrees, radians, sqrt, acos, pi

import geopandas as gpd
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from shapely.geometry import Point, LineString
from shapely.ops import unary_union


# =============================================================================
# SUBSAMPLING ENGINE
# Encapsulates all seven subsampling algorithms together with their shared
# statistical machinery.  Instantiate one engine per run; the engine is
# stateless between method calls so the same instance can be reused for
# multiple datasets or parameter sweeps.
# =============================================================================

class SubsamplingEngine:
    """
    Main engine for structural orientation data subsampling.

    Wraps all seven subsampling algorithms and their shared statistical helpers
    (direction-cosine conversion, mean-orientation, Kent distribution) in a
    single class.  Instantiate once and call  subsample()  with the chosen
    method name, or call individual method functions directly.

    Parameters
    ----------
    dip      : Standard column name for dip angle.   Default ``'DIP'``.
    dipdir   : Standard column name for dip direction.  Default ``'DIP_DIR'``.
    easting  : Standard column name for easting.   Default ``'EASTING'``.
    northing : Standard column name for northing.  Default ``'NORTHING'``.
    """

    def __init__(self,
                 dip      = 'DIP',
                 dipdir   = 'DIP_DIR',
                 easting  = 'EASTING',
                 northing = 'NORTHING'):
        self.dip      = dip
        self.dipdir   = dipdir
        self.easting  = easting
        self.northing = northing

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def _bounds(gdf):
        """Return integer bounding-box tuple (min_x, max_x, min_y, max_y)."""
        b = gdf.total_bounds      # [min_x, min_y, max_x, max_y]
        return int(b[0]), int(b[2]) + 1, int(b[1]), int(b[3]) + 1

    # =========================================================================
    # STATISTICAL HELPERS
    # Private methods used internally by the grid-cell and outlier algorithms.
    # =========================================================================

    def _add_direction_cosines(self, gpdataframe):
        """
        Append a ``vector`` column of direction-cosine tuples to a DataFrame.

        Each row receives a ``(l, m, n)`` tuple derived from its dip and
        dip-direction values.  The computation is fully vectorised using NumPy.

        Parameters
        ----------
        gpdataframe : Input GeoDataFrame or DataFrame.

        Returns
        -------
        DataFrame with an additional column ``vector`` containing
        ``(l, m, n)`` tuples.
        """
        df = pd.DataFrame(gpdataframe)
        dip_r    = np.radians(df[self.dip].astype(float))
        dipdir_r = np.radians(df[self.dipdir].astype(float))
        # Vectorised: pole pointing downward into the plane
        l_arr = np.sin(dip_r) * np.sin(dipdir_r)   # east
        m_arr = np.sin(dip_r) * np.cos(dipdir_r)   # north
        n_arr = np.cos(dip_r)                        # vertical (down)
        df['vector'] = list(zip(l_arr, m_arr, n_arr))
        return df

    def _calc_mean_orientation(self, df):
        """
        Compute the mean orientation of a group of structural measurements.

        The mean is the normalised sum of the individual unit vectors
        (direction cosines), converted back to dip / dip-direction.

        Parameters
        ----------
        df : DataFrame that **must** already contain a ``vector`` column of
             ``(l, m, n)`` tuples (added by :meth:`_add_direction_cosines`).

        Returns
        -------
        Tuple ``(dip, dipdir)`` in degrees, or ``(nan, nan)`` if the resultant
        vector length is near zero (antipodal vectors cancel).
        """
        df = pd.DataFrame(df)
        df[['l', 'm', 'n']] = pd.DataFrame(df['vector'].tolist(), index=df.index)

        l_sum, m_sum, n_sum = df['l'].sum(), df['m'].sum(), df['n'].sum()
        count = len(df.index)

        r_bar = sqrt(float((l_sum / count)**2 + (m_sum / count)**2 +
                           (n_sum / count)**2))

        if r_bar < 1e-12:
            return (float('nan'), float('nan'))

        l_unit = (l_sum / count) / r_bar
        m_unit = (m_sum / count) / r_bar
        n_unit = (n_sum / count) / r_bar
        return dircos2ddd(l_unit, m_unit, n_unit)

    def _calc_kent(self, gpdataframe):
        """
        Estimate Kent distribution parameters for a cell of measurements.

        The Kent (5-parameter Fisher–Bingham) distribution on the sphere is
        characterised by:

        - **kappa (κ)** : concentration parameter (higher = tighter cluster)
        - **beta (β)**  : ovalness / anisotropy parameter (0 = symmetric)

        The estimation follows Leong & Carlile (1998) using the orientation
        matrix T and rotation matrices H, S, G.

        Parameters
        ----------
        gpdataframe : DataFrame with a ``vector`` column of ``(l, m, n)`` tuples.

        Returns
        -------
        Tuple ``(count, kappa, beta)``.

        Notes
        -----
        Two known transcription errors in both published sources are corrected:

        *H-matrix fix* — position [2, 1] must be  sin θ sin φ  (not
        sin θ cos φ as printed in Leong & Carlile 1998 and Carmichael &
        Ailleres 2016).

        *Sword-angle fix* — denominator must be  B[0,0] − B[1,1]  (diagonal
        difference), not  B[0,1] − B[1,1]  (off-diagonal minus diagonal)
        as printed in both sources.
        """
        df = pd.DataFrame(gpdataframe)
        df[['l', 'm', 'n']] = pd.DataFrame(df['vector'].tolist(), index=df.index)

        l_sum, m_sum, n_sum = df['l'].sum(), df['m'].sum(), df['n'].sum()
        count = len(df.index)

        r = sqrt(float(l_sum**2 + m_sum**2 + n_sum**2))
        if r < 1e-12:
            return (count, float('nan'), float('nan'))

        l_mean, m_mean, n_mean = l_sum / r, m_sum / r, n_sum / r

        # --- Rotation matrix H (align z-axis with the mean direction) ---
        theta = acos(max(-1.0, min(1.0, n_mean)))
        phi   = (atan(m_mean / l_mean) if abs(l_mean) > 1e-12
                 else (pi / 2 if m_mean >= 0 else -pi / 2))

        # NOTE: H[2,1] = sin θ sin φ  (transcription error in both published sources)
        H = np.array([
            [cos(theta) * cos(phi),  cos(theta) * sin(phi), -sin(theta)],
            [-sin(phi),               cos(phi),               0.0       ],
            [sin(theta) * cos(phi),   sin(theta) * sin(phi),  cos(theta)],
        ])

        # --- Orientation matrix T (second-moment matrix of direction cosines) ---
        T = np.array([
            [sum(df['l']**2),        sum(df['l'] * df['m']), sum(df['l'] * df['n'])],
            [sum(df['l'] * df['m']), sum(df['m']**2),        sum(df['m'] * df['n'])],
            [sum(df['l'] * df['n']), sum(df['m'] * df['n']), sum(df['n']**2)       ],
        ])

        # B = H^T (T/N) H
        B = H.T @ (T / count) @ H

        # --- Sword angle S ---
        # NOTE: denominator = B[0,0] − B[1,1]  (transcription error in both sources)
        b_diag_diff = B[0, 0] - B[1, 1]
        sword = (0.5 * atan((2 * B[0, 1]) / b_diag_diff)
                 if abs(b_diag_diff) > 1e-12 else 0.0)

        S = np.array([
            [cos(sword), -sin(sword), 0.0],
            [sin(sword),  cos(sword), 0.0],
            [0.0,         0.0,        1.0],
        ])

        G = H @ S
        V = G.T @ (T / count) @ G
        Q = V[0, 0] - V[1, 1]
        R = r

        # --- Leong & Carlile (1998) moment estimator for κ and β ---
        denom1 = 2 - 2 * R - Q
        denom2 = 2 - 2 * R + Q
        kappa_raw = (
            (1 / denom1 if abs(denom1) > 1e-12 else float('inf')) +
            (1 / denom2 if abs(denom2) > 1e-12 else float('inf'))
        )
        beta_raw = 0.5 * (
            (1 / denom1 if abs(denom1) > 1e-12 else float('inf')) -
            (1 / denom2 if abs(denom2) > 1e-12 else float('inf'))
        )

        # Enforce Kent validity constraints: κ > 0, 0 ≤ 2β < κ
        kappa = max(1e-6, kappa_raw)
        beta  = max(0.0, min(beta_raw, kappa / 2.0 - 1e-9))
        return (count, kappa, beta)

    # =========================================================================
    # METHOD 1: DECIMATION
    # Retain every nth point by digitisation order.
    # =========================================================================

    def decimation(self, gpdataframe, n, path_out='outputs/'):
        """
        Retain every nth point by digitisation order.

        The simplest possible subsample: slice the dataset with a fixed step
        so that evenly-spaced records are kept.

        Parameters
        ----------
        gpdataframe : Input GeoDataFrame of structural measurements.
        n           : Step size — keep every nth record (e.g. ``n=5`` keeps
                      records 0, 5, 10, …).
        path_out    : Output directory path.

        Returns
        -------
        GeoDataFrame of retained measurements with a point geometry column.
        """
        df = pd.DataFrame(gpdataframe)
        df_sub = df.iloc[::n].copy()
        df_sub['geometry'] = df_sub.apply(
            lambda row: Point(float(row[self.easting]),
                              float(row[self.northing])), axis=1)
        df_sub = gpd.GeoDataFrame(df_sub, geometry='geometry')
        if df_sub.empty:
            print("DataFrame Empty")
        else:
            df_sub.to_file(
                os.path.join(path_out, f"structure_file_decimation_{n}.shp"),
                driver='ESRI Shapefile')
            df_sub.to_csv(
                os.path.join(path_out, f"structure_file_decimation_{n}.csv"))
        return df_sub

    # =========================================================================
    # METHOD 2: STOCHASTIC SUBSAMPLING
    # Randomly sample a fraction of the input dataset.
    # =========================================================================

    def stochastic(self, gpdataframe, frac=0.5, replace=False,
                   random_state=42, path_out='outputs/'):
        """
        Randomly subsample a fraction of the input dataset.

        Applies ``DataFrame.sample`` with the specified fraction and random
        seed, giving a reproducible random subset.

        Parameters
        ----------
        gpdataframe  : Input GeoDataFrame of structural measurements.
        frac         : Fraction of rows to retain, in the range (0, 1].
                       Default: 0.5 (50 %).
        replace      : Whether to sample with replacement.  Default: ``False``.
        random_state : Integer seed for reproducibility.  Default: 42.
        path_out     : Output directory path.

        Returns
        -------
        GeoDataFrame of sampled measurements with a point geometry column.
        """
        df = pd.DataFrame(gpdataframe)
        df_sub = df.sample(frac=frac, replace=replace, random_state=random_state)
        df_sub['geometry'] = df_sub.apply(
            lambda row: Point(float(row[self.easting]),
                              float(row[self.northing])), axis=1)
        df_sub = gpd.GeoDataFrame(df_sub, geometry='geometry')
        if df_sub.empty:
            print("DataFrame Empty")
        else:
            df_sub.to_file(
                os.path.join(path_out, f"structure_file_stochastic_{frac}.shp"),
                driver='ESRI Shapefile')
            df_sub.to_csv(
                os.path.join(path_out, f"structure_file_stochastic_{frac}.csv"))
        return df_sub

    # =========================================================================
    # METHOD 3: GRID-CELL AVERAGING
    # Compute the mean orientation vector for measurements within each cell.
    # =========================================================================

    def gridcell_average(self, gpdataframe, minx, maxx, miny, maxy,
                         n=1000, path_out='outputs/'):
        """
        Compute the mean orientation vector for measurements within each grid cell.

        The study area is divided into a regular grid of n × n metre cells.
        All measurements that fall within a cell are combined into a single
        representative orientation using the normalised vector mean.  Cells
        with no measurements are omitted from the output.

        Parameters
        ----------
        gpdataframe : Input GeoDataFrame of structural measurements.
        minx, maxx  : Western and eastern study-area boundaries (metres).
        miny, maxy  : Southern and northern study-area boundaries (metres).
        n           : Grid cell size in metres.  Default: 1000 m.
        path_out    : Output directory path.

        Returns
        -------
        Filename stem (str) of the CSV written to ``path_out``.  Pass this to
        :func:`save_grid_to_shapefile` to produce a point shapefile.
        """
        df = self._add_direction_cosines(gpdataframe)

        x = np.arange(minx, maxx, n, dtype=np.int64)
        y = np.arange(miny, maxy, n, dtype=np.int64)

        file = f"structure_file_gridcell_{n}"
        fieldnames = ['EASTING', 'NORTHING', 'DIP', 'DIP_DIR']

        with open(os.path.join(path_out, file + ".csv"), "w",
                  encoding="utf-8") as out:
            out.write(','.join(fieldnames) + '\n')
            for linex in x:
                for liney in y:
                    grid = df.loc[
                        (df[self.northing] >= int(liney)) &
                        (df[self.northing] <= int(liney + n)) &
                        (df[self.easting]  >= int(linex)) &
                        (df[self.easting]  <= int(linex + n))
                    ]
                    if grid.empty:
                        continue
                    centx = linex + (n / 2)
                    centy = liney + (n / 2)
                    dip_val, dipdir_val = self._calc_mean_orientation(grid)
                    dip_str    = ('NaN' if math.isnan(dip_val)
                                  else str(int(dip_val)))
                    dipdir_str = ('NaN' if math.isnan(dipdir_val)
                                  else str(int(dipdir_val)))
                    out.write(f"{int(centx)},{int(centy)},"
                              f"{dip_str},{dipdir_str}\n")
        return file

    # =========================================================================
    # METHOD 4: SPHERICAL STATISTICS (KENT DISTRIBUTION)
    # Compute Kent distribution parameters (mean, κ, β) per grid cell.
    # =========================================================================

    def spherical_kent(self, gpdataframe, minx, maxx, miny, maxy,
                       n=1000, path_out='outputs/'):
        """
        Compute Kent distribution statistics (mean orientation, κ, β) per cell.

        In addition to the mean orientation provided by :meth:`gridcell_average`,
        this method estimates the Kent distribution parameters:

        - **kappa (κ)**: concentration — larger values = tighter clustering.
        - **beta (β)**:  ovalness — zero means rotationally symmetric.

        Empty cells are omitted from the output.

        Parameters
        ----------
        gpdataframe : Input GeoDataFrame of structural measurements.
        minx, maxx  : Western and eastern study-area boundaries (metres).
        miny, maxy  : Southern and northern study-area boundaries (metres).
        n           : Grid cell size in metres.  Default: 1000 m.
        path_out    : Output directory path.

        Returns
        -------
        Filename stem (str) of the CSV written to ``path_out``.
        """
        df = self._add_direction_cosines(gpdataframe)

        x = np.arange(minx, maxx, n, dtype=np.int64)
        y = np.arange(miny, maxy, n, dtype=np.int64)

        file = f"structure_file_spherical_{n}"
        fieldnames = ['EASTING', 'NORTHING', 'DIP', 'DIP_DIR',
                      'count', 'kappa', 'beta']

        with open(os.path.join(path_out, file + ".csv"), "w",
                  encoding="utf-8") as out:
            out.write(','.join(fieldnames) + '\n')
            for linex in x:
                for liney in y:
                    grid = df.loc[
                        (df[self.northing] >= int(liney)) &
                        (df[self.northing] <= int(liney + n)) &
                        (df[self.easting]  >= int(linex)) &
                        (df[self.easting]  <= int(linex + n))
                    ]
                    if grid.empty:
                        continue
                    centx = linex + (n / 2)
                    centy = liney + (n / 2)
                    dip2, dipdir2 = self._calc_mean_orientation(grid)
                    count, kappa, beta = self._calc_kent(grid)
                    dip2_str = (
                        '-999' if (isinstance(dip2, float) and math.isnan(dip2))
                        else str(int(dip2))
                    )
                    dipdir2_str = (
                        '-999' if (isinstance(dipdir2, float) and
                                   math.isnan(dipdir2))
                        else str(int(dipdir2))
                    )
                    out.write(f"{int(centx)},{int(centy)},"
                              f"{dip2_str},{dipdir2_str},"
                              f"{int(count)},{kappa},{beta}\n")
        return file

    # =========================================================================
    # METHOD 5: OUTLIER REMOVAL (SINGLE-REMOVAL PER CELL)
    # Remove the one measurement per cell whose removal most increases kappa.
    # =========================================================================

    def outlier_removal(self, gdf, minx, maxx, miny, maxy,
                        n=1000, path_out='outputs/'):
        """
        Remove the single greatest outlier per grid cell using the max-Δκ criterion.

        For each grid cell containing more than 3 measurements, every point is
        tentatively removed in turn.  The point whose removal produces the
        greatest increase in kappa is identified as the outlier and permanently
        excluded.  Cells with ≤ 3 measurements are excluded from the output.

        Parameters
        ----------
        gdf         : Input GeoDataFrame of structural measurements.
        minx, maxx  : Western and eastern study-area boundaries (metres).
        miny, maxy  : Southern and northern study-area boundaries (metres).
        n           : Grid cell size in metres.  Default: 1000 m.
        path_out    : Output directory path.

        Returns
        -------
        Filename stem (str) of the cleaned CSV written to ``path_out``.
        """
        df = self._add_direction_cosines(gdf)

        x_coords = np.arange(minx, maxx, n)
        y_coords = np.arange(miny, maxy, n)
        file     = f"structure_file_outlier_{n}"
        csv_path = os.path.join(path_out, f"{file}.csv")

        with open(csv_path, "w", encoding="utf-8") as out:
            out.write("EASTING,NORTHING,DIP,DIP_DIR,"
                      "count,kappa,beta,REMOVED_INDEX\n")

            for linex in x_coords:
                for liney in y_coords:
                    grid = df[
                        (df[self.northing] >= liney) &
                        (df[self.northing] <= liney + n) &
                        (df[self.easting]  >= linex) &
                        (df[self.easting]  <= linex + n)
                    ]
                    centx = int(linex + n / 2)
                    centy = int(liney + n / 2)

                    if len(grid) <= 3:
                        out.write(f"{centx},{centy},-999,-999,"
                                  f"{len(grid)},-999,-999,-1\n")
                        continue

                    dip0, dipdir0 = self._calc_mean_orientation(grid)
                    _, k0, _ = self._calc_kent(grid)

                    delta_kappas = []
                    for idx in grid.index:
                        temp = grid.drop(index=idx)
                        _, k_t, _ = self._calc_kent(temp)
                        delta_kappas.append((idx, k_t - k0))

                    removed_idx, _ = max(delta_kappas, key=lambda x: x[1])
                    final_grid = grid.drop(index=removed_idx)
                    dip_f, dipdir_f = self._calc_mean_orientation(final_grid)
                    count, kappa, beta = self._calc_kent(final_grid)
                    out.write(f"{centx},{centy},{int(dip_f)},{int(dipdir_f)},"
                              f"{count},{kappa},{beta},{removed_idx}\n")

        # Strip sentinel rows — final CSV contains only valid cells
        df_clean = pd.read_csv(csv_path)
        df_clean = df_clean[~(
            (df_clean['DIP']   == -999) &
            (df_clean['DIP_DIR'] == -999) &
            (np.isclose(df_clean['kappa'].astype(float), -999)) &
            (np.isclose(df_clean['beta'].astype(float),  -999))
        )]
        df_clean.to_csv(csv_path, index=False)
        return file

    # =========================================================================
    # METHOD 5b: CARMICHAEL ITERATIVE OUTLIER REMOVAL
    # Remove multiple outliers per cell using the Δκ-break algorithm.
    # =========================================================================

    def outlier_carmichael(self, gdf, minx, maxx, miny, maxy,
                           n=1000, path_out='outputs/'):
        """
        Remove multiple outliers per grid cell using the Carmichael (2016)
        iterative Δκ-break algorithm.

        For each grid cell, points are ranked by angular distance from the cell
        mean direction (farthest first).  They are then removed iteratively and
        κ is recomputed after each removal.  The step with the largest increase
        in κ defines the outlier boundary: every point removed up to and
        including that step is treated as an outlier.  At most N // 2 points
        may be removed from any cell of N measurements.

        If no removal step produces a positive Δκ, the cell is left unchanged.

        Parameters
        ----------
        gdf         : Input GeoDataFrame of structural measurements.
        minx, maxx  : Western and eastern study-area boundaries (metres).
        miny, maxy  : Southern and northern study-area boundaries (metres).
        n           : Grid cell size in metres.  Default: 1000 m.
        path_out    : Output directory path.

        Returns
        -------
        Filename stem (str) of the cleaned CSV written to ``path_out``.
        """
        df = self._add_direction_cosines(gdf)
        df[['dc_l', 'dc_m', 'dc_n']] = pd.DataFrame(
            df['vector'].tolist(), index=df.index)

        x_coords = np.arange(minx, maxx, n)
        y_coords = np.arange(miny, maxy, n)
        file     = f"structure_file_outlier_carmichael_{n}"
        csv_path = os.path.join(path_out, f"{file}.csv")

        with open(csv_path, "w", encoding="utf-8") as out:
            out.write("EASTING,NORTHING,DIP,DIP_DIR,"
                      "count,kappa,beta,n_removed\n")

            for linex in x_coords:
                for liney in y_coords:
                    grid = df[
                        (df[self.northing] >= liney) &
                        (df[self.northing] <= liney + n) &
                        (df[self.easting]  >= linex) &
                        (df[self.easting]  <= linex + n)
                    ]
                    centx = int(linex + n / 2)
                    centy = int(liney + n / 2)
                    N = len(grid)

                    if N <= 3:
                        out.write(f"{centx},{centy},-999,-999,"
                                  f"{N},-999,-999,-1\n")
                        continue

                    # Compute cell mean direction as a unit vector
                    l_s = float(grid['dc_l'].sum())
                    m_s = float(grid['dc_m'].sum())
                    n_s = float(grid['dc_n'].sum())
                    g_len = sqrt(l_s**2 + m_s**2 + n_s**2)
                    if g_len < 1e-12:
                        out.write(f"{centx},{centy},-999,-999,"
                                  f"{N},-999,-999,-1\n")
                        continue
                    mean_vec = np.array([l_s / g_len, m_s / g_len,
                                         n_s / g_len])

                    # Rank points by angular distance from mean (farthest first)
                    vecs = grid[['dc_l', 'dc_m', 'dc_n']].values.astype(float)
                    dots = np.clip(vecs @ mean_vec, -1.0, 1.0)
                    ang_dists = np.degrees(np.arccos(dots))
                    ranked_indices = list(
                        grid.index[np.argsort(ang_dists)[::-1]])

                    # Build κ sequence: remove one point at a time
                    _, k_prev, _ = self._calc_kent(grid)
                    max_remove = N // 2
                    delta_seq  = []
                    remaining  = list(grid.index)

                    for i in range(max_remove):
                        remaining.remove(ranked_indices[i])
                        temp = grid.loc[remaining]
                        _, k_t, _ = self._calc_kent(temp)
                        delta_seq.append(k_t - k_prev)
                        k_prev = k_t

                    # Find the Δκ break
                    best_i     = int(np.argmax(delta_seq))
                    best_delta = delta_seq[best_i]

                    if best_delta <= 0:
                        # No removal helps — retain all points
                        n_removed  = 0
                        final_grid = grid
                    else:
                        n_removed   = best_i + 1
                        outlier_set = set(ranked_indices[:n_removed])
                        kept        = [idx for idx in grid.index
                                       if idx not in outlier_set]
                        final_grid  = grid.loc[kept]

                    dip_f, dipdir_f = self._calc_mean_orientation(final_grid)
                    if math.isnan(dip_f) or math.isnan(dipdir_f):
                        out.write(f"{centx},{centy},-999,-999,"
                                  f"{N - n_removed},-999,-999,{n_removed}\n")
                        continue
                    count, kappa_f, beta_f = self._calc_kent(final_grid)
                    out.write(f"{centx},{centy},{int(dip_f)},{int(dipdir_f)},"
                              f"{count},{kappa_f},{beta_f},{n_removed}\n")

        # Strip sentinel rows
        df_clean = pd.read_csv(csv_path)
        df_clean = df_clean[~(
            (df_clean['DIP']   == -999) &
            (df_clean['DIP_DIR'] == -999) &
            (np.isclose(df_clean['kappa'].astype(float), -999)) &
            (np.isclose(df_clean['beta'].astype(float),  -999))
        )]
        df_clean.to_csv(csv_path, index=False)
        return file

    # =========================================================================
    # METHOD 6: FIRST-ORDER SUBSAMPLING
    # Filter by proximity to contacts and angular alignment with them.
    # =========================================================================

    def firstorder(self, gpdataframe, contact_gdf,
                   dist_buffer=500, angle_tol=15, path_out='outputs/'):
        """
        Filter measurements by proximity to stratigraphic contacts and angular
        alignment with them (first-order subsampling).

        A measurement is retained only if **both** criteria are satisfied:

        (a) **Proximity** — its distance to the nearest contact line is within
            ``dist_buffer`` metres.
        (b) **Angular alignment** — the bedding strike differs from the contact
            azimuth by no more than ``angle_tol`` degrees.

        Parameters
        ----------
        gpdataframe : Input GeoDataFrame with point geometries.
        contact_gdf : GeoDataFrame of stratigraphic contact lines
                      (LineString or MultiLineString geometries).
        dist_buffer : Maximum distance from any contact (metres).
                      Default: 500 m.
        angle_tol   : Maximum allowable strike angular difference (degrees).
                      Default: 15°.
        path_out    : Output directory path.

        Returns
        -------
        GeoDataFrame of retained measurements passing both filters.
        """
        gdf = gpd.GeoDataFrame(gpdataframe.copy())
        if gdf.crs is not None and contact_gdf.crs is not None:
            contact_gdf = contact_gdf.to_crs(gdf.crs)

        contact_union = unary_union(contact_gdf.geometry)

        # Step 1: Distance filter
        dists      = gdf.geometry.distance(contact_union)
        candidates = gdf[dists <= dist_buffer].copy()
        if candidates.empty:
            print("DataFrame Empty — no measurements within the distance buffer.")
            return candidates

        # Step 2: Decompose contacts into per-segment azimuths
        seg_gdf = _segment_azimuths(contact_gdf)

        # Step 3: Nearest-segment join to get contact azimuth per point
        cands_reset = candidates.reset_index(drop=False)

        try:
            joined = gpd.sjoin_nearest(
                cands_reset[['index', 'geometry', self.dipdir]],
                seg_gdf[['geometry', 'azimuth']],
                how='left'
            )
            joined = joined.drop_duplicates(subset='index')

            contact_az     = joined['azimuth'].values
            bedding_strike = (joined[self.dipdir].astype(float).values - 90) % 180
            diff = np.abs(((contact_az - bedding_strike + 90) % 180) - 90)
            keep_orig_idx = joined.loc[diff <= angle_tol, 'index'].values
            df_sub = candidates.loc[keep_orig_idx].copy()

        except AttributeError:
            # Fallback for GeoPandas < 0.10 without sjoin_nearest
            seg_geoms  = seg_gdf.geometry.values
            seg_az_arr = seg_gdf['azimuth'].values
            contact_az = np.array([
                seg_az_arr[np.argmin([pt.distance(s) for s in seg_geoms])]
                for pt in candidates.geometry
            ])
            bedding_strike = (candidates[self.dipdir].astype(float).values
                              - 90) % 180
            diff   = np.abs(((contact_az - bedding_strike + 90) % 180) - 90)
            df_sub = candidates[diff <= angle_tol].copy()

            # Write outputs
            """        if df_sub.empty:
                        print("DataFrame Empty — no measurements pass the angle filter.")
                    else:
                        fname = (f"structure_file_firstorder"
                                f"_d{int(dist_buffer)}_a{int(angle_tol)}")
                        df_sub.to_file(os.path.join(path_out, fname + ".shp"),
                                    driver='ESRI Shapefile')
                        df_sub.to_csv(os.path.join(path_out, fname + ".csv"))
            """        
        return df_sub

    # =========================================================================
    # DISPATCH
    # Unified interface: call any method by name.
    # =========================================================================

    def subsample(self, method, gpdataframe, path_out='outputs/', **kwargs):
        """
        Run a named subsampling algorithm on the supplied dataset.

        Parameters
        ----------
        method      : Algorithm name.  One of:
                      ``'decimation'``, ``'stochastic'``,
                      ``'gridcell_average'``, ``'spherical_kent'``,
                      ``'outlier_removal'``, ``'outlier_carmichael'``,
                      ``'firstorder'``.
        gpdataframe : Input GeoDataFrame of structural measurements.
        path_out    : Output directory path.
        **kwargs    : Method-specific keyword arguments forwarded to the
                      chosen algorithm (e.g. ``n=5`` for decimation,
                      ``grid_n=1000`` for grid-cell methods).

        Returns
        -------
        The return value of the selected algorithm (GeoDataFrame or filename
        stem, depending on the method).

        Raises
        ------
        ValueError
            If ``method`` is not a recognised algorithm name.
        """
        valid = [
            'decimation', 'stochastic',
            'gridcell_average', 'spherical_kent',
            'outlier_removal', 'outlier_carmichael',
            'firstorder',
        ]
        if method not in valid:
            raise ValueError(
                f"method must be one of: {', '.join(valid)}\n"
                f"  Got: '{method}'"
            )

        # Extract shared grid-cell bounds if supplied
        minx = kwargs.pop('minx', None)
        maxx = kwargs.pop('maxx', None)
        miny = kwargs.pop('miny', None)
        maxy = kwargs.pop('maxy', None)
        grid_n = kwargs.pop('grid_n', 1000)

        if method == 'decimation':
            return self.decimation(
                gpdataframe, n=kwargs.get('n', 5), path_out=path_out)

        if method == 'stochastic':
            return self.stochastic(
                gpdataframe,
                frac=kwargs.get('frac', 0.5),
                replace=kwargs.get('replace', False),
                random_state=kwargs.get('random_state', 42),
                path_out=path_out)

        if method == 'gridcell_average':
            return self.gridcell_average(
                gpdataframe, minx, maxx, miny, maxy,
                n=grid_n, path_out=path_out)

        if method == 'spherical_kent':
            return self.spherical_kent(
                gpdataframe, minx, maxx, miny, maxy,
                n=grid_n, path_out=path_out)

        if method == 'outlier_removal':
            return self.outlier_removal(
                gpdataframe, minx, maxx, miny, maxy,
                n=grid_n, path_out=path_out)

        if method == 'outlier_carmichael':
            return self.outlier_carmichael(
                gpdataframe, minx, maxx, miny, maxy,
                n=grid_n, path_out=path_out)

        if method == 'firstorder':
            return self.firstorder(
                gpdataframe,
                contact_gdf=kwargs['contact_gdf'],
                dist_buffer=kwargs.get('dist_buffer', 500),
                angle_tol=kwargs.get('angle_tol', 15),
                path_out=path_out)


# =============================================================================
# STANDALONE CONVERSION HELPERS
# Pure-math functions with no dependency on engine state.
# =============================================================================

def strike_to_dipdir(gpdataframe, strike):
    """
    Convert a strike column to dip-direction (right-hand rule).

    Dip-direction = strike − 90, wrapped to the range 0–360°.

    Parameters
    ----------
    gpdataframe : GeoDataFrame or DataFrame containing the strike column.
    strike      : Name of the column holding strike values (degrees, 0–360).

    Returns
    -------
    DataFrame with an additional column ``dipdir`` containing dip-direction
    values in degrees (0–360).
    """
    df = pd.DataFrame(gpdataframe)
    df['dipdir'] = (df[strike] - 90) % 360
    return df


def ddd2dircos(dip, dipdir):
    """
    Convert dip and dip-direction to direction cosines.

    The pole to a plane (unit normal pointing downward into the plane)::

        l = sin(dip) × sin(dip_dir)   (east component)
        m = sin(dip) × cos(dip_dir)   (north component)
        n = cos(dip)                   (vertical component)

    Parameters
    ----------
    dip    : Dip angle in degrees (0 = horizontal, 90 = vertical).
    dipdir : Dip-direction in degrees (0–360, clockwise from north).

    Returns
    -------
    Tuple ``(l, m, n)`` of floats.
    """
    l = sin(radians(dip)) * sin(radians(dipdir))
    m = sin(radians(dip)) * cos(radians(dipdir))
    n = cos(radians(dip))
    return (l, m, n)


def dircos2ddd(l, m, n):
    """
    Convert direction cosines back to dip and dip-direction.

    Parameters
    ----------
    l : East direction cosine.
    m : North direction cosine.
    n : Vertical direction cosine.

    Returns
    -------
    Tuple ``(dip, dipdir)`` in degrees.

    Notes
    -----
    ``n`` is clamped to [−1, 1] before ``asin`` to guard against
    floating-point drift.  If the recovered dip exceeds 90°, the
    antipodal pole is used.
    """
    dipdir = (360 + degrees(atan2(l, m))) % 360
    dip    = 90 - degrees(asin(max(-1.0, min(1.0, n))))
    if dip > 90:
        dip    = 180 - dip
        dipdir = (dipdir + 180) % 360
    return (dip, dipdir)


# =============================================================================
# STANDALONE I/O HELPERS
# =============================================================================

def save_grid_to_shapefile(path_out, file):
    """
    Read a grid-method CSV, discard −999 sentinel rows, and save as a shapefile.

    Methods 5 and 5b write −999 sentinel rows for grid cells with ≤ 3
    measurements.  This function strips those rows and produces a clean
    point shapefile of valid results.  Methods 3 and 4 omit empty cells
    entirely, so their CSVs contain no sentinels — the filter is a no-op
    but harmless.

    Parameters
    ----------
    path_out : Directory containing the CSV (and where the shapefile is written).
    file     : Filename stem (no extension).

    Returns
    -------
    GeoDataFrame of valid (non-sentinel) grid-cell results, or ``None`` if
    the CSV is empty.
    """
    df_sub = pd.read_csv(os.path.join(path_out, file + ".csv"), sep=",")
    if df_sub.empty:
        print("DataFrame Empty")
        return None
    df_sub = df_sub[df_sub.DIP != -999]
    df_sub['geometry'] = df_sub.apply(
        lambda x: Point((float(x.EASTING), float(x.NORTHING))), axis=1)
    df_sub = gpd.GeoDataFrame(df_sub, geometry='geometry')
    df_sub.to_file(os.path.join(path_out, file + ".shp"),
                   driver='ESRI Shapefile')
    return df_sub


def export_shapefile(path_out, file):
    """
    Read a CSV produced by a point-based method and export it as a shapefile.

    Unlike :func:`save_grid_to_shapefile`, this function does **not** filter
    sentinel rows — it is used for methods that only write valid records to
    CSV (e.g. decimation, stochastic).

    Parameters
    ----------
    path_out : Directory containing the CSV (and where the shapefile is written).
    file     : Filename stem (no extension).
    """
    try:
        df = pd.read_csv(os.path.join(path_out, f"{file}.csv"))
    except EmptyDataError:
        print(f"Empty CSV: {file}")
        return
    if df.empty:
        print("CSV is empty.")
        return
    df['geometry'] = df.apply(
        lambda x: Point(float(x.EASTING), float(x.NORTHING)), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.to_file(os.path.join(path_out, f"{file}.shp"), driver='ESRI Shapefile')


# =============================================================================
# PRIVATE HELPER — METHOD 6
# =============================================================================

def _segment_azimuths(contact_gdf):
    """
    Decompose contact geometries into individual line segments with azimuths.

    Each geometry in ``contact_gdf`` is split into its constituent segments
    and the azimuth of each segment is computed in the range 0–180° (strike
    convention: direction is ambiguous).

    Parameters
    ----------
    contact_gdf : GeoDataFrame of contact lines.

    Returns
    -------
    GeoDataFrame with one row per segment, columns ``geometry`` and ``azimuth``.
    """
    rows = []
    for geom in contact_gdf.geometry:
        if geom is None:
            continue
        lines = ([geom] if geom.geom_type in ('LineString', 'LinearRing')
                 else list(geom.geoms))
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                seg = LineString([coords[i], coords[i + 1]])
                dx  = coords[i + 1][0] - coords[i][0]
                dy  = coords[i + 1][1] - coords[i][1]
                az  = (degrees(atan2(dx, dy)) + 360) % 180
                rows.append({'geometry': seg, 'azimuth': az})
    return gpd.GeoDataFrame(rows, crs=contact_gdf.crs)


# =============================================================================
# STAGE 0: COLUMN STANDARDISATION AND BEDDING FILTER
# Reads each raw bedding shapefile, filters to bedding-only records,
# standardises column names, and writes a clean shapefile to bedding_dir.
# All subsampling methods subsequently read from that folder.
# =============================================================================

def prepare_bedding_inputs(datasets, bedding_dir='input_beddings_only',
                            verbose=True):
    """
    Filter each dataset to bedding-only records, standardise column names,
    and write the result to ``bedding_dir``.

    For each dataset the following steps are applied in order:

    1. Load the raw bedding shapefile.
    2. Optionally filter to bedding-only records using ``filter_col`` /
       ``filter_val`` (e.g. to remove foliations or joints).
    3. Rename the source dip column to the standard name ``DIP``.
    4. Create a standard ``DIP_DIR`` column, converting from strike if
       needed::

           dip_direction  →  DIP_DIR = source % 360
           strike         →  DIP_DIR = (source + 90) % 360

    5. Drop rows with null values in DIP, DIP_DIR, EASTING, or NORTHING.
    6. Write the processed shapefile to ``bedding_dir`` using the same
       filename as the source.

    Parameters
    ----------
    datasets : dict
        Dictionary where each key is a dataset name and each value is a
        configuration dict with keys:

        - ``bedding_raw``       : raw bedding shapefile path
        - ``dip_input_col``     : source column name for dip angle
        - ``dipdir_input_col``  : source column name for dip direction or strike
        - ``dipdir_input_type`` : ``'dip_direction'`` or ``'strike'``
        - ``filter_col``        : column to filter on (or ``None``)
        - ``filter_val``        : value to retain (or ``None``)

    bedding_dir : str
        Directory where standardised shapefiles are written.
        Default: ``'input_beddings_only'``.

    verbose : bool
        Print a progress summary for each dataset.  Default: ``True``.

    Returns
    -------
    dict
        Mapping of dataset name → path of the standardised shapefile written.

    Notes
    -----
    Standard column names used throughout:
    ``DIP``, ``DIP_DIR``, ``EASTING``, ``NORTHING``.
    """
    STANDARD_DIP      = 'DIP'
    STANDARD_DIPDIR   = 'DIP_DIR'
    STANDARD_EASTING  = 'EASTING'
    STANDARD_NORTHING = 'NORTHING'

    os.makedirs(bedding_dir, exist_ok=True)
    output_paths = {}

    for ds_name, cfg in datasets.items():
        if verbose:
            print(f"\n  {ds_name.upper()}")

        # Load raw bedding shapefile
        gdf   = gpd.read_file(cfg['bedding_raw'])
        n_raw = len(gdf)

        # Filter to bedding-only records if required
        if cfg.get('filter_col'):
            gdf = gdf[gdf[cfg['filter_col']] == cfg['filter_val']].copy()
            if verbose:
                print(f"    Filtered {cfg['filter_col']} == "
                      f"'{cfg['filter_val']}': {n_raw} → {len(gdf)} records")
        else:
            if verbose:
                print(f"    {n_raw} records (source is already bedding-only; "
                      f"no filter applied)")

        # Standardise DIP column
        src_dip = cfg['dip_input_col']
        gdf[STANDARD_DIP] = gdf[src_dip].astype(float)
        if verbose:
            if src_dip != STANDARD_DIP:
                print(f"    DIP     : '{src_dip}' renamed → '{STANDARD_DIP}'")
            else:
                print(f"    DIP     : '{src_dip}' (already standard name)")

        # Standardise DIP_DIR column
        src_dipdir  = cfg['dipdir_input_col']
        input_type  = cfg['dipdir_input_type']

        if input_type == 'strike':
            gdf[STANDARD_DIPDIR] = (gdf[src_dipdir].astype(float) + 90) % 360
            if verbose:
                print(f"    DIP_DIR : '{src_dipdir}' (strike) converted → "
                      f"'{STANDARD_DIPDIR}'  via  (strike + 90) % 360")
        else:
            gdf[STANDARD_DIPDIR] = gdf[src_dipdir].astype(float) % 360
            if verbose:
                if src_dipdir != STANDARD_DIPDIR:
                    print(f"    DIP_DIR : '{src_dipdir}' (dip direction) "
                          f"renamed → '{STANDARD_DIPDIR}', "
                          f"normalised to 0–360°")
                else:
                    print(f"    DIP_DIR : '{src_dipdir}' "
                          f"(already standard name), normalised to 0–360°")

        # Drop rows with null values in key columns
        n_before = len(gdf)
        gdf = gdf.dropna(
            subset=[STANDARD_DIP, STANDARD_DIPDIR,
                    STANDARD_EASTING, STANDARD_NORTHING]).copy()
        n_dropped = n_before - len(gdf)
        if verbose and n_dropped > 0:
            print(f"    Dropped {n_dropped} rows with null "
                  f"DIP / DIP_DIR / EASTING / NORTHING")

        # Save standardised shapefile
        out_name = os.path.basename(cfg['bedding_raw'])
        out_path = os.path.join(bedding_dir, out_name)
        gdf.to_file(out_path, driver='ESRI Shapefile')
        if verbose:
            print(f"    Saved {len(gdf)} standardised records → {out_path}")

        output_paths[ds_name] = out_path

    return output_paths


# =============================================================================
# UNIFIED ENTRY POINT
# Two-stage pipeline: Stage 0 column standardisation + Stage 1 subsampling.
# =============================================================================

def run_subsampling_file(
        bedding_file,
        output_dir,
        methods            = 'all',
        geology_file       = None,
        grid_n             = 1000,
        decimation_n       = 5,
        stoch_frac         = 0.5,
        stoch_replace      = False,
        stoch_random_state = 42,
        dist_buffer        = 500,
        angle_tol          = 15,
        dip_col            = 'DIP',
        dipdir_col         = 'DIP_DIR',
        easting_col        = 'EASTING',
        northing_col       = 'NORTHING',
        verbose            = True):
    """
    Two-stage structural subsampling pipeline.

    Loads a standardised bedding shapefile and runs the requested subsampling
    algorithm(s), writing CSV and shapefile outputs to ``output_dir``.

    The input file is expected to have already been standardised by
    :func:`prepare_bedding_inputs` (or equivalent), so that ``dip_col``,
    ``dipdir_col``, ``easting_col``, and ``northing_col`` are all present.

    Parameters
    ----------
    bedding_file       : Path to the standardised bedding point shapefile.
    output_dir         : Directory where outputs are written (created if absent).
    methods            : List of method names to run, or ``'all'`` to run
                         every method.  Valid names:
                         ``'decimation'``, ``'stochastic'``,
                         ``'gridcell_average'``, ``'spherical_kent'``,
                         ``'outlier_removal'``, ``'outlier_carmichael'``,
                         ``'firstorder'``.
    geology_file       : Geology polygon shapefile path.  Required when
                         ``'firstorder'`` is included in ``methods``.
    grid_n             : Grid cell size in metres for Methods 3–5b.
                         Default: 1000 m.
    decimation_n       : Step size for Method 1.  Default: 5.
    stoch_frac         : Sampling fraction for Method 2.  Default: 0.5.
    stoch_replace      : Sample with replacement for Method 2.
                         Default: ``False``.
    stoch_random_state : Random seed for Method 2.  Default: 42.
    dist_buffer        : Contact proximity buffer for Method 6.
                         Default: 500 m.
    angle_tol          : Strike alignment tolerance for Method 6.
                         Default: 15°.
    dip_col            : Column name for dip angle.   Default ``'DIP'``.
    dipdir_col         : Column name for dip direction.  Default ``'DIP_DIR'``.
    easting_col        : Column name for easting.   Default ``'EASTING'``.
    northing_col       : Column name for northing.  Default ``'NORTHING'``.
    verbose            : Print progress to stdout.  Default: ``True``.

    Returns
    -------
    dict
        Mapping of method name → output count (int) or filename stem (str),
        one entry per method that was run.
    """
    ALL_METHODS = [
        'decimation', 'stochastic',
        'gridcell_average', 'spherical_kent',
        'outlier_removal', 'outlier_carmichael',
        'firstorder',
    ]

    if methods == 'all':
        methods_to_run = ALL_METHODS
    else:
        methods_to_run = list(methods)
        invalid = [m for m in methods_to_run if m not in ALL_METHODS]
        if invalid:
            raise ValueError(
                f"Unknown method(s): {', '.join(invalid)}\n"
                f"  Valid: {', '.join(ALL_METHODS)}"
            )

    os.makedirs(output_dir, exist_ok=True)

    # Load standardised bedding shapefile
    if verbose:
        print(f"\nLoading {bedding_file} …")
    gdf = gpd.read_file(bedding_file)
    if verbose:
        print(f"  {len(gdf)} records   CRS: {gdf.crs}")

    bounds   = gdf.total_bounds    # [min_x, min_y, max_x, max_y]
    min_x    = int(bounds[0])
    max_x    = int(bounds[2]) + 1
    min_y    = int(bounds[1])
    max_y    = int(bounds[3]) + 1
    if verbose:
        print(f"  Bounds: E {min_x}–{max_x}  N {min_y}–{max_y}")

    engine  = SubsamplingEngine(
        dip=dip_col, dipdir=dipdir_col,
        easting=easting_col, northing=northing_col)
    results = {}

    # ── Method 1: Decimation ──────────────────────────────────────────────────
    if 'decimation' in methods_to_run:
        if verbose:
            print(f"\n--- 1. Decimation (every {decimation_n}th point) ---")
        r = engine.decimation(gdf, n=decimation_n, path_out=output_dir)
        results['decimation'] = len(r)
        if verbose:
            print(f"   → {len(r)} points")

    # ── Method 2: Stochastic subsampling ─────────────────────────────────────
    if 'stochastic' in methods_to_run:
        if verbose:
            print(f"\n--- 2. Stochastic subsampling (frac={stoch_frac}) ---")
        r = engine.stochastic(
            gdf, frac=stoch_frac,
            replace=stoch_replace, random_state=stoch_random_state,
            path_out=output_dir)
        results['stochastic'] = len(r)
        if verbose:
            print(f"   → {len(r)} points")

    # ── Method 3: Grid-cell averaging ─────────────────────────────────────────
    if 'gridcell_average' in methods_to_run:
        if verbose:
            print(f"\n--- 3. Grid-cell averaging (n={grid_n} m) ---")
        f = engine.gridcell_average(
            gdf, min_x, max_x, min_y, max_y,
            n=grid_n, path_out=output_dir)
        r = save_grid_to_shapefile(output_dir, f)
        results['gridcell_average'] = len(r) if r is not None else 0
        if verbose:
            print(f"   → {results['gridcell_average']} cells")

    # ── Method 4: Spherical statistics (Kent distribution) ────────────────────
    if 'spherical_kent' in methods_to_run:
        if verbose:
            print(f"\n--- 4. Spherical / Kent (n={grid_n} m) ---")
        f = engine.spherical_kent(
            gdf, min_x, max_x, min_y, max_y,
            n=grid_n, path_out=output_dir)
        r = save_grid_to_shapefile(output_dir, f)
        results['spherical_kent'] = len(r) if r is not None else 0
        if verbose:
            print(f"   → {results['spherical_kent']} cells")

    # ── Method 5: Outlier removal (single-removal per cell) ───────────────────
    if 'outlier_removal' in methods_to_run:
        if verbose:
            print(f"\n--- 5. Outlier removal (n={grid_n} m) ---")
        f = engine.outlier_removal(
            gdf, min_x, max_x, min_y, max_y,
            n=grid_n, path_out=output_dir)
        r = save_grid_to_shapefile(output_dir, f)
        results['outlier_removal'] = len(r) if r is not None else 0
        if verbose:
            print(f"   → {results['outlier_removal']} cells")

    # ── Method 5b: Carmichael iterative outlier removal ───────────────────────
    if 'outlier_carmichael' in methods_to_run:
        if verbose:
            print(f"\n--- 5b. Outlier — Carmichael (n={grid_n} m) ---")
        f = engine.outlier_carmichael(
            gdf, min_x, max_x, min_y, max_y,
            n=grid_n, path_out=output_dir)
        r = save_grid_to_shapefile(output_dir, f)
        results['outlier_carmichael'] = len(r) if r is not None else 0
        if verbose:
            print(f"   → {results['outlier_carmichael']} cells")

    # ── Method 6: First-order subsampling ─────────────────────────────────────
    if 'firstorder' in methods_to_run:
        if verbose:
            print(f"\n--- 6. First-order subsampling "
                  f"(dist={dist_buffer} m, angle_tol={angle_tol}°) ---")
        if geology_file is None:
            print("   SKIPPED — geology_file not provided.")
            results['firstorder'] = 'skipped'
        else:
            if verbose:
                print(f"   Loading geology polygons: {geology_file} …")
            geology     = gpd.read_file(geology_file)
            contact_gdf = geology[['geometry']].copy()
            contact_gdf['geometry'] = geology.boundary
            contact_gdf = contact_gdf.to_crs(gdf.crs)
            contact_union = unary_union(contact_gdf.geometry)
            n_within = (gdf.geometry.distance(contact_union) <= dist_buffer).sum()
            if verbose:
                print(f"   {n_within} bedding points within "
                      f"{dist_buffer} m of a contact")
            r = engine.firstorder(
                gdf, contact_gdf,
                dist_buffer=dist_buffer, angle_tol=angle_tol,
                path_out=output_dir)
            results['firstorder'] = len(r)
            if verbose:
                print(f"   → {len(r)} points")

    return results


# =============================================================================
# SIMPLE ENTRY POINT
# Run a single subsampling method without the two-stage pipeline wrapper.
# =============================================================================

def subsample_file(bedding_file, output_dir, method,
                   geology_file=None, grid_n=1000, **kwargs):
    """
    Run a single subsampling method on a standardised bedding shapefile.

    A lightweight alternative to :func:`run_subsampling_file` when only one
    method is needed and no pipeline overhead is required.

    Parameters
    ----------
    bedding_file : Path to the standardised bedding point shapefile.
    output_dir   : Directory where outputs are written.
    method       : Algorithm name (see :meth:`SubsamplingEngine.subsample`).
    geology_file : Geology polygon shapefile path (required for
                   ``'firstorder'`` only).
    grid_n       : Grid cell size in metres.  Default: 1000 m.
    **kwargs     : Additional keyword arguments forwarded to the algorithm
                   (e.g. ``n=5`` for decimation).

    Returns
    -------
    The return value of the selected algorithm.
    """
    os.makedirs(output_dir, exist_ok=True)

    gdf    = gpd.read_file(bedding_file)
    engine = SubsamplingEngine()
    bounds = gdf.total_bounds
    min_x, max_x = int(bounds[0]), int(bounds[2]) + 1
    min_y, max_y = int(bounds[1]), int(bounds[3]) + 1

    if method in ('gridcell_average', 'spherical_kent',
                  'outlier_removal', 'outlier_carmichael'):
        return engine.subsample(
            method, gdf, path_out=output_dir,
            minx=min_x, maxx=max_x, miny=min_y, maxy=max_y,
            grid_n=grid_n, **kwargs)

    if method == 'firstorder':
        if geology_file is None:
            raise ValueError(
                "geology_file is required for the 'firstorder' method.")
        geology     = gpd.read_file(geology_file)
        contact_gdf = geology[['geometry']].copy()
        contact_gdf['geometry'] = geology.boundary
        contact_gdf = contact_gdf.to_crs(gdf.crs)
        kwargs['contact_gdf'] = contact_gdf
        return engine.subsample(method, gdf, path_out=output_dir, **kwargs)

    return engine.subsample(method, gdf, path_out=output_dir, **kwargs)

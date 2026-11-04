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
4. spherical_kent        – grid-cell Kent statistics (κ, β); optional kappa_min filter
5. outlier_removal       – max-Δκ single-outlier removal per cell (min_for_outlier=3)
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
import shapely
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

        The Kent (5-parameter Fisher-Bingham) distribution on the sphere is
        characterised by:

        - **kappa** : concentration parameter (higher = tighter cluster)
        - **beta**  : ovalness / anisotropy parameter (0 = rotationally symmetric)

        Moment estimates of Kent (1982), as given by Leong & Carlile (1998)::

            kappa = 1 / (2 - 2*r1 - r2) + 1 / (2 - 2*r1 + r2)
            beta  = 0.5 * [1 / (2 - 2*r1 - r2) - 1 / (2 - 2*r1 + r2)]

        where ``r1`` is the **mean** resultant length (resultant length / N)
        and ``r2`` the difference between the two largest eigenvalues of the
        scatter matrix in the plane normal to the mean direction, found by
        rotating the scatter matrix with H (mean direction to the z axis) and
        then K (the sword angle psi).

        Parameters
        ----------
        gpdataframe : DataFrame with a ``vector`` column of ``(l, m, n)`` tuples.

        Returns
        -------
        Tuple ``(count, kappa, beta)``.  kappa is ``inf`` and beta ``nan`` when
        the estimate has no finite value: a single measurement, identical
        measurements, or a cell whose scatter makes a denominator <= 0.

        Notes
        -----
        Corrections (v1.1.0) to the v1.0.0 implementation, which produced
        invalid kappa values (0, -1 or negative):

        * ``r1`` is the mean resultant length, not the resultant length itself
          (the resultant length grows with N, so 2 - 2*r1 was negative).
        * H has the rotated axes as **columns** (H^T S H puts the mean
          direction on z); v1.0.0 used the transpose.
        * The mean-direction azimuth uses ``atan2`` so the quadrant is kept.
        * The sword angle uses the diagonal difference B[0,0] - B[1,1]
          (unchanged from v1.0.0).
        """
        df = pd.DataFrame(gpdataframe)
        x = np.array(df['vector'].tolist(), dtype=float)
        count = len(x)
        if count < 2:
            return (count, float('inf'), float('nan'))

        xbar = x.mean(axis=0)
        r1 = float(np.sqrt((xbar ** 2).sum()))          # mean resultant length
        if r1 >= 1 - 1e-12:                              # identical measurements
            return (count, float('inf'), float('nan'))
        if r1 < 1e-12:                                   # antipodal vectors cancel
            return (count, float('nan'), float('nan'))

        g = xbar / r1
        theta = np.arccos(np.clip(g[2], -1.0, 1.0))
        phi = np.arctan2(g[1], g[0])
        H = np.array([
            [np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
            [np.cos(theta) * np.sin(phi),  np.cos(phi), np.sin(theta) * np.sin(phi)],
            [-np.sin(theta),               0.0,         np.cos(theta)],
        ])
        S = x.T @ x / count                              # scatter (orientation) matrix / N
        B = H.T @ S @ H
        psi = 0.5 * np.arctan2(2 * B[0, 1], B[0, 0] - B[1, 1])   # sword angle
        K = np.array([
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi),  np.cos(psi), 0.0],
            [0.0,          0.0,         1.0],
        ])
        G = H @ K
        V = G.T @ S @ G
        r2 = V[0, 0] - V[1, 1]

        d1 = 2 - 2 * r1 - r2
        d2 = 2 - 2 * r1 + r2
        if d1 <= 0 or d2 <= 0:
            return (count, float('inf'), float('nan'))
        kappa = 1 / d1 + 1 / d2
        beta = 0.5 * (1 / d1 - 1 / d2)
        return (count, float(kappa), float(beta))

    @staticmethod
    def _fmt_kent(value):
        """Kent parameter for a CSV field: 2 decimals, or -998 when undefined."""
        return f"{value:.2f}" if np.isfinite(value) else "-998"

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

    # =========================================================================
    # ARRAY HELPERS FOR THE GRID-CELL METHODS
    # The grid-cell methods work on plain numpy arrays sorted by cell: no pandas work
    # is done per cell (that was most of their run time).
    # =========================================================================

    def _grid_groups(self, gdf, minx, maxx, miny, maxy, n):
        """
        Bin the measurements into grid cells.  Returns a dict of arrays sorted by cell
        (cells in the same order as ``groupby(['_cx', '_cy'])``, measurements within a cell
        in their original order): ``X`` direction cosines (n, 3), ``e`` / ``n`` coordinates,
        ``labels`` original index labels, and ``starts`` / ``ends`` of each cell.
        Measurements outside the bounds or with a null dip / dip direction are left out.
        """
        e = np.asarray(gdf[self.easting], dtype=float)
        nn = np.asarray(gdf[self.northing], dtype=float)
        dip = np.asarray(gdf[self.dip], dtype=float)
        dd = np.asarray(gdf[self.dipdir], dtype=float)
        ok = ((e >= minx) & (e < maxx) & (nn >= miny) & (nn < maxy)
              & np.isfinite(dip) & np.isfinite(dd))
        labels = np.asarray(gdf.index)[ok]
        e, nn = e[ok], nn[ok]
        dip_r, dd_r = np.radians(dip[ok]), np.radians(dd[ok])
        X = np.stack([np.sin(dip_r) * np.sin(dd_r),       # east
                      np.sin(dip_r) * np.cos(dd_r),       # north
                      np.cos(dip_r)], axis=1)             # down
        cx = ((e - minx) / n).astype(np.int64)
        cy = ((nn - miny) / n).astype(np.int64)
        order = np.lexsort((cy, cx))                      # stable: original order inside a cell
        cx, cy = cx[order], cy[order]
        if len(cx):
            starts = np.flatnonzero(np.r_[True, (cx[1:] != cx[:-1]) | (cy[1:] != cy[:-1])])
            ends = np.r_[starts[1:], len(cx)]
        else:
            starts = ends = np.zeros(0, dtype=np.int64)
        return {'X': X[order], 'e': e[order], 'n': nn[order], 'labels': labels[order],
                'starts': starts, 'ends': ends}

    @staticmethod
    def _orientation_from_sums(l_sum, m_sum, n_sum, count):
        """(dip, dipdir) of the vector mean given the component sums - as :meth:`_calc_mean_orientation`."""
        r_bar = sqrt(float((l_sum / count)**2 + (m_sum / count)**2 + (n_sum / count)**2))
        if r_bar < 1e-12:
            return (float('nan'), float('nan'))
        return dircos2ddd((l_sum / count) / r_bar, (m_sum / count) / r_bar, (n_sum / count) / r_bar)

    @staticmethod
    def _cell_sums(X, starts):
        """Per-cell sum of the direction cosines (K, 3) and of their outer products (K, 3, 3)."""
        sums = np.add.reduceat(X, starts, axis=0) + 0.0
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        comp = [np.add.reduceat(v, starts) for v in (x * x, x * y, x * z, y * y, y * z, z * z)]
        T = np.empty((len(starts), 3, 3))
        T[:, 0, 0], T[:, 0, 1], T[:, 0, 2] = comp[0], comp[1], comp[2]
        T[:, 1, 0], T[:, 1, 1], T[:, 1, 2] = comp[1], comp[3], comp[4]
        T[:, 2, 0], T[:, 2, 1], T[:, 2, 2] = comp[2], comp[4], comp[5]
        return sums, T

    def _kappa_beta_of(self, X):
        """Kent (kappa, beta) of one set of direction cosines X (n, 3), as :meth:`_calc_kent`."""
        n = len(X)
        k, b = self._kent_kappa_batch(n, X.mean(axis=0)[None, :], (X.T @ X / n)[None, :, :],
                                      return_beta=True)
        return float(k[0]), float(b[0])

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
        Filename stem (str) of the CSV written to ``path_out``.  Columns:
        EASTING, NORTHING, DIP, DIP_DIR and ``count`` (the number of
        measurements averaged in the cell).  Pass this to
        :func:`save_grid_to_shapefile` to produce a point shapefile.
        """
        gr = self._grid_groups(gpdataframe, minx, maxx, miny, maxy, n)
        e, nn, starts, ends = gr['e'], gr['n'], gr['starts'], gr['ends']

        file = f"structure_file_gridcell_{n}"
        fieldnames = ['EASTING', 'NORTHING', 'DIP', 'DIP_DIR', 'count']

        with open(os.path.join(path_out, file + ".csv"), "w",
                  encoding="utf-8") as out:
            out.write(','.join(fieldnames) + '\n')
            if len(starts):
                counts = ends - starts
                sums = np.add.reduceat(gr['X'], starts, axis=0) + 0.0
                for k in range(len(starts)):
                    s, t_ = starts[k], ends[k]
                    # Output at the mean position of the input points, not the cell
                    # centre, so a single point in a cell stays where it was.
                    centx = float(e[s:t_].mean())
                    centy = float(nn[s:t_].mean())
                    dip_val, dipdir_val = self._orientation_from_sums(
                        sums[k, 0], sums[k, 1], sums[k, 2], int(counts[k]))
                    dip_str    = ('NaN' if math.isnan(dip_val)
                                  else str(int(dip_val)))
                    dipdir_str = ('NaN' if math.isnan(dipdir_val)
                                  else str(int(dipdir_val)))
                    out.write(f"{centx:.8f},{centy:.8f},"
                              f"{dip_str},{dipdir_str},{int(counts[k])}\n")
        return file

    def count_cells(self, gpdataframe, minx, maxx, miny, maxy, n, min_points=1):
        """
        Number of grid cells (same grid as the grid-cell methods) holding at least
        ``min_points`` measurements - a fast stand-in for running a grid method
        just to count its output (used by the auto-fit to N points search).

        ``min_points=1`` gives the number of output points of the Kent and Kent
        Outlier methods (without a kappa filter); ``min_points=4`` that of
        :meth:`outlier_carmichael`, which skips cells of 3 or fewer.
        """
        e = np.asarray(gpdataframe[self.easting], dtype=float)
        nn = np.asarray(gpdataframe[self.northing], dtype=float)
        keep = (e >= minx) & (e < maxx) & (nn >= miny) & (nn < maxy)
        cx = ((e[keep] - minx) / n).astype(np.int64)
        cy = ((nn[keep] - miny) / n).astype(np.int64)
        _, sizes = np.unique(np.stack([cx, cy], axis=1), axis=0, return_counts=True)
        return int((sizes >= min_points).sum())

    # =========================================================================
    # METHOD 4: SPHERICAL STATISTICS (KENT DISTRIBUTION)
    # Compute Kent distribution parameters (mean, κ, β) per grid cell.
    # =========================================================================

    def spherical_kent(self, gpdataframe, minx, maxx, miny, maxy,
                       n=1000, kappa_min=None, path_out='outputs/'):
        """
        Spherical statistics per grid cell: Kent mean orientation, kappa and beta,
        with an optional kappa filter.

        The study area is divided into n x n metre cells (the same grid as
        :meth:`gridcell_average`).  For each non-empty cell the representative
        orientation is the Kent mean direction (the normalised vector mean, so
        the orientations equal those of :meth:`gridcell_average`), written at
        the mean position of the cell's measurements together with the number
        of measurements and the Kent moment estimates of kappa (concentration)
        and beta (ovalness), see :meth:`_calc_kent`.

        Two variants:

        * ``kappa_min=None`` (spherical statistics **without** kappa filter):
          every non-empty cell is retained; kappa and beta are reported but do
          not change the output orientations.
        * ``kappa_min=k`` (spherical statistics **with** kappa filter): only
          cells whose kappa is finite and >= k are retained, i.e. cells with a
          statistically concentrated mean.  Cells with a single measurement
          (kappa undefined) and dispersed or girdle-shaped cells are dropped.

        Undefined kappa or beta (one measurement, identical measurements) are
        written as -998.

        Parameters
        ----------
        gpdataframe : Input GeoDataFrame of structural measurements.
        minx, maxx  : Western and eastern study-area boundaries (metres).
        miny, maxy  : Southern and northern study-area boundaries (metres).
        n           : Grid cell size in metres.  Default: 1000 m.
        kappa_min   : None (no filter, default) or the minimum Kent kappa a
                      cell must reach to be retained.
        path_out    : Output directory path.

        Returns
        -------
        Filename stem (str) of the CSV written to ``path_out``:
        ``structure_file_spherical_{n}`` without filter,
        ``structure_file_spherical_kappa{kappa_min}_{n}`` with filter.
        """
        gr = self._grid_groups(gpdataframe, minx, maxx, miny, maxy, n)
        e, nn, starts, ends = gr['e'], gr['n'], gr['starts'], gr['ends']

        file = (f"structure_file_spherical_{n}" if kappa_min is None
                else f"structure_file_spherical_kappa{kappa_min:g}_{n}")
        fieldnames = ['EASTING', 'NORTHING', 'DIP', 'DIP_DIR',
                      'count', 'kappa', 'beta']

        with open(os.path.join(path_out, file + ".csv"), "w",
                  encoding="utf-8") as out:
            out.write(','.join(fieldnames) + '\n')
            if len(starts):
                counts = ends - starts
                sums, T = self._cell_sums(gr['X'], starts)       # all cells at once
                kappa, beta = self._kent_kappa_batch(
                    counts, sums / counts[:, None], T / counts[:, None, None], return_beta=True)
                for k in range(len(starts)):
                    if kappa_min is not None and not (
                            np.isfinite(kappa[k]) and kappa[k] >= kappa_min):
                        continue
                    dip2, dipdir2 = self._orientation_from_sums(
                        sums[k, 0], sums[k, 1], sums[k, 2], int(counts[k]))
                    if math.isnan(dip2) or math.isnan(dipdir2):
                        continue
                    s, t_ = starts[k], ends[k]
                    # Mean position of the input points (see gridcell_average)
                    centx = float(e[s:t_].mean())
                    centy = float(nn[s:t_].mean())
                    out.write(f"{centx:.8f},{centy:.8f},"
                              f"{int(dip2)},{int(dipdir2)},{int(counts[k])},"
                              f"{self._fmt_kent(kappa[k])},{self._fmt_kent(beta[k])}\n")
        return file

    # =========================================================================
    # METHOD 5: OUTLIER REMOVAL (SINGLE-REMOVAL PER CELL)
    # Remove the one measurement per cell whose removal most increases kappa.
    # =========================================================================

    @staticmethod
    def _kent_kappa_batch(count, xbar, S, return_beta=False):
        """
        Kent kappa (and optionally beta) for a batch of sets, from their sufficient statistics.

        The same estimator as :meth:`_calc_kent`, vectorised so that the kappa of many
        sets can be computed in one go (cells, leave-one-out sets, ...).

        Parameters
        ----------
        count : number of measurements in each set (int, or array of shape (m,)).
        xbar  : mean direction cosines of each set, shape (m, 3).
        S     : scatter matrix (sum of x x^T) / count of each set, shape (m, 3, 3).

        Returns
        -------
        kappa (m,), or (kappa, beta) with ``return_beta``.  ``inf`` where undefined for a
        finite reason (fewer than 2 measurements, identical measurements, or a
        non-positive denominator; beta is then ``nan``) and ``nan`` where the resultant
        vector cancels - exactly the conventions of :meth:`_calc_kent`.
        """
        m = xbar.shape[0]
        count = np.broadcast_to(np.asarray(count), (m,))
        r1 = np.sqrt((xbar ** 2).sum(axis=1))
        kappa = np.full(m, np.inf)
        beta = np.full(m, np.nan)
        kappa[(r1 < 1e-12) & (count >= 2)] = np.nan
        ok = (count >= 2) & (r1 < 1 - 1e-12) & (r1 >= 1e-12)
        if not ok.any():
            return (kappa, beta) if return_beta else kappa
        xb, S_, r = xbar[ok], S[ok], r1[ok]
        g = xb / r[:, None]
        theta = np.arccos(np.clip(g[:, 2], -1.0, 1.0))
        phi = np.arctan2(g[:, 1], g[:, 0])
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        # H maps the mean direction to z; only its first two columns h1, h2 are needed
        h1 = np.stack([ct * cp, ct * sp, -st], axis=1)
        h2 = np.stack([-sp, cp, np.zeros_like(cp)], axis=1)
        S1, S2 = np.einsum('mij,mj->mi', S_, h1), np.einsum('mij,mj->mi', S_, h2)
        b00 = (h1 * S1).sum(axis=1)
        b01 = (h1 * S2).sum(axis=1)
        b11 = (h2 * S2).sum(axis=1)
        psi = 0.5 * np.arctan2(2 * b01, b00 - b11)            # sword angle
        c, s_ = np.cos(psi), np.sin(psi)
        v00 = b00 * c * c + 2 * b01 * c * s_ + b11 * s_ * s_
        v11 = b00 * s_ * s_ - 2 * b01 * c * s_ + b11 * c * c
        r2 = v00 - v11
        d1 = 2 - 2 * r - r2
        d2 = 2 - 2 * r + r2
        good = (d1 > 0) & (d2 > 0)
        k = np.full(len(r), np.inf)
        b = np.full(len(r), np.nan)
        k[good] = 1 / d1[good] + 1 / d2[good]
        b[good] = 0.5 * (1 / d1[good] - 1 / d2[good])
        kappa[ok] = k
        beta[ok] = b
        return (kappa, beta) if return_beta else kappa

    def _loo_kappas(self, X):
        """
        Kent kappa of the cell with each measurement left out in turn (X: (n, 3)).

        The same estimator as :meth:`_kent_kappa_batch`, but evaluated without building a
        scatter matrix per left-out set: the quadratic forms of the reduced scatter matrix
        are the quadratic forms of the full one minus the left-out point's own contribution.
        """
        n = len(X)
        m = n - 1
        kappa = np.full(n, np.inf)
        if m < 2:
            return kappa
        total = X.sum(axis=0)
        T = X.T @ X
        xbar = (total - X) / m
        r1 = np.sqrt((xbar ** 2).sum(axis=1))
        kappa[r1 < 1e-12] = np.nan
        ok = (r1 < 1 - 1e-12) & (r1 >= 1e-12)
        if not ok.any():
            return kappa
        x, xb, r = X[ok], xbar[ok], r1[ok]
        g = xb / r[:, None]
        theta = np.arccos(np.clip(g[:, 2], -1.0, 1.0))
        phi = np.arctan2(g[:, 1], g[:, 0])
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        h1 = np.stack([ct * cp, ct * sp, -st], axis=1)
        h2 = np.stack([-sp, cp, np.zeros_like(cp)], axis=1)
        Th1, Th2 = h1 @ T, h2 @ T
        p1, p2 = (x * h1).sum(axis=1), (x * h2).sum(axis=1)
        b00 = ((h1 * Th1).sum(axis=1) - p1 * p1) / m
        b01 = ((h2 * Th1).sum(axis=1) - p1 * p2) / m
        b11 = ((h2 * Th2).sum(axis=1) - p2 * p2) / m
        psi = 0.5 * np.arctan2(2 * b01, b00 - b11)            # sword angle
        c, s_ = np.cos(psi), np.sin(psi)
        r2 = (b00 * c * c + 2 * b01 * c * s_ + b11 * s_ * s_) - (b00 * s_ * s_ - 2 * b01 * c * s_ + b11 * c * c)
        d1 = 2 - 2 * r - r2
        d2 = 2 - 2 * r + r2
        good = (d1 > 0) & (d2 > 0)
        k = np.full(len(r), np.inf)
        k[good] = 1 / d1[good] + 1 / d2[good]
        kappa[ok] = k
        return kappa

    def _iterative_outliers(self, X, min_keep, break_ratio):
        """
        Outliers of one cell by repeated leave-one-out removal with a break rule.

        Points are removed one at a time, each time the one whose removal gives
        the highest Kent kappa, up to half the cell and never below ``min_keep``
        points after the first removal.  The relative kappa gain of each removal
        is recorded.  Just stopping once a removal no longer raises kappa does
        not work, because removing any point from a small sample usually raises
        kappa a little (30-70% for 8-15 points): in a clean cell that would trim
        half of it.  Real outliers instead show up as one large jump, so the
        sequence is cut at its largest gain (as in :meth:`outlier_carmichael`),
        and only if that gain is at least ``break_ratio`` times the median gain
        of the other steps; otherwise nothing is removed.

        ``X`` is the (n, 3) array of direction cosines of the cell.  Returns the list
        of row positions (into X) to remove, in removal order (possibly empty).
        """
        n = len(X)
        active = np.arange(n)
        k_prev = float(self._kent_kappa_batch(
            n, X.mean(axis=0)[None, :], (X.T @ X / n)[None, :, :])[0])
        seq = []                                # (position removed, relative kappa gain)
        max_remove = n // 2
        while len(seq) < max_remove and np.isfinite(k_prev) and k_prev > 0:
            if seq and len(active) <= min_keep:
                break
            kap = self._loo_kappas(X[active])
            kap = np.where(np.isfinite(kap), kap, -np.inf)
            j = int(np.argmax(kap))
            if not np.isfinite(kap[j]):
                break                           # kappa undefined for everything left
            seq.append((int(active[j]), (kap[j] - k_prev) / k_prev))
            k_prev = float(kap[j])
            active = np.delete(active, j)
        if not seq:
            return []
        gains = [g for _, g in seq]
        best_i = int(np.argmax(gains))
        if gains[best_i] <= 0:
            return []
        others = gains[:best_i] + gains[best_i + 1:]
        if others and gains[best_i] < break_ratio * max(float(np.median(others)), 0.05):
            return []                           # no jump stands out from the noise
        return [pos for pos, _ in seq[:best_i + 1]]

    def outlier_removal(self, gdf, minx, maxx, miny, maxy,
                        n=1000, min_for_outlier=3, path_out='outputs/',
                        iterative=False, break_ratio=5.0):
        """
        Remove outliers per grid cell using the max-delta-kappa criterion: at
        most one by default, or several with ``iterative=True``.

        Same grid as :meth:`gridcell_average`.  For each cell with at least
        ``min_for_outlier`` measurements, the Kent kappa of the cell is
        computed, then each measurement is left out in turn and
        ``delta_kappa = kappa(without it) - kappa(all)`` recorded.  The
        measurement with the largest delta_kappa is the outlier and is removed,
        provided its removal increases kappa (delta_kappa > 0); otherwise
        nothing is removed.  The representative orientation is the vector mean
        of the remaining measurements, at their mean position.

        Cells with one measurement keep it; cells with two are averaged, since
        removing either leaves a single measurement whose kappa is undefined,
        so delta_kappa cannot tell the two apart.

        The single-removal rule is a deliberate simplification of the iterative
        method of Carmichael & Ailleres (2016), which is available as
        :meth:`outlier_carmichael`.  With ``iterative=True`` the leave-one-out
        step is repeated to remove several outliers: points are taken out in
        order of their effect on kappa and the sequence is cut at its largest
        kappa jump, provided that jump stands out (see
        :meth:`_iterative_outliers`).  This differs from
        :meth:`outlier_carmichael`, which ranks points by angle from the cell
        mean rather than by their effect on kappa, and has no check that the
        jump stands out.

        Parameters
        ----------
        gdf             : Input GeoDataFrame of structural measurements.
        minx, maxx      : Western and eastern study-area boundaries (metres).
        miny, maxy      : Southern and northern study-area boundaries (metres).
        n               : Grid cell size in metres.  Default: 1000 m.
        min_for_outlier : Minimum measurements in a cell for outlier removal.
                          Default: 3.
        path_out        : Output directory path.
        iterative       : Remove several outliers (see above).  Default: False.
        break_ratio     : With ``iterative``, the largest kappa jump must be at
                          least this many times the median of the other steps
                          to count as an outlier boundary.  Default: 5.

        Returns
        -------
        Filename stem (str) of the CSV written to ``path_out``.  Columns:
        EASTING, NORTHING, DIP, DIP_DIR, count (measurements kept), kappa and
        beta of the kept measurements (-998 if undefined), REMOVED_INDEX
        (index of the first removed measurement, -1 if none) and n_removed.
        """
        gr = self._grid_groups(gdf, minx, maxx, miny, maxy, n)
        X, e, nn, labels = gr['X'], gr['e'], gr['n'], gr['labels']
        starts, ends = gr['starts'], gr['ends']

        file     = (f"structure_file_outlier_iterative_{n}" if iterative
                    else f"structure_file_outlier_{n}")
        csv_path = os.path.join(path_out, f"{file}.csv")

        with open(csv_path, "w", encoding="utf-8") as out:
            out.write("EASTING,NORTHING,DIP,DIP_DIR,"
                      "count,kappa,beta,REMOVED_INDEX,n_removed\n")

            for s, t_ in zip(starts, ends):
                Xc = X[s:t_]
                removed = []
                if len(Xc) >= min_for_outlier and not iterative:
                    k0, _ = self._kappa_beta_of(Xc)
                    if np.isfinite(k0):
                        deltas = self._loo_kappas(Xc) - k0      # all leave-one-out at once
                        deltas = np.where(np.isnan(deltas), -np.inf, deltas)
                        j = int(np.argmax(deltas))
                        if deltas[j] > 0:
                            removed = [j]
                elif len(Xc) >= min_for_outlier:
                    removed = self._iterative_outliers(Xc, min_for_outlier, break_ratio)
                if removed:
                    keep = np.ones(len(Xc), dtype=bool)
                    keep[removed] = False
                else:
                    keep = slice(None)
                Xf = Xc[keep]
                sums = Xf.sum(axis=0) + 0.0
                dip_f, dipdir_f = self._orientation_from_sums(
                    sums[0], sums[1], sums[2], len(Xf))
                if math.isnan(dip_f) or math.isnan(dipdir_f):
                    continue
                kappa, beta = self._kappa_beta_of(Xf)
                # Mean position of the retained points, not the cell centre
                centx = float(e[s:t_][keep].mean())
                centy = float(nn[s:t_][keep].mean())
                first = labels[s + removed[0]] if removed else -1
                out.write(f"{centx:.8f},{centy:.8f},{int(dip_f)},{int(dipdir_f)},"
                          f"{len(Xf)},{self._fmt_kent(kappa)},"
                          f"{self._fmt_kent(beta)},"
                          f"{first},"
                          f"{len(removed)}\n")

        return file

    # =========================================================================
    # METHOD 5b: CARMICHAEL ITERATIVE OUTLIER REMOVAL
    # Remove multiple outliers per cell using the Δκ-break algorithm.
    # =========================================================================

    def outlier_carmichael(self, gdf, minx, maxx, miny, maxy,
                           n=1000, path_out='outputs/'):
        """
        Remove multiple outliers per grid cell using the Carmichael (2016)
        iterative delta-kappa-break algorithm.

        For each grid cell, points are ranked by angular distance from the cell
        mean direction (farthest first).  They are then removed iteratively and
        kappa is recomputed after each removal.  The step with the largest
        increase in kappa defines the outlier boundary: every point removed up
        to and including that step is treated as an outlier.  At most N // 2
        points may be removed from any cell of N measurements.

        If no removal step produces a positive delta-kappa, the cell is left
        unchanged.  Cells with 3 or fewer measurements are excluded.  Each
        output point is at the mean position of the retained measurements.

        Parameters
        ----------
        gdf         : Input GeoDataFrame of structural measurements.
        minx, maxx  : Western and eastern study-area boundaries (metres).
        miny, maxy  : Southern and northern study-area boundaries (metres).
        n           : Grid cell size in metres.  Default: 1000 m.
        path_out    : Output directory path.

        Returns
        -------
        Filename stem (str) of the CSV written to ``path_out``.  Columns:
        EASTING, NORTHING, DIP, DIP_DIR, count (measurements kept), kappa and
        beta of the kept measurements (-998 if undefined), n_removed.

        References
        ----------
        Carmichael, T. & Ailleres, L. (2016). Method and analysis for the
        upscaling of structural data. Journal of Structural Geology, 83, 121-133.
        """
        gr = self._grid_groups(gdf, minx, maxx, miny, maxy, n)
        X, e, nn = gr['X'], gr['e'], gr['n']
        starts, ends = gr['starts'], gr['ends']

        file     = f"structure_file_outlier_carmichael_{n}"
        csv_path = os.path.join(path_out, f"{file}.csv")

        with open(csv_path, "w", encoding="utf-8") as out:
            out.write("EASTING,NORTHING,DIP,DIP_DIR,"
                      "count,kappa,beta,n_removed\n")

            for s, t_ in zip(starts, ends):
                N = t_ - s
                if N <= 3:
                    continue            # too few measurements to define an outlier
                P = X[s:t_]

                # Cell mean direction as a unit vector
                l_s, m_s, n_s = (float(v) for v in P.sum(axis=0) + 0.0)
                g_len = sqrt(l_s**2 + m_s**2 + n_s**2)
                if g_len < 1e-12:
                    continue            # antipodal vectors cancel: no mean direction
                mean_vec = np.array([l_s / g_len, m_s / g_len, n_s / g_len])

                # Rank points by angular distance from the mean (farthest first)
                dots = np.clip(P @ mean_vec, -1.0, 1.0)
                ang_dists = np.degrees(np.arccos(dots))
                rank = np.argsort(ang_dists)[::-1]

                # Kappa after removing the 0, 1, ..., N//2 farthest points - all sets at once
                # from suffix sums (sfx[i] = sum of the points ranked i and beyond)
                m_rem = N // 2
                Pr = P[rank]
                sfx = np.cumsum(Pr[::-1], axis=0)[::-1]
                outer = Pr[:, :, None] * Pr[:, None, :]
                sfx_T = np.cumsum(outer[::-1], axis=0)[::-1]
                cnt = N - np.arange(m_rem + 1)
                kap = self._kent_kappa_batch(
                    cnt, sfx[:m_rem + 1] / cnt[:, None], sfx_T[:m_rem + 1] / cnt[:, None, None])
                # kappa is undefined (inf/nan) for one or identical measurements:
                # such steps cannot define the break
                k_prev, k_t = kap[:-1], kap[1:]
                delta_seq = np.where(np.isfinite(k_t) & np.isfinite(k_prev), k_t - k_prev, np.nan)

                # Find the delta-kappa break (finite steps only)
                if np.all(np.isnan(delta_seq)):
                    best_i, best_delta = 0, 0.0
                else:
                    best_i     = int(np.nanargmax(delta_seq))
                    best_delta = delta_seq[best_i]

                if best_delta <= 0:
                    # No removal improves concentration - retain all points
                    n_removed = 0
                    keep = slice(None)
                else:
                    n_removed = best_i + 1
                    keep = np.ones(N, dtype=bool)
                    keep[rank[:n_removed]] = False
                Xf = P[keep]
                sums = Xf.sum(axis=0) + 0.0
                dip_f, dipdir_f = self._orientation_from_sums(
                    sums[0], sums[1], sums[2], len(Xf))
                if math.isnan(dip_f) or math.isnan(dipdir_f):
                    continue
                kappa_f, beta_f = self._kappa_beta_of(Xf)
                # Mean position of the retained points, not the cell centre
                centx = float(e[s:t_][keep].mean())
                centy = float(nn[s:t_][keep].mean())
                out.write(f"{centx:.8f},{centy:.8f},{int(dip_f)},{int(dipdir_f)},"
                          f"{len(Xf)},{self._fmt_kent(kappa_f)},"
                          f"{self._fmt_kent(beta_f)},{n_removed}\n")

        return file

    # =========================================================================
    # METHOD 6: FIRST-ORDER SUBSAMPLING
    # Filter by proximity to contacts and angular alignment with them.
    # =========================================================================

    def firstorder(self, gpdataframe, contact_gdf,
                   dist_buffer=500, angle_tol=15, path_out='outputs/',
                   max_points=None):
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
        max_points  : Optional cap on the number of points returned. If more
                      points pass both filters, the ``max_points`` best are
                      kept, ranked by
                      ``distance/dist_buffer + angle_diff/angle_tol``
                      (lowest score = closest to a contact and best aligned
                      with it). If fewer pass, all are returned.

        Returns
        -------
        GeoDataFrame of retained measurements passing both filters.
        """
        gdf = gpd.GeoDataFrame(gpdataframe.copy())
        if gdf.crs is not None and contact_gdf.crs is not None:
            contact_gdf = contact_gdf.to_crs(gdf.crs)

        # Contact segments and their azimuths, as arrays; nearest segment of every point from
        # a spatial index (the distance to one big unioned contact geometry has no index and
        # was by far the slowest step: ~90 s for 28,000 points against 146,000 contact vertices)
        a, b, seg_az = _contact_segments(contact_gdf.geometry.values)
        if len(a) == 0:
            print("DataFrame Empty — no contact lines.")
            return gdf.iloc[0:0].copy()
        tree = shapely.STRtree(shapely.linestrings(np.stack([a, b], axis=1)))
        idx, dist = tree.query_nearest(gdf.geometry.values, return_distance=True,
                                       all_matches=True)
        ip, isg = idx[0], idx[1]

        # Several segments can be exactly as near (a shared contact is in the data twice, and a
        # point can be equally near two segments meeting at a vertex): use the best-aligned one,
        # so the result does not depend on the order of the data
        strike = (gdf[self.dipdir].astype(float).values - 90) % 180
        diff = np.abs(((seg_az[isg] - strike[ip] + 90) % 180) - 90)
        dists = np.full(len(gdf), np.inf)
        dists[ip] = dist
        ang = np.full(len(gdf), np.inf)
        np.minimum.at(ang, ip, np.where(np.isnan(diff), np.inf, diff))

        # Step 1: Distance filter
        near = dists <= dist_buffer
        if not near.any():
            print("DataFrame Empty — no measurements within the distance buffer.")
            return gdf[near].copy()

        # Step 2: Angular filter (strike against the nearest contact segment)
        passed = near & (ang <= angle_tol)
        df_sub = gdf[passed].copy()

        # Optional cap: keep the best-ranked points (closest to a contact and
        # best aligned with it), giving an exact count with no iteration.
        self.last_firstorder_n_pass = len(df_sub)
        if max_points is not None and len(df_sub) > max_points:
            eps = 1e-12
            score = (dists[passed] / max(dist_buffer, eps)
                     + ang[passed] / max(angle_tol, eps))
            best = np.argsort(score, kind='stable')[:int(max_points)]
            df_sub = df_sub.iloc[best]
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
                n=grid_n, kappa_min=kwargs.get('kappa_min'), path_out=path_out)

        if method == 'outlier_removal':
            return self.outlier_removal(
                gpdataframe, minx, maxx, miny, maxy,
                n=grid_n, min_for_outlier=kwargs.get('min_for_outlier', 3),
                path_out=path_out, iterative=kwargs.get('iterative', False),
                break_ratio=kwargs.get('break_ratio', 5.0))

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
                path_out=path_out,
                max_points=kwargs.get('max_points'))


# =============================================================================
# STANDALONE CONVERSION HELPERS
# Pure-math functions with no dependency on engine state.
# =============================================================================

def dip_field_problem(values, dip_col='dip'):
    """
    Sanity-check a field chosen as *dip*: dip must lie between 0 and 90 degrees.

    Returns a message describing the problem, or ``None`` if the values are
    plausible.  A dip field holding values above 90 almost always means the dip
    and dip direction / strike fields have been swapped, which otherwise gives
    plausible-looking but wrong output (the averaging treats e.g. a "dip" of 335
    as a 25 degree dip in the opposite direction).
    """
    v = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    n_bad = int(((v > 90) | (v < 0)).sum())
    if not n_bad:
        return None
    return (f"Dip field '{dip_col}' has {n_bad} of {len(v)} values outside 0-90 degrees "
            f"(range {v.min():g} to {v.max():g}). Are the Dip and Dip Direction/Strike "
            f"fields swapped?")


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

def _contact_segments(geoms):
    """
    Split contact geometries (lines, rings or polygons) into individual segments.

    Returns ``(a, b, azimuth)``: segment start points (n, 2), end points (n, 2) and the
    azimuth of each segment in the range 0-180 degrees (strike convention: the direction
    is ambiguous).  Fully vectorised.
    """
    parts = shapely.get_parts(np.asarray(geoms, dtype=object))
    parts = parts[~shapely.is_missing(parts)]
    types = shapely.get_type_id(parts)
    polys = parts[types == 3]
    lines = parts[(types == 1) | (types == 2)]
    paths = np.concatenate([shapely.get_rings(polys), lines]) if len(polys) else lines
    if len(paths) == 0:
        empty = np.zeros((0, 2))
        return empty, empty, np.zeros(0)
    c, idx = shapely.get_coordinates(paths, return_index=True)
    same = idx[:-1] == idx[1:]
    a, b = c[:-1][same], c[1:][same]
    az = (np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) + 360) % 180
    return a, b, az


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
    a, b, az = _contact_segments(contact_gdf.geometry.values)
    return gpd.GeoDataFrame({'azimuth': az},
                            geometry=shapely.linestrings(np.stack([a, b], axis=1)) if len(a) else [],
                            crs=contact_gdf.crs)


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

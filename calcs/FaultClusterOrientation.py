# Complete Circular K-Means Clustering for Shapefile Trend Data
# Includes all functions and example usage

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point, LineString
import warnings
import os
warnings.filterwarnings('ignore')
showPlots = False

# ============================================================================
# CORE CLUSTERING CLASS
# ============================================================================

class CircularKMeans:
    """
    K-means clustering for non-directional azimuth data.
    
    Non-directional means that 0° and 180° represent the same direction,
    and the distance between 179° and 1° is 2°, not 178°.
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_centers_azimuth_ = None
        self.labels_ = None
        
    def _azimuth_to_cartesian(self, azimuths):
        """Convert non-directional azimuth angles to 2D cartesian coordinates."""
        # Double the angles to handle non-directional nature
        doubled_angles = np.radians(azimuths * 2)
        
        # Convert to cartesian coordinates
        x = np.cos(doubled_angles)
        y = np.sin(doubled_angles)
        
        return np.column_stack([x, y])
    
    def _cartesian_to_azimuth(self, cartesian_coords):
        """Convert 2D cartesian coordinates back to azimuth angles."""
        x, y = cartesian_coords[:, 0], cartesian_coords[:, 1]
        
        # Convert back to angles and halve them
        angles = np.arctan2(y, x)
        azimuths = np.degrees(angles) / 2
        
        # Ensure angles are in [0, 180) range for non-directional data
        azimuths = azimuths % 180
        
        return azimuths
    
    def fit(self, azimuths):
        """Fit k-means clustering to azimuth data."""
        # Normalize input to [0, 180) for non-directional data
        azimuths = np.array(azimuths) % 180
        
        # Convert to cartesian coordinates
        cartesian_coords = self._azimuth_to_cartesian(azimuths)
        
        # Perform k-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                           random_state=self.random_state,
                           n_init=10)
        self.labels_ = self.kmeans.fit_predict(cartesian_coords)
        
        # Convert cluster centers back to azimuth
        self.cluster_centers_azimuth_ = self._cartesian_to_azimuth(
            self.kmeans.cluster_centers_
        )
        
        return self
    
    def predict(self, azimuths):
        """Predict cluster labels for new azimuth data."""
        if self.kmeans is None:
            raise ValueError("Model must be fitted before making predictions")
        
        azimuths = np.array(azimuths) % 180
        cartesian_coords = self._azimuth_to_cartesian(azimuths)
        return self.kmeans.predict(cartesian_coords)
    
    def get_cluster_centers(self):
        """Get cluster centers in azimuth degrees."""
        return self.cluster_centers_azimuth_
    
    def calculate_silhouette_score(self, azimuths):
        """Calculate silhouette score for the clustering."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before calculating silhouette score")
        
        azimuths = np.array(azimuths) % 180
        cartesian_coords = self._azimuth_to_cartesian(azimuths)
        
        if len(np.unique(self.labels_)) > 1:
            return silhouette_score(cartesian_coords, self.labels_)
        else:
            return 0.0

# ============================================================================
# SHAPEFILE PROCESSING FUNCTIONS CLASS
# ============================================================================
class FaultsOrientations:
    def __init__(
        self
    ):
        """
        Initialize the Fault Orientations Cluster Calculation.

        
        """
        pass
    
    def read_trend_from_shapefile(self,shapefile_path, trend_field='trend'):
        """
        Read orientation data from a shapefile.
        
        Parameters:
        -----------
        shapefile_path : str or Path
            Path to the shapefile
        trend_field : str
            Name of the field containing trend/azimuth data (default: 'trend')
        
        Returns:
        --------
        tuple: (geopandas.GeoDataFrame, numpy.array)
            The full GeoDataFrame and extracted trend values
        """
        try:
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            print(f"Successfully loaded shapefile: {shapefile_path}")
            print(f"Number of features: {len(gdf)}")
            print(f"Available fields: {list(gdf.columns)}")
            
            # Check if the trend field exists
            if trend_field not in gdf.columns:
                available_fields = [col for col in gdf.columns if col.lower() != 'geometry']
                raise ValueError(f"Field '{trend_field}' not found. Available fields: {available_fields}")
            
            # Extract trend values and remove NaN values
            trend_data = gdf[trend_field].dropna().values
            
            if len(trend_data) == 0:
                raise ValueError(f"No valid data found in field '{trend_field}'")
            
            # Convert to numeric if needed
            try:
                trend_data = pd.to_numeric(trend_data)
            except ValueError:
                raise ValueError(f"Field '{trend_field}' contains non-numeric data")
            
            print(f"Extracted {len(trend_data)} valid trend measurements")
            print(f"Trend range: {trend_data.min():.1f}° to {trend_data.max():.1f}°")
            
            return gdf, trend_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        except Exception as e:
            raise Exception(f"Error reading shapefile: {str(e)}")

    def find_optimal_clusters(self,azimuths, max_clusters=10):
        """Find optimal number of clusters using silhouette score."""
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(np.unique(azimuths % 180))))
        
        for n_clusters in cluster_range:
            clusterer = CircularKMeans(n_clusters=n_clusters)
            clusterer.fit(azimuths)
            score = clusterer.calculate_silhouette_score(azimuths)
            silhouette_scores.append(score)
        
        # Plot silhouette scores
        if showPlots:
            plt.figure(figsize=(10, 6))
            plt.plot(cluster_range, silhouette_scores, 'bo-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score vs Number of Clusters')
            plt.grid(True, alpha=0.3)
        
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        if showPlots:
            plt.axvline(optimal_clusters, color='red', linestyle='--', 
                    label=f'Optimal: {optimal_clusters} clusters')
            plt.legend()
            plt.show()
        
        return optimal_clusters, silhouette_scores

    def analyze_shapefile_trends(self,shapefile_path, trend_field='trend', n_clusters=None, 
                            max_clusters=10, output_shapefile=None):
        """
        Complete workflow for clustering trend data from a shapefile.
        
        Parameters:
        -----------
        shapefile_path : str or Path
            Path to the input shapefile
        trend_field : str
            Name of the field containing trend/azimuth data
        n_clusters : int, optional
            Number of clusters (if None, will find optimal)
        max_clusters : int
            Maximum number of clusters to test when finding optimal
        output_shapefile : str or Path, optional
            Path to save results as a new shapefile
        
        Returns:
        --------
        dict: Results including cluster labels, centers, and statistics
        """
        
        print("=== Shapefile Trend Analysis ===\n")
        
        # Read the shapefile
        gdf, trend_data = self.read_trend_from_shapefile(shapefile_path, trend_field)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            print("Finding optimal number of clusters...")
            n_clusters, scores = self.find_optimal_clusters(trend_data, max_clusters)
            print(f"Optimal number of clusters: {n_clusters}\n")
        
        # Perform clustering
        print(f"Performing clustering with {n_clusters} clusters...")
        clusterer = CircularKMeans(n_clusters=n_clusters, random_state=42)
        clusterer.fit(trend_data)
        
        labels = clusterer.labels_
        centers = clusterer.get_cluster_centers()
        silhouette = clusterer.calculate_silhouette_score(trend_data)
        
        print(f"Silhouette score: {silhouette:.3f}")
        print("\nCluster centers and statistics:")
        
        results = {
            'gdf': gdf,
            'trend_data': trend_data,
            'labels': labels,
            'centers': centers,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters,
            'clusterer': clusterer
        }
        
        # Calculate statistics for each cluster
        cluster_stats = []
        for i in range(n_clusters):
            mask = labels == i
            cluster_trends = trend_data[mask]
            stats = {
                'cluster_id': i,
                'center': centers[i],
                'count': np.sum(mask),
                'percentage': (np.sum(mask) / len(labels)) * 100,
                'std_dev': np.std(cluster_trends),
                'min_trend': np.min(cluster_trends),
                'max_trend': np.max(cluster_trends)
            }
            cluster_stats.append(stats)
            
            print(f"  Cluster {i}: {centers[i]:.1f}° ({stats['count']} points, "
                f"{stats['percentage']:.1f}%, σ={stats['std_dev']:.1f}°)")
        
        results['cluster_stats'] = cluster_stats
        
        # Add cluster labels to the GeoDataFrame
        valid_indices = gdf[trend_field].dropna().index
        gdf_with_clusters = gdf.copy()
        gdf_with_clusters['cluster'] = np.nan
        gdf_with_clusters.loc[valid_indices, 'cluster'] = labels
        gdf_with_clusters['cluster_center'] = np.nan
        gdf_with_clusters.loc[valid_indices, 'cluster_center'] = [centers[label] for label in labels]
        
        results['gdf_with_clusters'] = gdf_with_clusters
        
        # Save results if requested
        if output_shapefile:
            gdf_with_clusters.to_file(output_shapefile)
            print(f"\nResults saved to: {output_shapefile}")
        
        if showPlots:
            # Plot results
            print("\nGenerating plots...")
            self.plot_circular_clusters(trend_data, labels, centers, 
                                title=f"Trend Clustering: {Path(shapefile_path).name}")
            self.plot_spatial_clusters(gdf_with_clusters, trend_field)
        
        return results

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def plot_circular_clusters(self,azimuths, labels, cluster_centers, title="Circular K-Means Clustering"):
        """Plot the clustering results on a circular plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Circular plot - FIXED FOR NON-DIRECTIONAL DATA
        ax1 = plt.subplot(121, projection='polar')
        
        # For non-directional data, normalize to [0, 180) and create symmetric display
        azimuths_180 = azimuths % 180
        centers_180 = cluster_centers % 180
        
        # Convert to radians
        azimuths_rad = np.radians(azimuths_180)
        centers_rad = np.radians(centers_180)
        
        # Create colors for clusters
        colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
        
        # Plot data points - both directions for symmetry
        for i, color in enumerate(colors):
            mask = labels == i
            cluster_angles = azimuths_rad[mask]
            
            # Plot original directions
            ax1.scatter(cluster_angles, np.ones(np.sum(mask)), 
                    c=[color], label=f'Cluster {i}', s=50, alpha=0.7)
            # Plot symmetric directions (+ 180°)
            ax1.scatter(cluster_angles + np.pi, np.ones(np.sum(mask)), 
                    c=[color], s=50, alpha=0.7)
        
        # Plot cluster centers - both directions
        ax1.scatter(centers_rad, np.ones(len(centers_rad)), 
                c='red', marker='x', s=200, linewidths=3, label='Centers')
        ax1.scatter(centers_rad + np.pi, np.ones(len(centers_rad)), 
                c='red', marker='x', s=200, linewidths=3)
        
        ax1.set_title('Circular Plot (Non-Directional, Symmetric)')
        ax1.set_ylim(0, 1.5)
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Linear histogram
        ax2.hist([azimuths[labels == i] for i in range(len(np.unique(labels)))], 
                bins=20, alpha=0.7, label=[f'Cluster {i}' for i in range(len(np.unique(labels)))],
                color=colors)
        
        # Plot cluster centers as vertical lines
        for i, center in enumerate(cluster_centers):
            ax2.axvline(center, color='red', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Azimuth (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Linear Histogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_spatial_clusters(self,gdf_with_clusters, trend_field='trend'):
        """Create a spatial map showing the clustered trend data."""
        if 'cluster' not in gdf_with_clusters.columns:
            raise ValueError("GeoDataFrame must contain 'cluster' column")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Spatial plot colored by cluster
        gdf_with_clusters.plot(column='cluster', categorical=True, 
                            legend=True, ax=ax1, markersize=50, 
                            cmap='Set1', missing_kwds={'color': 'lightgray'})
        ax1.set_title('Spatial Distribution of Trend Clusters')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        
        # Spatial plot colored by trend value
        gdf_with_clusters.plot(column=trend_field, ax=ax2, markersize=50,
                            cmap='viridis', legend=True)
        ax2.set_title(f'Spatial Distribution of {trend_field.title()} Values')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        plt.show()

    # ============================================================================
    # EXAMPLE USAGE FUNCTIONS
    # ============================================================================

    def example_basic_usage(self):
        """Example showing basic usage with automatic cluster detection."""
        
        print("EXAMPLE 1: Basic Usage")
        print("=" * 40)
        
        # Replace with your actual shapefile path
        shapefile_path = 'structural_measurements.shp'  # CHANGE THIS TO YOUR FILE
        
        try:
            # Basic usage - automatically finds optimal number of clusters
            results = self.analyze_shapefile_trends(
                shapefile_path=shapefile_path,
                trend_field='trend'  # or 'azimuth', 'strike', 'bearing', etc.
            )
            
            # Access the results
            gdf_with_clusters = results['gdf_with_clusters']
            cluster_centers = results['centers']
            silhouette_score = results['silhouette_score']
            n_clusters = results['n_clusters']
            
            print(f"\nResults Summary:")
            print(f"- Number of clusters found: {n_clusters}")
            print(f"- Silhouette score: {silhouette_score:.3f}")
            print(f"- Cluster centers: {cluster_centers}")
            
            # Print cluster statistics
            for stats in results['cluster_stats']:
                print(f"Cluster {stats['cluster_id']}: {stats['count']} points "
                    f"around {stats['center']:.1f}° (±{stats['std_dev']:.1f}°)")
            
            return results
            
        except FileNotFoundError:
            print(f"Error: Shapefile '{shapefile_path}' not found.")
            print("Please update the shapefile_path variable with your actual file path.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def example_custom_parameters(self):
        """Example with custom parameters and saving results."""
        
        print("\nEXAMPLE 2: Custom Parameters")
        print("=" * 40)
        
        shapefile_path = 'structural_measurements.shp'  # CHANGE THIS
        
        try:
            # Custom usage with specific number of clusters and output file
            results = self.analyze_shapefile_trends(
                shapefile_path=shapefile_path,
                trend_field='trend',                    # Field containing your azimuth data
                n_clusters=4,                          # Force 4 clusters instead of auto-detect
                max_clusters=8,                        # Max clusters to test if n_clusters=None
                output_shapefile='clustered_trends.shp' # Save results to new shapefile
            )
            
            print("Results saved to 'clustered_trends.shp'")
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def create_custom_plots(self,gdf, trend_data, labels, centers):
        """Create comprehensive custom visualization plots."""
        
        print("Creating detailed custom plots...")
        
        # Create a comprehensive plot with 4 subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Rose diagram (circular histogram) - FIXED FOR NON-DIRECTIONAL DATA
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        
        # Create symmetric rose diagram for non-directional data
        # Normalize to [0, 180) since data is non-directional
        trend_data_180 = trend_data % 180
        
        n_bins = 18  # 18 bins for 180°, each bin = 10°
        hist, bin_edges = np.histogram(trend_data_180, bins=n_bins, range=(0, 180))
        
        # Convert bin edges to radians and create symmetric display
        bin_centers_deg = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_rad = np.radians(bin_centers_deg)
        width = np.pi / n_bins  # Width for 180° range
        
        # Plot both directions to create symmetric rose diagram
        for height, angle in zip(hist, bin_centers_rad):
            # Plot original direction
            ax1.bar(angle, height, width=width, alpha=0.7, color='skyblue')
            # Plot symmetric direction (angle + 180°)
            ax1.bar(angle + np.pi, height, width=width, alpha=0.7, color='skyblue')
        
        ax1.set_title('Rose Diagram - All Data (Symmetric)')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        
        # 2. Cluster rose diagrams - FIXED FOR NON-DIRECTIONAL DATA
        ax2 = plt.subplot(2, 3, 2, projection='polar')
        colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
        
        for i, color in enumerate(colors):
            cluster_data_180 = trend_data[labels == i] % 180
            hist, bin_edges = np.histogram(cluster_data_180, bins=n_bins, range=(0, 180))
            
            for j, (height, angle) in enumerate(zip(hist, bin_centers_rad)):
                if height > 0:  # Only plot if there's data
                    # Plot both directions for symmetry
                    ax2.bar(angle, height, width=width, alpha=0.6, 
                        color=color, label=f'Cluster {i}' if j == 0 else "")
                    ax2.bar(angle + np.pi, height, width=width, alpha=0.6, color=color)
        
        ax2.set_title('Rose Diagram - By Cluster (Symmetric)')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.legend(bbox_to_anchor=(1.1, 1))
        
        # 3. Box plot of clusters
        ax3 = plt.subplot(2, 3, 3)
        cluster_data = [trend_data[labels == i] for i in range(len(np.unique(labels)))]
        bp = ax3.boxplot(cluster_data, labels=[f'C{i}' for i in range(len(cluster_data))],
                        patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Trend (degrees)')
        ax3.set_xlabel('Cluster')
        ax3.set_title('Trend Distribution by Cluster')
        ax3.grid(True, alpha=0.3)
        
        # 4. Spatial map with trend directions
        ax4 = plt.subplot(2, 3, 4)
        
        # Plot points colored by cluster
        gdf.plot(column='cluster', categorical=True, ax=ax4, 
                markersize=60, cmap='Set1', alpha=0.7,
                missing_kwds={'color': 'lightgray'})
        
        ax4.set_title('Spatial Distribution by Cluster')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        
        # 5. Cluster center comparison
        ax5 = plt.subplot(2, 3, 5)
        bars = ax5.bar(range(len(centers)), centers, color=colors, alpha=0.7)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Center Trend (degrees)')
        ax5.set_title('Cluster Centers')
        ax5.set_xticks(range(len(centers)))
        ax5.set_xticklabels([f'C{i}' for i in range(len(centers))])
        
        # Add value labels on bars
        for bar, center in zip(bars, centers):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{center:.1f}°', ha='center', va='bottom')
        
        ax5.grid(True, alpha=0.3)
        
        # 6. Cluster size pie chart
        ax6 = plt.subplot(2, 3, 6)
        cluster_sizes = [np.sum(labels == i) for i in range(len(centers))]
        wedges, texts, autotexts = ax6.pie(cluster_sizes, labels=[f'C{i}' for i in range(len(centers))],
                                        autopct='%1.1f%%', colors=colors, startangle=90)
        ax6.set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        plt.show()

    def example_custom_analysis(self,results):
        """Example showing how to work with results and create custom plots."""
        
        if results is None:
            print("No results to analyze - check your shapefile path")
            return
        
        print("\nEXAMPLE 3: Custom Analysis and Plotting")
        print("=" * 40)
        
        gdf = results['gdf_with_clusters']
        trend_data = results['trend_data']
        labels = results['labels']
        centers = results['centers']
        
        # Statistical analysis
        print("Statistical Analysis:")
        print(f"Total measurements: {len(trend_data)}")
        print(f"Trend range: {trend_data.min():.1f}° to {trend_data.max():.1f}°")
        print(f"Mean trend: {trend_data.mean():.1f}°")
        print(f"Standard deviation: {trend_data.std():.1f}°")
        
        # Analyze each cluster
        print("\nDetailed Cluster Analysis:")
        for i in range(results['n_clusters']):
            cluster_trends = trend_data[labels == i]
            print(f"Cluster {i}:")
            print(f"  Center: {centers[i]:.1f}°")
            print(f"  Count: {len(cluster_trends)} ({len(cluster_trends)/len(trend_data)*100:.1f}%)")
            print(f"  Range: {cluster_trends.min():.1f}° to {cluster_trends.max():.1f}°")
            print(f"  Std Dev: {cluster_trends.std():.1f}°")
        
        # Create custom plots
        if showPlots:
            self.create_custom_plots(gdf, trend_data, labels, centers)
        
        # Create summary table
        print("\nCluster Summary Table:")
        print("=" * 70)
        df_stats = pd.DataFrame(results['cluster_stats'])
        df_display = df_stats.copy()
        df_display['center'] = df_display['center'].round(1)
        df_display['percentage'] = df_display['percentage'].round(1)
        df_display['std_dev'] = df_display['std_dev'].round(1)
        print(df_display.to_string(index=False))

    def add_endpoint_azimuth(self,shapefile_path, output_path=None, azimuth_field='end_azimuth'):
        """
        Load a polyline shapefile and add a field with the azimuth of the line 
        joining each polyline's endpoints, normalized to 0-180 degrees.
        
        Parameters:
        -----------
        shapefile_path : str
            Path to the input polyline shapefile
        output_path : str, optional
            Path to save the modified shapefile. If None, returns GeoDataFrame only
        azimuth_field : str
            Name of the new field to store azimuth values (default: 'end_azimuth')
        
        Returns:
        --------
        geopandas.GeoDataFrame
            GeoDataFrame with the new azimuth field added
        """
        
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        def calculate_endpoint_azimuth(geometry):
            """
            Calculate azimuth between endpoints of a LineString or MultiLineString.
            Returns azimuth normalized to 0-180 degrees.
            """
            if geometry is None or geometry.is_empty:
                return np.nan
            
            try:
                # Handle both LineString and MultiLineString
                if geometry.geom_type == 'LineString':
                    coords = list(geometry.coords)
                    if len(coords) < 2:
                        return np.nan
                    first_point = coords[0]
                    last_point = coords[-1]
                    
                elif geometry.geom_type == 'MultiLineString':
                    # For MultiLineString, use first point of first line 
                    # and last point of last line
                    lines = list(geometry.geoms)
                    if len(lines) == 0:
                        return np.nan
                    first_coords = list(lines[0].coords)
                    last_coords = list(lines[-1].coords)
                    if len(first_coords) == 0 or len(last_coords) == 0:
                        return np.nan
                    first_point = first_coords[0]
                    last_point = last_coords[-1]
                    
                else:
                    # Not a line geometry
                    return np.nan
                
                # Calculate azimuth from first to last point
                dx = last_point[0] - first_point[0]
                dy = last_point[1] - first_point[1]
                
                # Handle case where endpoints are the same
                if dx == 0 and dy == 0:
                    return np.nan
                
                # Calculate azimuth in radians, then convert to degrees
                # atan2 returns values from -π to π
                azimuth_rad = np.arctan2(dx, dy)
                azimuth_deg = np.degrees(azimuth_rad)
                
                # Convert to 0-360 range
                if azimuth_deg < 0:
                    azimuth_deg += 360
                
                # Normalize to 0-180 range (undirected)
                # Since we don't care about direction, azimuth and azimuth+180 are equivalent
                azimuth_deg = azimuth_deg % 180.0
                
                return azimuth_deg
                
            except Exception as e:
                print(f"Error calculating azimuth: {e}")
                return np.nan
        
        # Calculate azimuth for each polyline
        gdf[azimuth_field] = gdf['geometry'].apply(calculate_endpoint_azimuth)
        
        # Save if output path is provided
        if output_path:
            gdf.to_file(output_path)
            print(f"Shapefile with azimuth field saved to: {output_path}")
        
        # Print summary statistics
        valid_azimuths = gdf[azimuth_field].dropna()
        print(f"\nAzimuth statistics:")
        print(f"  Total features: {len(gdf)}")
        print(f"  Valid azimuths: {len(valid_azimuths)}")
        print(f"  Invalid/null: {len(gdf) - len(valid_azimuths)}")
        if len(valid_azimuths) > 0:
            print(f"  Min azimuth: {valid_azimuths.min():.2f}°")
            print(f"  Max azimuth: {valid_azimuths.max():.2f}°")
            print(f"  Mean azimuth: {valid_azimuths.mean():.2f}°")
        
        return gdf








    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    def demonstrate_nondirectional_symmetry(self):
        """Demonstrate why non-directional data should create symmetric rose diagrams."""
        
        print("DEMONSTRATING NON-DIRECTIONAL SYMMETRY")
        print("=" * 40)
        
        # Create example data: lines at 30°, 45°, 60°
        # In non-directional terms, these are the same as 210°, 225°, 240°
        example_trends = np.array([30, 45, 60, 210, 225, 240])
        
        print("Example trend data (degrees):")
        print(f"Original measurements: {example_trends}")
        print(f"Normalized to [0,180): {example_trends % 180}")
        print()
        print("For non-directional data:")
        print("- 30° line is the same as 210° line")
        print("- 45° line is the same as 225° line") 
        print("- 60° line is the same as 240° line")
        print()
        print("Therefore, rose diagrams should show symmetric peaks!")
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
        
        # WRONG WAY (old method) - doubling angles
        ax1.set_title('WRONG: Doubled Angles (Asymmetric)', pad=20)
        angles_wrong = np.radians(example_trends * 2)
        hist_wrong, bins_wrong = np.histogram(angles_wrong, bins=36, range=(0, 4*np.pi))
        width_wrong = 2 * np.pi / 36
        
        for height, angle in zip(hist_wrong, bins_wrong[:-1]):
            if height > 0:
                ax1.bar(angle, height, width=width_wrong, alpha=0.7, color='red')
        
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        
        # CORRECT WAY - symmetric display
        ax2.set_title('CORRECT: Symmetric Display', pad=20)
        trends_180 = example_trends % 180
        hist_correct, bins_correct = np.histogram(trends_180, bins=18, range=(0, 180))
        bin_centers = (bins_correct[:-1] + bins_correct[1:]) / 2
        bin_centers_rad = np.radians(bin_centers)
        width_correct = np.pi / 18
        
        for height, angle in zip(hist_correct, bin_centers_rad):
            if height > 0:
                # Plot both directions for symmetry
                ax2.bar(angle, height, width=width_correct, alpha=0.7, color='green')
                ax2.bar(angle + np.pi, height, width=width_correct, alpha=0.7, color='green')
        
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        
        plt.tight_layout()
        plt.show()
        
        return example_trends

    def example_manual_clusters(self,shapefile_path,layer_name,azimuth_field='endpt_az'):
        """Example showing how to define specific number of clusters."""
        
        print("EXAMPLE: Manual Cluster Definition")
        print("=" * 40)
        
        #shapefile_path = '100k_IGB_faults_clip2.shp'  # CHANGE THIS
        
        try:
            # Method 1: Define specific number of clusters
            """ print("Method 1: Using 3 clusters")
            results_3 = self.analyze_shapefile_trends(
                shapefile_path=shapefile_path,
                trend_field=azimuth_field,
                n_clusters=3  # Define exactly 3 clusters
            )"""
            
            # Method 2: Compare different cluster numbers
            print("\nMethod 2: Comparing different cluster numbers")
            cluster_numbers = [2, 3, 4, 5, 6, 7]
            silhouette_scores = []
            
            for n in cluster_numbers:
                layer_path = os.path.dirname(shapefile_path)
                new_path = (
                    layer_path
                    + "/"
                    + layer_name
                    + f"_fault_clusters_{n}.shp"
                ) 
                results = self.analyze_shapefile_trends(
                    shapefile_path=shapefile_path,
                    trend_field=azimuth_field,
                    n_clusters=n, output_shapefile=new_path
                )
                silhouette_scores.append(results['silhouette_score'])
                print(f"{n} clusters - Silhouette score: {results['silhouette_score']:.3f}")
            
            best_idx = np.argmax(silhouette_scores)
            best_n = cluster_numbers[best_idx]
            print(f"\nBest number of clusters: {best_n} (Silhouette score: {silhouette_scores[best_idx]:.3f})")
            
            # Plot comparison
            if showPlots:
                plt.figure(figsize=(10, 6))
                plt.plot(cluster_numbers, silhouette_scores, 'bo-', linewidth=2, markersize=8)
                plt.xlabel('Number of Clusters')
                plt.ylabel('Silhouette Score')
                plt.title('Manual Cluster Comparison')
                plt.grid(True, alpha=0.3)
                
                # Highlight best
                plt.axvline(best_n, color='red', linestyle='--', 
                        label=f'Best: {best_n} clusters (score: {silhouette_scores[best_idx]:.3f})')
                plt.legend()
                plt.show()
            
            return best_n
            
        except Exception as e:
            print(f"Error: {e}")
            return None


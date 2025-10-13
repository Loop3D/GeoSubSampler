import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import numpy as np
from collections import defaultdict
from shapely.ops import split

class FaultsGraph:
    def __init__(
        self
    ):
        """
        Initialize the Fault Strat Offset Calculation.

        
        """
        pass
    
    def polylines_to_network_graph(self,gdf, tolerance=1e-8):
        """
        Convert a shapefile of polylines into a NetworkX graph.
        
        Parameters:
        -----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing polylines
        tolerance : float
            Tolerance for considering vertices as identical (default: 1e-8)
            
        Returns:
        --------
        tuple: (networkx.Graph, geopandas.GeoDataFrame)
            Network graph with nodes at intersections and endpoints, and original GeoDataFrame
        """
        
        # Ensure we're working with LineString geometries
        gdf = gdf[gdf.geometry.type == 'LineString']
        
        # Create a graph
        G = nx.Graph()
        
        # Dictionary to store all vertices and their frequencies
        vertex_count = defaultdict(int)
        
        # First pass: collect all vertices and count occurrences
        print("Collecting vertices...")
        for idx, row in gdf.iterrows():
            line = row.geometry
            coords = list(line.coords)
            
            for coord in coords:
                # Round coordinates to handle floating point precision
                rounded_coord = (round(coord[0], 10), round(coord[1], 10))
                vertex_count[rounded_coord] += 1
        
        # Identify intersection points (vertices shared by multiple lines) and endpoints
        intersection_points = set()
        endpoint_candidates = set()
        
        for coord, count in vertex_count.items():
            if count > 2:  # Definitely an intersection
                intersection_points.add(coord)
            elif count == 2:  # Could be a simple connection or intersection
                endpoint_candidates.add(coord)
            else:  # count == 1, definitely an endpoint
                intersection_points.add(coord)
        
        # Add endpoint candidates to intersections (they're either endpoints or connection points)
        intersection_points.update(endpoint_candidates)
        
        # Create nodes for all intersection points
        node_id = 0
        coord_to_node = {}
        
        for coord in intersection_points:
            G.add_node(node_id, x=coord[0], y=coord[1], pos=coord)
            coord_to_node[coord] = node_id
            node_id += 1
        
        print(f"Created {len(intersection_points)} nodes")
        
        # Second pass: create edges between intersection points
        print("Creating edges...")
        edge_id = 1  # Start edge IDs from 1
        
        for idx, row in gdf.iterrows():
            line = row.geometry
            coords = list(line.coords)
            
            # Find intersection points along this line
            line_intersections = []
            for i, coord in enumerate(coords):
                rounded_coord = (round(coord[0], 10), round(coord[1], 10))
                if rounded_coord in intersection_points:
                    line_intersections.append((i, rounded_coord))
            
            # Create edges between consecutive intersection points
            for i in range(len(line_intersections) - 1):
                start_idx, start_coord = line_intersections[i]
                end_idx, end_coord = line_intersections[i + 1]
                
                start_node = coord_to_node[start_coord]
                end_node = coord_to_node[end_coord]
                
                # Calculate edge attributes
                segment_coords = coords[start_idx:end_idx + 1]
                segment_line = LineString(segment_coords)
                edge_length = segment_line.length
                
                # Add edge with attributes including unique edge ID
                edge_attrs = {
                    'ID': edge_id,  # Unique edge ID
                    'length': edge_length,
                    'geometry': segment_line,
                    'original_feature_id': idx,
                    'start_node': start_node,
                    'end_node': end_node
                }
                
                # Add any additional attributes from the original shapefile
                for col in gdf.columns:
                    if col != 'geometry':
                        edge_attrs[col] = row[col]
                
                G.add_edge(start_node, end_node, **edge_attrs)
                edge_id += 1  # Increment for next edge
        
        print(f"Created {G.number_of_edges()} edges with unique IDs")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gdf


    def add_centrality_measures(self,G, weight=None, normalized=True):
        """
        Add betweenness centrality and load centrality to nodes in a NetworkX graph.
        
        Parameters:
        -----------
        G : networkx.Graph or networkx.MultiGraph
            The input graph
        weight : str, optional
            Edge attribute to use as weight for centrality calculations (default: 'length')
            Set to None for unweighted calculations
        normalized : bool, optional
            Whether to normalize centrality values (default: True)
            
        Returns:
        --------
        networkx.Graph or networkx.MultiGraph
            New graph with centrality measures added as node attributes
        """
        print(f"\n=== Adding Centrality Measures ===")
        print(f"Graph type: {type(G).__name__}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"Weight attribute: {weight}")
        print(f"Normalized: {normalized}")
        
        # Create a copy of the graph to avoid modifying the original
        G_with_centrality = G.copy()
        
        # Check if the graph is connected
        is_connected = nx.is_connected(G)
        print(f"Graph is connected: {is_connected}")
        
        if not is_connected:
            components = list(nx.connected_components(G))
            print(f"Found {len(components)} connected components")
            largest_component_size = max(len(comp) for comp in components)
            print(f"Largest component has {largest_component_size} nodes")
        
        try:
            # Calculate betweenness centrality
            print("Calculating betweenness centrality...")
            betweenness = nx.betweenness_centrality(
                G, 
                weight=weight, 
                normalized=normalized,
                endpoints=False  # Don't include endpoints in path counts
            )
            
            # Add betweenness centrality to node attributes
            for node, centrality_value in betweenness.items():
                G_with_centrality.nodes[node]['betweenness_centrality'] = centrality_value
            
            print(f"✓ Betweenness centrality calculated for {len(betweenness)} nodes")
            if betweenness:
                max_betweenness = max(betweenness.values())
                min_betweenness = min(betweenness.values())
                avg_betweenness = sum(betweenness.values()) / len(betweenness)
                print(f"  Range: {min_betweenness:.6f} to {max_betweenness:.6f}")
                print(f"  Average: {avg_betweenness:.6f}")
            
        except Exception as e:
            print(f"Error calculating betweenness centrality: {e}")
            # Add zero values for all nodes
            for node in G_with_centrality.nodes():
                G_with_centrality.nodes[node]['betweenness_centrality'] = 0.0
        
        try:
            # Calculate load centrality
            print("Calculating load centrality...")
            load = nx.load_centrality(
                G, 
                weight=weight, 
                normalized=normalized
            )
            
            # Add load centrality to node attributes
            for node, centrality_value in load.items():
                G_with_centrality.nodes[node]['load_centrality'] = centrality_value
            
            print(f"✓ Load centrality calculated for {len(load)} nodes")
            if load:
                max_load = max(load.values())
                min_load = min(load.values())
                avg_load = sum(load.values()) / len(load)
                print(f"  Range: {min_load:.6f} to {max_load:.6f}")
                print(f"  Average: {avg_load:.6f}")
            
        except Exception as e:
            print(f"Error calculating load centrality: {e}")
            # Add zero values for all nodes
            for node in G_with_centrality.nodes():
                G_with_centrality.nodes[node]['load_centrality'] = 0.0
        
        # Find nodes with highest centrality values
        try:
            betweenness_values = [(node, data.get('betweenness_centrality', 0)) 
                                for node, data in G_with_centrality.nodes(data=True)]
            load_values = [(node, data.get('load_centrality', 0)) 
                        for node, data in G_with_centrality.nodes(data=True)]
            
            # Sort by centrality values
            top_betweenness = sorted(betweenness_values, key=lambda x: x[1], reverse=True)[:5]
            top_load = sorted(load_values, key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\nTop 5 nodes by betweenness centrality:")
            for i, (node, value) in enumerate(top_betweenness, 1):
                node_degree = G.degree(node)
                print(f"  {i}. Node {node}: {value:.6f} (degree: {node_degree})")
            
            print(f"\nTop 5 nodes by load centrality:")
            for i, (node, value) in enumerate(top_load, 1):
                node_degree = G.degree(node)
                print(f"  {i}. Node {node}: {value:.6f} (degree: {node_degree})")
                
        except Exception as e:
            print(f"Error in centrality analysis: {e}")
        
        # Add some summary statistics as graph attributes
        try:
            betweenness_stats = {
                'max_betweenness': max(betweenness.values()) if betweenness else 0,
                'min_betweenness': min(betweenness.values()) if betweenness else 0,
                'avg_betweenness': sum(betweenness.values()) / len(betweenness) if betweenness else 0
            }
            
            load_stats = {
                'max_load': max(load.values()) if load else 0,
                'min_load': min(load.values()) if load else 0,
                'avg_load': sum(load.values()) / len(load) if load else 0
            }
            
            # Add to graph attributes
            G_with_centrality.graph.update(betweenness_stats)
            G_with_centrality.graph.update(load_stats)
            G_with_centrality.graph['centrality_weight'] = weight
            G_with_centrality.graph['centrality_normalized'] = normalized
            
        except Exception as e:
            print(f"Error adding summary statistics: {e}")
        
        print(f"\n✓ Centrality measures added successfully")
        return G_with_centrality


    def save_graph_to_shapefile(self,G, output_path_nodes, output_path_edges, original_gdf=None, prefix=""):
        """
        Save the network graph back to shapefiles (nodes and edges).
        Handles both regular Graph and MultiGraph types.
        
        Parameters:
        -----------
        G : networkx.Graph or networkx.MultiGraph
            The network graph
        output_path_nodes : str
            Path for output nodes shapefile
        output_path_edges : str
            Path for output edges shapefile
        original_gdf : GeoDataFrame, optional
            Original GeoDataFrame to preserve CRS
        prefix : str, optional
            Prefix for output files to distinguish different graph types
        """
        
        # Add prefix to filenames if provided
        if prefix:
            import os
            nodes_dir = os.path.dirname(output_path_nodes)
            nodes_name = os.path.basename(output_path_nodes)
            edges_dir = os.path.dirname(output_path_edges)
            edges_name = os.path.basename(output_path_edges)
            
            output_path_nodes = os.path.join(nodes_dir, f"{prefix}_{nodes_name}")
            output_path_edges = os.path.join(edges_dir, f"{prefix}_{edges_name}")
        
        # Create nodes GeoDataFrame
        nodes_data = []
        for node_id, data in G.nodes(data=True):
            nodes_data.append({
                'node_id': node_id,
                'x': data['x'],
                'y': data['y'],
                'degree': G.degree(node_id),
                'geometry': Point(data['x'], data['y'])
            })
        
        nodes_gdf = gpd.GeoDataFrame(nodes_data)
        
        # Set CRS if available from original data
        if original_gdf is not None and original_gdf.crs is not None:
            nodes_gdf.crs = original_gdf.crs
        
        nodes_gdf.to_file(output_path_nodes)
        
        # Create edges GeoDataFrame - handle MultiGraph differently
        edges_data = []
        
        if isinstance(G, nx.MultiGraph):

            # MultiGraph - edges have keys
            for start, end, key, data in G.edges(data=True, keys=True):
                edge_dict = {
                    'start_node': start,
                    'end_node': end,
                    'edge_key': key,  # Include the edge key for MultiGraph
                    'length': data.get('length', 0),
                    'geometry': data.get('geometry', LineString([
                        (G.nodes[start]['x'], G.nodes[start]['y']),
                        (G.nodes[end]['x'], G.nodes[end]['y'])
                    ]))
                }
                
                # Handle different types of edges (detailed vs simplified)
                if 'original_feature_ids' in data:
                    # This is a simplified edge
                    edge_dict['seg_count'] = data.get('segment_count', 1)
                    edge_dict['orig_ids'] = str(data.get('original_feature_ids', []))[:254]  # Truncate for shapefile
                else:
                    # This is a detailed edge
                    edge_dict['orig_id'] = data.get('original_feature_id', -1)
                    edge_dict['seg_count'] = 1
                
                # Add other edge attributes with shortened names for shapefile compatibility
                for attr_key, value in data.items():
                    if attr_key not in ['geometry', 'original_feature_id', 'original_feature_ids', 
                                'segment_count', 'length', 'original_lengths']:
                        # Shorten field names to 10 characters for shapefile compatibility
                        short_key = attr_key[:10] if len(attr_key) > 10 else attr_key
                        
                        # Handle different data types
                        if isinstance(value, (list, tuple)):
                            edge_dict[short_key] = str(value)[:254]  # Convert to string and truncate
                        elif isinstance(value, (int, float)):
                            edge_dict[short_key] = value
                        else:
                            edge_dict[short_key] = str(value)[:254] if value is not None else None
                edges_data.append(edge_dict)
        else:
            # Regular Graph - no edge keys
            for start, end, data in G.edges(data=True):
                edge_dict = {
                    'start_node': start,
                    'end_node': end,
                    'length': data.get('length', 0),
                    'geometry': data.get('geometry', LineString([
                        (G.nodes[start]['x'], G.nodes[start]['y']),
                        (G.nodes[end]['x'], G.nodes[end]['y'])
                    ]))
                }
                
                # Handle different types of edges (detailed vs simplified)
                if 'original_feature_ids' in data:
                    # This is a simplified edge
                    edge_dict['seg_count'] = data.get('segment_count', 1)
                    edge_dict['orig_ids'] = str(data.get('original_feature_ids', []))[:254]  # Truncate for shapefile
                else:
                    # This is a detailed edge
                    edge_dict['orig_id'] = data.get('original_feature_id', -1)
                    edge_dict['seg_count'] = 1
                
                # Add other edge attributes with shortened names for shapefile compatibility
                for attr_key, value in data.items():
                    if attr_key not in ['geometry', 'original_feature_id', 'original_feature_ids', 
                                'segment_count', 'length', 'original_lengths']:
                        # Shorten field names to 10 characters for shapefile compatibility
                        short_key = attr_key[:10] if len(attr_key) > 10 else attr_key
                        
                        # Handle different data types
                        if isinstance(value, (list, tuple)):
                            edge_dict[short_key] = str(value)[:254]  # Convert to string and truncate
                        elif isinstance(value, (int, float)):
                            edge_dict[short_key] = value
                        else:
                            edge_dict[short_key] = str(value)[:254] if value is not None else None
                

                edges_data.append(edge_dict)
        
        edges_gdf = gpd.GeoDataFrame(edges_data)
        
        # Set CRS if available from original data
        if original_gdf is not None and original_gdf.crs is not None:
            edges_gdf.crs = original_gdf.crs
        
        edges_gdf.to_file(output_path_edges)
        
        print(f"Saved {len(nodes_data)} nodes to {output_path_nodes}")
        print(f"Saved {len(edges_data)} edges to {output_path_edges}")

    def analyze_graph(self,G):
        """
        Print basic analysis of the network graph.
        
        Parameters:
        -----------
        G : networkx.Graph
            The network graph to analyze
        """
        print("\n=== Network Graph Analysis ===")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Is connected: {nx.is_connected(G)}")
        
        if G.number_of_nodes() > 0:
            # Connected components analysis
            components = list(nx.connected_components(G))
            print(f"Number of connected components: {len(components)}")
            if len(components) > 1:
                component_sizes = [len(comp) for comp in components]
                print(f"Largest component size: {max(component_sizes)} nodes")
                print(f"Smallest component size: {min(component_sizes)} nodes")
            
            # Degree statistics
            degrees = [G.degree(n) for n in G.nodes()]
            print(f"Average degree: {np.mean(degrees):.2f}")
            print(f"Max degree: {max(degrees)}")
            
            # Find nodes with degree > 2 (true intersections)
            intersections = [n for n in G.nodes() if G.degree(n) > 2]
            print(f"Number of intersections (degree > 2): {len(intersections)}")
            
            # Find endpoints (degree = 1)
            endpoints = [n for n in G.nodes() if G.degree(n) == 1]
            print(f"Number of endpoints (degree = 1): {len(endpoints)}")
            
            # Find isolated nodes (degree = 0)
            isolated = [n for n in G.nodes() if G.degree(n) == 0]
            if isolated:
                print(f"Number of isolated nodes: {len(isolated)}")


    def create_simplified_graph_full(self,G):
        """
        Create a simplified graph keeping only intersection nodes (degree > 2) and endpoints (degree = 1).
        Intermediate nodes (degree = 2) are removed and their edges are merged.
        Uses MultiGraph to handle multiple edges between same nodes.
        Edge attributes match those from create_simplified_graph_simple.
        
        Parameters:
        -----------
        G : networkx.Graph
            The detailed network graph
            
        Returns:
        --------
        networkx.MultiGraph
            Simplified multigraph with only key nodes
        """
        print("\n=== Creating Simplified Graph ===")
        
        # Create a MultiGraph to handle multiple edges between same nodes
        G_simplified = nx.MultiGraph()
        
        # Copy all key nodes to the simplified graph
        nodes_to_keep = set()
        degree_stats = defaultdict(int)
        
        for node in G.nodes():
            degree = G.degree(node)
            degree_stats[degree] += 1
            if degree != 2:  # Keep intersections (degree > 2), endpoints (degree = 1), and isolated (degree = 0)
                nodes_to_keep.add(node)
        
        print(f"Node degree distribution: {dict(degree_stats)}")
        print(f"Keeping {len(nodes_to_keep)} key nodes out of {G.number_of_nodes()} total nodes")
        
        # Add all key nodes to simplified graph
        for node in nodes_to_keep:
            G_simplified.add_node(node, **G.nodes[node])
        
        # Find all chains and process them
        # Only intermediate degree-2 nodes should be marked as processed
        # Key nodes should never be marked as processed so they can serve as start points for multiple chains
        processed_nodes = set()
        
        # First, handle all edges that connect key nodes directly
        direct_edges_count = 0
        for start, end, data in G.edges(data=True):
            if start in nodes_to_keep and end in nodes_to_keep:
                # Direct edge between key nodes - preserve it with simplified format
                start_type = self.classify_node_type(G, start)
                end_type = self.classify_node_type(G, end)
                edge_type = self.create_edge_type(start_type, end_type)
                
                simplified_edge_data = {
                    'line_id': data.get('ID', 'unknown'),
                    'length': data.get('length', 0),
                    'edge_type': edge_type,
                    'start_node': start,
                    'end_node': end,
                    'st_type': start_type,
                    'end_type': end_type,
                    'geometry': data.get('geometry')  # Keep geometry for full graph
                }
                
                # Preserve any other attributes (but exclude unwanted ones)
                unwanted_attrs = {'ID', 'Len', 'Angle', 'Set', 'edge_key', 'orig_id', 'seg_count', 'original_e', 
                                'original_feature_id', 'start_node', 'end_node'}
                for key, value in data.items():
                    if key not in simplified_edge_data and key not in unwanted_attrs:
                        simplified_edge_data[key] = value
                
                G_simplified.add_edge(start, end, **simplified_edge_data)
                direct_edges_count += 1
        
        print(f"Direct edges between key nodes: {direct_edges_count}")
        
        # Now find and process chains
        chains_found = 0
        chains_processed = 0
        isolated_chains = 0
        
        for start_node in nodes_to_keep:
            # Key nodes are never added to processed_nodes, so no need to check
            
            # Find all neighbors of this key node
            for neighbor in list(G.neighbors(start_node)):
                if neighbor in processed_nodes:
                    continue
                    
                if neighbor in nodes_to_keep:
                    # Direct connection already handled above
                    continue
                elif G.degree(neighbor) == 2:
                    # This starts a chain - trace it
                    chains_found += 1
                    chain_path = [start_node]
                    current = neighbor
                    
                    # Follow the chain until we reach another key node or dead end
                    while current is not None:
                        chain_path.append(current)
                        
                        # Only add intermediate nodes (not key nodes) to processed_nodes
                        if current not in nodes_to_keep:
                            processed_nodes.add(current)
                        
                        # Find next node in chain
                        neighbors = [n for n in G.neighbors(current) if n != chain_path[-2]]
                        
                        if len(neighbors) == 0:
                            # Dead end - shouldn't happen in degree-2 chain
                            print(f"Warning: Dead end in chain at node {current}")
                            break
                        elif len(neighbors) == 1:
                            next_node = neighbors[0]
                            if next_node in nodes_to_keep:
                                # Reached another key node - complete the chain
                                chain_path.append(next_node)
                                # Don't add the end key node to processed_nodes!
                                break
                            elif G.degree(next_node) == 2 and next_node not in processed_nodes:
                                # Continue following the chain
                                current = next_node
                            else:
                                # Reached a node that breaks the chain pattern
                                chain_path.append(next_node)
                                # Only add to processed if it's not a key node
                                if next_node not in nodes_to_keep:
                                    processed_nodes.add(next_node)
                                break
                        else:
                            # Node has more than 1 unvisited neighbor - shouldn't happen in simple chain
                            print(f"Warning: Chain node {current} has {len(neighbors)} unvisited neighbors")
                            break
                    
                    # Process the complete chain
                    if len(chain_path) >= 3:  # start_key -> intermediate(s) -> end_key
                        end_node = chain_path[-1]
                        if end_node in nodes_to_keep:
                            # Valid chain between two key nodes
                            merged_chain_data = self.merge_chain_edges_with_line_id(G, chain_path)
                            
                            # Create simplified edge data in the same format as create_simplified_graph_simple
                            start_type = self.classify_node_type(G, chain_path[0])
                            end_type = self.classify_node_type(G, chain_path[-1])
                            edge_type = self.create_edge_type(start_type, end_type)
                            
                            simplified_edge_data = {
                                'line_id': merged_chain_data.get('line_id', 'unknown'),
                                'length': merged_chain_data.get('length', 0),
                                'edge_type': edge_type,
                                'start_node': chain_path[0],
                                'end_node': chain_path[-1],
                                'st_type': start_type,
                                'end_type': end_type,
                                'geometry': merged_chain_data.get('geometry')  # Keep geometry for full graph
                            }
                            
                            # Add any additional attributes from the merged data
                            unwanted_attrs = {'line_id', 'length', 'geometry', 'original_feature_ids', 'segment_count', 
                                            'original_lengths', 'start_node', 'end_node'}
                            for key, value in merged_chain_data.items():
                                if key not in simplified_edge_data and key not in unwanted_attrs:
                                    simplified_edge_data[key] = value
                            
                            # Use MultiGraph to allow multiple edges between same nodes
                            G_simplified.add_edge(chain_path[0], chain_path[-1], **simplified_edge_data)
                            chains_processed += 1
                        else:
                            # Chain doesn't end at a key node - this might be a problem
                            print(f"Warning: Chain from {chain_path[0]} doesn't end at key node. Ends at {end_node} (degree {G.degree(end_node)})")
                            isolated_chains += 1
                    elif len(chain_path) == 2:
                        # Single intermediate node - might be isolated
                        print(f"Warning: Single-node chain: {chain_path}")
                        isolated_chains += 1
        
        print(f"Chains found: {chains_found}, processed: {chains_processed}, isolated/problematic: {isolated_chains}")
        
        # Handle any remaining degree-2 nodes that weren't part of chains starting from key nodes
        remaining_degree2 = [n for n in G.nodes() if G.degree(n) == 2 and n not in processed_nodes]
        if remaining_degree2:
            print(f"Processing {len(remaining_degree2)} remaining degree-2 nodes as isolated components")
            
            # These might be isolated loops or chains not connected to key nodes
            isolated_components = []
            for node in remaining_degree2:
                if node not in processed_nodes:
                    # Try to trace a complete chain/loop starting from this node
                    chain_path = [node]
                    current = node
                    visited = {node}
                    
                    # Follow chain in one direction
                    neighbors = list(G.neighbors(current))
                    if len(neighbors) >= 1:
                        current = neighbors[0]
                        
                        while current not in visited and current not in processed_nodes:
                            chain_path.append(current)
                            visited.add(current)
                            
                            if G.degree(current) == 2:
                                next_neighbors = [n for n in G.neighbors(current) if n not in visited]
                                if len(next_neighbors) == 1:
                                    current = next_neighbors[0]
                                else:
                                    break
                            else:
                                # Reached a non-degree-2 node
                                break
                        
                        # Add final node if it's not already in the path
                        if current not in visited and current not in processed_nodes:
                            chain_path.append(current)
                    
                    isolated_components.append(chain_path)
                    
                    # Mark these nodes as processed (only the intermediate ones)
                    for n in chain_path:
                        if n not in nodes_to_keep:  # Only mark non-key nodes as processed
                            processed_nodes.add(n)
            
            # Process isolated components
            for component in isolated_components:
                if len(component) >= 2:
                    # Add endpoints as nodes if they aren't already key nodes
                    start_node = component[0]
                    end_node = component[-1]
                    
                    if start_node not in G_simplified:
                        G_simplified.add_node(start_node, **G.nodes[start_node])
                    if end_node not in G_simplified and end_node != start_node:
                        G_simplified.add_node(end_node, **G.nodes[end_node])
                    
                    # Create merged edge for this isolated component
                    if start_node != end_node:  # Not a perfect loop
                        merged_component_data = self.merge_chain_edges_with_line_id(G, component)
                        
                        start_type = self.classify_node_type(G, start_node)
                        end_type = self.classify_node_type(G, end_node)
                        edge_type = self.create_edge_type(start_type, end_type)
                        
                        simplified_edge_data = {
                            'line_id': merged_component_data.get('line_id', 'unknown'),
                            'length': merged_component_data.get('length', 0),
                            'edge_type': edge_type,
                            'start_node': start_node,
                            'end_node': end_node,
                            'st_type': start_type,
                            'end_type': end_type,
                            'geometry': merged_component_data.get('geometry')
                        }
                        
                        G_simplified.add_edge(start_node, end_node, **simplified_edge_data)
                    else:  # Perfect loop - create self-loop
                        # For loops, we might want to pick a representative node
                        if len(component) > 1:
                            # Create a self-loop on the first node
                            merged_component_data = self.merge_chain_edges_with_line_id(G, component + [component[0]])  # Close the loop
                            
                            start_type = self.classify_node_type(G, start_node)
                            edge_type = self.create_edge_type(start_type, start_type)
                            
                            simplified_edge_data = {
                                'line_id': merged_component_data.get('line_id', 'unknown'),
                                'length': merged_component_data.get('length', 0),
                                'edge_type': edge_type,
                                'start_node': start_node,
                                'end_node': start_node,
                                'st_type': start_type,
                                'end_type': start_type,
                                'geometry': merged_component_data.get('geometry')
                            }
                            
                            G_simplified.add_edge(start_node, start_node, **simplified_edge_data)
            
            print(f"Processed {len(isolated_components)} isolated components")
        
        # Verification
        original_edge_count = G.number_of_edges()
        simplified_edge_count = G_simplified.number_of_edges()
        
        # Also check total length
        original_total_length = sum(data.get('length', 0) for _, _, data in G.edges(data=True))
        simplified_total_length = sum(data.get('length', 0) for _, _, data in G_simplified.edges(data=True))
        
        print(f"Original edges: {original_edge_count}")
        print(f"Simplified edges: {simplified_edge_count}")
        print(f"Original total length: {original_total_length:.2f}")
        print(f"Simplified total length: {simplified_total_length:.2f}")
        
        length_diff = abs(original_total_length - simplified_total_length)
        if length_diff > 0.01:  # Allow for small floating point errors
            print(f"WARNING: Length difference of {length_diff:.2f} detected!")
            print("This suggests some edges may have been lost or double-counted")
        else:
            print("✓ Total length preserved successfully")
        
        print(f"Simplified graph: {G_simplified.number_of_nodes()} nodes, {G_simplified.number_of_edges()} edges")
        
        return G_simplified


    def merge_chain_edges_with_line_id(self,G, chain_path):
        """
        Merge edges along a chain path, combining their attributes.
        Ensures geometries are oriented correctly along the chain direction.
        Returns line_id based on the first edge's ID attribute (like create_simplified_graph_simple).
        
        Parameters:
        -----------
        G : networkx.Graph
            The original graph
        chain_path : list
            List of nodes forming a chain
            
        Returns:
        --------
        dict
            Merged edge attributes with line_id matching create_simplified_graph_simple format
        """
        merged_data = {
            'length': 0.0,
            'geometry': None,
            'line_id': 'unknown'
        }
        
        # Collect geometries and other attributes
        geometries = []
        all_attributes = defaultdict(list)
        first_edge = True
        
        for i in range(len(chain_path) - 1):
            start_node, end_node = chain_path[i], chain_path[i + 1]
            
            if G.has_edge(start_node, end_node):
                edge_data = G[start_node][end_node]
                
                # Get line_id from the first edge (matches create_simplified_graph_simple behavior)
                if first_edge:
                    merged_data['line_id'] = edge_data.get('ID', 'unknown')
                    first_edge = False
                
                # Accumulate length
                edge_length = edge_data.get('length', 0)
                merged_data['length'] += edge_length
                
                # Collect and orient geometry correctly
                if 'geometry' in edge_data and edge_data['geometry'] is not None:
                    edge_geom = edge_data['geometry']
                    
                    # Get the actual coordinates of start and end nodes
                    start_pos = (G.nodes[start_node]['x'], G.nodes[start_node]['y'])
                    end_pos = (G.nodes[end_node]['x'], G.nodes[end_node]['y'])
                    
                    # Get the geometry coordinates
                    geom_coords = list(edge_geom.coords)
                    geom_start = geom_coords[0]
                    geom_end = geom_coords[-1]
                    
                    # Check if geometry orientation matches chain direction
                    # Calculate distances to determine correct orientation
                    start_to_geom_start = ((start_pos[0] - geom_start[0])**2 + (start_pos[1] - geom_start[1])**2)**0.5
                    start_to_geom_end = ((start_pos[0] - geom_end[0])**2 + (start_pos[1] - geom_end[1])**2)**0.5
                    
                    if start_to_geom_start <= start_to_geom_end:
                        # Geometry is oriented correctly (start_node closer to geom_start)
                        oriented_geom = edge_geom
                    else:
                        # Geometry is reversed - flip it
                        oriented_geom = LineString(geom_coords[::-1])
                    
                    geometries.append(oriented_geom)
                
                # Collect other attributes (excluding those we handle specially)
                excluded_attrs = {'length', 'geometry', 'ID', 'original_feature_id', 'start_node', 'end_node'}
                for key, value in edge_data.items():
                    if key not in excluded_attrs:
                        all_attributes[key].append(value)
        
        # Merge geometries
        if geometries:
            try:
                # Combine all coordinates
                all_coords = []
                for i, geom in enumerate(geometries):
                    coords = list(geom.coords)
                    if i == 0:
                        # First geometry - include all coordinates
                        all_coords.extend(coords)
                    else:
                        # Subsequent geometries - skip first coordinate to avoid duplication
                        all_coords.extend(coords[1:])
                
                merged_data['geometry'] = LineString(all_coords)
            except Exception as e:
                print(f"Warning: Could not merge geometries: {e}")
                merged_data['geometry'] = geometries[0] if geometries else None
        
        # Handle other attributes - take the first value or most common value
        for key, values in all_attributes.items():
            if len(set(values)) == 1:
                # All values are the same
                merged_data[key] = values[0]
            else:
                # Multiple different values - take the first one (consistent with create_simplified_graph_simple)
                merged_data[key] = values[0]
        
        return merged_data

    def create_simplified_graph_fullx(self,G):
        """
        Create a simplified graph keeping only intersection nodes (degree > 2) and endpoints (degree = 1).
        Intermediate nodes (degree = 2) are removed and their edges are merged.
        Uses MultiGraph to handle multiple edges between same nodes.
        
        Parameters:
        -----------
        G : networkx.Graph
            The detailed network graph
            
        Returns:
        --------
        networkx.MultiGraph
            Simplified multigraph with only key nodes
        """
        print("\n=== Creating Simplified Graph ===")
        
        # Create a MultiGraph to handle multiple edges between same nodes
        G_simplified = nx.MultiGraph()
        
        # Copy all key nodes to the simplified graph
        nodes_to_keep = set()
        degree_stats = defaultdict(int)
        
        for node in G.nodes():
            degree = G.degree(node)
            degree_stats[degree] += 1
            if degree != 2:  # Keep intersections (degree > 2), endpoints (degree = 1), and isolated (degree = 0)
                nodes_to_keep.add(node)
        
        print(f"Node degree distribution: {dict(degree_stats)}")
        print(f"Keeping {len(nodes_to_keep)} key nodes out of {G.number_of_nodes()} total nodes")
        
        # Add all key nodes to simplified graph
        for node in nodes_to_keep:
            G_simplified.add_node(node, **G.nodes[node])
        
        # Find all chains and process them
        # Only intermediate degree-2 nodes should be marked as processed
        # Key nodes should never be marked as processed so they can serve as start points for multiple chains
        processed_nodes = set()
        
        # First, handle all edges that connect key nodes directly
        direct_edges_count = 0
        for start, end, data in G.edges(data=True):
            if start in nodes_to_keep and end in nodes_to_keep:
                # Direct edge between key nodes - preserve it
                G_simplified.add_edge(start, end, **data)
                direct_edges_count += 1
        
        print(f"Direct edges between key nodes: {direct_edges_count}")
        
        # Now find and process chains
        chains_found = 0
        chains_processed = 0
        isolated_chains = 0
        
        for start_node in nodes_to_keep:
            # Key nodes are never added to processed_nodes, so no need to check
            
            # Find all neighbors of this key node
            for neighbor in list(G.neighbors(start_node)):
                if neighbor in processed_nodes:
                    continue
                    
                if neighbor in nodes_to_keep:
                    # Direct connection already handled above
                    continue
                elif G.degree(neighbor) == 2:
                    # This starts a chain - trace it
                    chains_found += 1
                    chain_path = [start_node]
                    current = neighbor
                    
                    # Follow the chain until we reach another key node or dead end
                    while current is not None:
                        chain_path.append(current)
                        
                        # Only add intermediate nodes (not key nodes) to processed_nodes
                        if current not in nodes_to_keep:
                            processed_nodes.add(current)
                        
                        # Find next node in chain
                        neighbors = [n for n in G.neighbors(current) if n != chain_path[-2]]
                        
                        if len(neighbors) == 0:
                            # Dead end - shouldn't happen in degree-2 chain
                            print(f"Warning: Dead end in chain at node {current}")
                            break
                        elif len(neighbors) == 1:
                            next_node = neighbors[0]
                            if next_node in nodes_to_keep:
                                # Reached another key node - complete the chain
                                chain_path.append(next_node)
                                # Don't add the end key node to processed_nodes!
                                break
                            elif G.degree(next_node) == 2 and next_node not in processed_nodes:
                                # Continue following the chain
                                current = next_node
                            else:
                                # Reached a node that breaks the chain pattern
                                chain_path.append(next_node)
                                # Only add to processed if it's not a key node
                                if next_node not in nodes_to_keep:
                                    processed_nodes.add(next_node)
                                break
                        else:
                            # Node has more than 1 unvisited neighbor - shouldn't happen in simple chain
                            print(f"Warning: Chain node {current} has {len(neighbors)} unvisited neighbors")
                            break
                    
                    # Process the complete chain
                    if len(chain_path) >= 3:  # start_key -> intermediate(s) -> end_key
                        end_node = chain_path[-1]
                        if end_node in nodes_to_keep:
                            # Valid chain between two key nodes
                            merged_edge_data = self.merge_chain_edges(G, chain_path)
                            # Use MultiGraph to allow multiple edges between same nodes
                            G_simplified.add_edge(chain_path[0], chain_path[-1], **merged_edge_data)
                            chains_processed += 1
                        else:
                            # Chain doesn't end at a key node - this might be a problem
                            print(f"Warning: Chain from {chain_path[0]} doesn't end at key node. Ends at {end_node} (degree {G.degree(end_node)})")
                            isolated_chains += 1
                    elif len(chain_path) == 2:
                        # Single intermediate node - might be isolated
                        print(f"Warning: Single-node chain: {chain_path}")
                        isolated_chains += 1
        
        print(f"Chains found: {chains_found}, processed: {chains_processed}, isolated/problematic: {isolated_chains}")
        
        # Handle any remaining degree-2 nodes that weren't part of chains starting from key nodes
        remaining_degree2 = [n for n in G.nodes() if G.degree(n) == 2 and n not in processed_nodes]
        if remaining_degree2:
            print(f"Processing {len(remaining_degree2)} remaining degree-2 nodes as isolated components")
            
            # These might be isolated loops or chains not connected to key nodes
            isolated_components = []
            for node in remaining_degree2:
                if node not in processed_nodes:
                    # Try to trace a complete chain/loop starting from this node
                    chain_path = [node]
                    current = node
                    visited = {node}
                    
                    # Follow chain in one direction
                    neighbors = list(G.neighbors(current))
                    if len(neighbors) >= 1:
                        current = neighbors[0]
                        
                        while current not in visited and current not in processed_nodes:
                            chain_path.append(current)
                            visited.add(current)
                            
                            if G.degree(current) == 2:
                                next_neighbors = [n for n in G.neighbors(current) if n not in visited]
                                if len(next_neighbors) == 1:
                                    current = next_neighbors[0]
                                else:
                                    break
                            else:
                                # Reached a non-degree-2 node
                                break
                        
                        # Add final node if it's not already in the path
                        if current not in visited and current not in processed_nodes:
                            chain_path.append(current)
                    
                    isolated_components.append(chain_path)
                    
                    # Mark these nodes as processed (only the intermediate ones)
                    for n in chain_path:
                        if n not in nodes_to_keep:  # Only mark non-key nodes as processed
                            processed_nodes.add(n)
            
            # Process isolated components
            for component in isolated_components:
                if len(component) >= 2:
                    # Add endpoints as nodes if they aren't already key nodes
                    start_node = component[0]
                    end_node = component[-1]
                    
                    if start_node not in G_simplified:
                        G_simplified.add_node(start_node, **G.nodes[start_node])
                    if end_node not in G_simplified and end_node != start_node:
                        G_simplified.add_node(end_node, **G.nodes[end_node])
                    
                    # Create merged edge for this isolated component
                    if start_node != end_node:  # Not a perfect loop
                        merged_edge_data = self.merge_chain_edges(G, component)
                        G_simplified.add_edge(start_node, end_node, **merged_edge_data)
                    else:  # Perfect loop - create self-loop
                        # For loops, we might want to pick a representative node
                        if len(component) > 1:
                            # Create a self-loop on the first node
                            merged_edge_data = self.merge_chain_edges(G, component + [component[0]])  # Close the loop
                            G_simplified.add_edge(start_node, start_node, **merged_edge_data)
            
            print(f"Processed {len(isolated_components)} isolated components")
        
        # Verify no edges were lost by checking edge count instead of length
        original_edge_count = G.number_of_edges()
        simplified_edge_count = G_simplified.number_of_edges()
        
        # Also check total length
        original_total_length = sum(data.get('length', 0) for _, _, data in G.edges(data=True))
        simplified_total_length = sum(data.get('length', 0) for _, _, data in G_simplified.edges(data=True))
        
        print(f"Original edges: {original_edge_count}")
        print(f"Simplified edges: {simplified_edge_count}")
        print(f"Original total length: {original_total_length:.2f}")
        print(f"Simplified total length: {simplified_total_length:.2f}")
        
        length_diff = abs(original_total_length - simplified_total_length)
        if length_diff > 0.01:  # Allow for small floating point errors
            print(f"WARNING: Length difference of {length_diff:.2f} detected!")
            print("This suggests some edges may have been lost or double-counted")
        else:
            print("✓ Total length preserved successfully")
        
        print(f"Simplified graph: {G_simplified.number_of_nodes()} nodes, {G_simplified.number_of_edges()} edges")
        
        return G_simplified


    def create_simplified_graph_simple(self,G):
        """
        Create a simplified graph that connects only key nodes (degree != 2) with single edges.
        Each edge stores the original polyline ID and total length of the merged segments.
        
        Parameters:
        -----------
        G : networkx.Graph
            The detailed network graph where edges should have 'polyline_id' and 'length' attributes
            
        Returns:
        --------
        networkx.MultiGraph
            Simplified multigraph with edges containing original polyline IDs
        """
        print("\n=== Creating Simplified Graph with Polyline IDs ===")
        
        # Create a MultiGraph to handle multiple edges between same nodes
        G_simplified = nx.MultiGraph()
        
        # Identify key nodes to keep (intersections, endpoints, isolated nodes)
        key_nodes = set()
        degree_stats = defaultdict(int)
        
        for node in G.nodes():
            degree = G.degree(node)
            degree_stats[degree] += 1
            if degree != 2:  # Keep intersections (degree > 2), endpoints (degree = 1), and isolated (degree = 0)
                key_nodes.add(node)
        
        print(f"Node degree distribution: {dict(degree_stats)}")
        print(f"Keeping {len(key_nodes)} key nodes out of {G.number_of_nodes()} total nodes")
        
        # Add all key nodes to simplified graph
        for node in key_nodes:
            G_simplified.add_node(node, **G.nodes[node])
        
        # Track processed edges and chains
        processed_edges = set()
        processed_nodes = set()
        polylines_processed = 0
        direct_edges = 0
        
        # Process all edges in the original graph
        for start, end, edge_data in G.edges(data=True):
            edge_key = tuple(sorted([start, end]))
            if edge_key in processed_edges:
                continue
                
            start_degree = G.degree(start)
            end_degree = G.degree(end)
            
            # Case 1: Direct edge between key nodes
            if start_degree != 2 and end_degree != 2:
                # Classify endpoint types
                start_type = self.classify_node_type(G, start)
                end_type = self.classify_node_type(G, end)
                edge_type = self.create_edge_type(start_type, end_type)
                
                # Copy edge directly with original polyline ID
                simplified_edge_data = {
                    'line_id': edge_data.get('ID', 'unknown'),
                    'length': edge_data.get('length', 0),
                    'edge_type': edge_type,
                    'start_node': start,
                    'end_node': end,
                    'st_type': start_type,
                    'end_type': end_type
                }
                # Preserve any other attributes (but exclude unwanted ones)
                unwanted_attrs = {'ID', 'Len', 'Angle', 'Set', 'edge_key', 'orig_id', 'seg_count', 'original_e'}
                for key, value in edge_data.items():
                    if key not in simplified_edge_data and key not in unwanted_attrs:
                        simplified_edge_data[key] = value
                        
                G_simplified.add_edge(start, end, **simplified_edge_data)
                processed_edges.add(edge_key)
                direct_edges += 1
                
            # Case 2: Edge involves degree-2 node(s) - trace the complete polyline
            elif start_degree == 2 or end_degree == 2:
                polyline = self.trace_complete_polyline(G, start, end, processed_edges, processed_nodes)
                
                if polyline and len(polyline['nodes']) >= 2:
                    polylines_processed += 1
                    
                    start_node = polyline['nodes'][0]
                    end_node = polyline['nodes'][-1]
                    
                    # Ensure endpoints are in simplified graph
                    if start_node not in G_simplified:
                        G_simplified.add_node(start_node, **G.nodes[start_node])
                    if end_node not in G_simplified and end_node != start_node:
                        G_simplified.add_node(end_node, **G.nodes[end_node])
                    
                    # Create single edge with polyline information
                    start_type = self.classify_node_type(G, start_node)
                    end_type = self.classify_node_type(G, end_node)
                    edge_type = self.create_edge_type(start_type, end_type)
                    
                    simplified_edge_data = {
                        'line_id': polyline['polyline_id'],
                        'length': polyline['total_length'],
                        'edge_type': edge_type,
                        'start_node': start_node,
                        'end_node': end_node,
                        'st_type': start_type,
                        'end_type': end_type
                    }
                    
                    # Add edge to simplified graph
                    if start_node != end_node:
                        G_simplified.add_edge(start_node, end_node, **simplified_edge_data)
                    else:
                        # Self-loop case (closed polyline)
                        G_simplified.add_edge(start_node, start_node, **simplified_edge_data)
        
        print(f"Direct edges between key nodes: {direct_edges}")
        print(f"Polylines processed: {polylines_processed}")
        
        # Verification
        original_edge_count = G.number_of_edges()
        simplified_edge_count = G_simplified.number_of_edges()
        
        original_total_length = sum(data.get('length', 0) for _, _, data in G.edges(data=True))
        simplified_total_length = sum(data.get('length', 0) for _, _, data in G_simplified.edges(data=True))
        
        print(f"Original edges: {original_edge_count}")
        print(f"Simplified edges: {simplified_edge_count}")
        print(f"Original total length: {original_total_length:.2f}")
        print(f"Simplified total length: {simplified_total_length:.2f}")
        
        length_diff = abs(original_total_length - simplified_total_length)
        if length_diff > 0.01:
            print(f"WARNING: Length difference of {length_diff:.2f} detected!")
        else:
            print("✓ Total length preserved successfully")
        
        print(f"Simplified graph: {G_simplified.number_of_nodes()} nodes, {G_simplified.number_of_edges()} edges")
        
        return G_simplified


    def classify_node_type(self,G, node):
        """
        Classify a node based on its degree:
        I = Isolated (degree 0)
        Z = Terminal/Dead-end (degree 1) 
        Y = Junction (degree 2) - shouldn't appear in simplified graph
        X = Intersection (degree 3+)
        
        Note: In the original graph context, we classify based on how many
        edges the node participates in.
        """
        degree = G.degree(node)
        
        if degree == 0:
            return 'I'  # Isolated
        elif degree == 1:
            return 'Z'  # Terminal/Dead-end (connects to one other edge)
        elif degree == 2:
            return 'Y'  # Junction (connects to two other edges) - rare in simplified graph
        else:  # degree >= 3
            return 'X'  # Intersection (connects to 3+ other edges)


    def create_edge_type(self,start_type, end_type):
        """
        Create edge type string with alphabetical ordering to make direction irrelevant.
        e.g., Y-X and X-Y both become X-Y
        """
        types = sorted([start_type, end_type])
        return f"{types[0]}-{types[1]}"


    def trace_complete_polyline(self,G, start, end, processed_edges, processed_nodes):
        """
        Trace a complete polyline starting from an edge that involves degree-2 nodes.
        Returns polyline information including all nodes, total length, and polyline ID.
        """
        edge_key = tuple(sorted([start, end]))
        if edge_key in processed_edges:
            return None
        
        # Get the polyline ID from the starting edge
        edge_data = G.get_edge_data(start, end)
        if not edge_data:
            return None
        
        polyline_id = edge_data.get('ID', 'unknown')
        
        # Start with the current edge
        polyline_nodes = [start, end]
        total_length = edge_data.get('length', 0)
        edges_in_polyline = [(start, end)]
        
        # Extend backwards from start
        current = start
        while G.degree(current) == 2 and current not in processed_nodes:
            # Find the neighbor that's not already in our polyline
            neighbors = [n for n in G.neighbors(current) if n not in polyline_nodes]
            if len(neighbors) != 1:
                break
                
            prev_node = neighbors[0]
            
            # Check if this edge belongs to the same polyline
            edge_data = G.get_edge_data(current, prev_node)
            if not edge_data or edge_data.get('ID', 'unknown') != polyline_id:
                break
                
            # Add to polyline
            polyline_nodes.insert(0, prev_node)
            total_length += edge_data.get('length', 0)
            edges_in_polyline.append((current, prev_node))
            current = prev_node
        
        # Extend forwards from end
        current = end
        while G.degree(current) == 2 and current not in processed_nodes:
            # Find the neighbor that's not already in our polyline
            neighbors = [n for n in G.neighbors(current) if n not in polyline_nodes]
            if len(neighbors) != 1:
                break
                
            next_node = neighbors[0]
            
            # Check if this edge belongs to the same polyline
            edge_data = G.get_edge_data(current, next_node)
            if not edge_data or edge_data.get('ID', 'unknown') != polyline_id:
                break
                
            # Add to polyline
            polyline_nodes.append(next_node)
            total_length += edge_data.get('length', 0)
            edges_in_polyline.append((current, next_node))
            current = next_node
        
        # Mark all edges as processed
        for edge_start, edge_end in edges_in_polyline:
            edge_key = tuple(sorted([edge_start, edge_end]))
            processed_edges.add(edge_key)
        
        # Mark intermediate nodes as processed
        for node in polyline_nodes[1:-1]:  # Don't mark endpoints
            if G.degree(node) == 2:
                processed_nodes.add(node)
        
        return {
            'nodes': polyline_nodes,
            'polyline_id': polyline_id,
            'total_length': total_length,
            'edge_count': len(edges_in_polyline)
        }

    def find_paths_in_components(self,G):
        """
        Find example paths within connected components.
        
        Parameters:
        -----------
        G : networkx.Graph
            The network graph
        """
        components = list(nx.connected_components(G))
        
        print(f"\n=== Path Analysis ===")
        print(f"Found {len(components)} connected components")
        
        for i, component in enumerate(components[:5]):  # Show first 5 components
            if len(component) >= 2:
                comp_nodes = list(component)
                subgraph = G.subgraph(component)
                
                # Find shortest path within this component
                try:
                    path = nx.shortest_path(subgraph, comp_nodes[0], comp_nodes[-1], weight='length')
                    path_length = nx.shortest_path_length(subgraph, comp_nodes[0], comp_nodes[-1], weight='length')
                    print(f"Component {i+1} ({len(component)} nodes): Path from {comp_nodes[0]} to {comp_nodes[-1]}")
                    print(f"  Path length: {path_length:.2f} units")
                    print(f"  Path nodes: {len(path)} nodes")
                except nx.NetworkXNoPath:
                    print(f"Component {i+1}: No path found (shouldn't happen in connected component)")
            else:
                print(f"Component {i+1}: Only {len(component)} node(s)")
        
        if len(components) > 5:
            print(f"... and {len(components) - 5} more components")

    def merge_chain_edges(self,G, chain_path):
        """
        Merge edges along a chain path, combining their attributes.
        Ensures geometries are oriented correctly along the chain direction.
        
        Parameters:
        -----------
        G : networkx.Graph
            The original graph
        chain_path : list
            List of nodes forming a chain
            
        Returns:
        --------
        dict
            Merged edge attributes
        """
        merged_data = {
            'length': 0.0,
            'geometry': None,
            'original_feature_ids': [],
            'segment_count': 0,
            'original_lengths': []
        }
        
        # Collect geometries and other attributes
        geometries = []
        all_attributes = defaultdict(list)
        
        for i in range(len(chain_path) - 1):
            start_node, end_node = chain_path[i], chain_path[i + 1]
            
            if G.has_edge(start_node, end_node):
                edge_data = G[start_node][end_node]
                
                # Accumulate length
                edge_length = edge_data.get('length', 0)
                merged_data['length'] += edge_length
                merged_data['original_lengths'].append(edge_length)
                
                # Collect and orient geometry correctly
                if 'geometry' in edge_data and edge_data['geometry'] is not None:
                    edge_geom = edge_data['geometry']
                    
                    # Get the actual coordinates of start and end nodes
                    start_pos = (G.nodes[start_node]['x'], G.nodes[start_node]['y'])
                    end_pos = (G.nodes[end_node]['x'], G.nodes[end_node]['y'])
                    
                    # Get the geometry coordinates
                    geom_coords = list(edge_geom.coords)
                    geom_start = geom_coords[0]
                    geom_end = geom_coords[-1]
                    
                    # Check if geometry orientation matches chain direction
                    # Calculate distances to determine correct orientation
                    start_to_geom_start = ((start_pos[0] - geom_start[0])**2 + (start_pos[1] - geom_start[1])**2)**0.5
                    start_to_geom_end = ((start_pos[0] - geom_end[0])**2 + (start_pos[1] - geom_end[1])**2)**0.5
                    
                    if start_to_geom_start <= start_to_geom_end:
                        # Geometry is oriented correctly (start_node closer to geom_start)
                        oriented_geom = edge_geom
                    else:
                        # Geometry is reversed - flip it
                        oriented_geom = LineString(geom_coords[::-1])
                    
                    geometries.append(oriented_geom)
                
                # Collect original feature IDs
                if 'original_feature_id' in edge_data:
                    merged_data['original_feature_ids'].append(edge_data['original_feature_id'])
                if 'original_feature_id' in edge_data and edge_data['original_feature_id'] == 13965:
                    print(f"Found edge with original feature ID 13965: {start_node} -> {end_node}")

                # Collect other attributes
                for key, value in edge_data.items():
                    if key not in ['length', 'geometry', 'original_feature_id']:
                        all_attributes[key].append(value)
                
                merged_data['segment_count'] += 1
        
        # Merge geometries
        if geometries:
            try:
                # Combine all coordinates
                all_coords = []
                for i, geom in enumerate(geometries):
                    coords = list(geom.coords)
                    if i == 0:
                        # First geometry - include all coordinates
                        all_coords.extend(coords)
                    else:
                        # Subsequent geometries - skip first coordinate to avoid duplication
                        all_coords.extend(coords[1:])
                
                if merged_data['original_feature_ids'] and merged_data['original_feature_ids'][0] == 13965:
                    print(f"Merging geometries for chain: {chain_path}")
                    print(f"Chain starts at node {chain_path[0]} with pos {(G.nodes[chain_path[0]]['x'], G.nodes[chain_path[0]]['y'])}")
                    print(f"Chain ends at node {chain_path[-1]} with pos {(G.nodes[chain_path[-1]]['x'], G.nodes[chain_path[-1]]['y'])}")
                    print(f"Merged coords start: {all_coords[0]}")
                    print(f"Merged coords end: {all_coords[-1]}")
                
                merged_data['geometry'] = LineString(all_coords)
            except Exception as e:
                print(f"Warning: Could not merge geometries: {e}")
                merged_data['geometry'] = geometries[0] if geometries else None
        
        # Handle other attributes - for now, take the most common value or first value
        for key, values in all_attributes.items():
            if len(set(values)) == 1:
                # All values are the same
                merged_data[key] = values[0]
            else:
                # Multiple different values - create a summary
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    # Numeric values - take average
                    numeric_values = [v for v in values if v is not None]
                    if numeric_values:
                        merged_data[f"{key}_avg"] = sum(numeric_values) / len(numeric_values)
                        merged_data[f"{key}_min"] = min(numeric_values)
                        merged_data[f"{key}_max"] = max(numeric_values)
                else:
                    # Non-numeric values - take most common or create list
                    merged_data[f"{key}_list"] = values
                    merged_data[key] = values[0]  # Keep first value as default
        
        return merged_data



    # Example usage
    def CalcFaultsGraph(self,shapefile_path):
        # Convert shapefile to network graph
        #shapefile_path = "waxi4_faults_utm29_merged.shp"  # Replace with your shapefile path
        
        try:
            gdf=gpd.read_file(shapefile_path)
            #new_gdf=split_polylines_at_intersections(gdf, tolerance=1e-6, id_column='ID')# Create the detailed network graph
            G_detailed, original_gdf = self.polylines_to_network_graph(gdf)
            
            # Analyze the detailed graph
            print("=== DETAILED GRAPH ===")
            self.analyze_graph(G_detailed)
            self.find_paths_in_components(G_detailed)
            
            # Create simplified graph
            G_simplified_simple = self.create_simplified_graph_simple(G_detailed)
            #G_simplified_simple=add_centrality_measures(G_simplified_simple)
            G_simplified_full = self.create_simplified_graph_full(G_detailed)
            
            # Analyze the simplified graph
            print("\n=== SIMPLIFIED GRAPH ===")
            self.analyze_graph(G_simplified_full)
            self.find_paths_in_components(G_simplified_full)
            
            # Save both graphs to shapefiles
            print("\n=== SAVING FILES ===")
            
            # Save detailed graph
            #self.save_graph_to_shapefile(G_detailed, "waxi_network_nodes.shp", "waxi_network_edges.shp", 
            #                    original_gdf, prefix="detailed")
            
            nodes=shapefile_path.replace(".shp","_nodes.shp")
            edges=shapefile_path.replace(".shp","_edges.shp")            
            
            # Save simplified graph
            self.save_graph_to_shapefile(G_simplified_full,  nodes, edges, 
                original_gdf, prefix="simplified_full")
            
            # Save simplified graph

            #self.save_graph_to_shapefile(G_simplified_simple, nodes, edges, 
            #     original_gdf, prefix="simplified_simple")
            
            print("\nFiles saved:")
            print("- detailed_network_nodes.shp / detailed_network_edges.shp (full graph)")
            print("- simplified_network_nodes.shp / simplified_network_edges.shp (intersections & endpoints only)")
            
            # Compare the two graphs
            print(f"\n=== COMPARISON ===")
            print(f"Detailed graph: {G_detailed.number_of_nodes()} nodes, {G_detailed.number_of_edges()} edges")
            print(f"Simplified graph: {G_simplified_full.number_of_nodes()} nodes, {G_simplified_full.number_of_edges()} edges")
            reduction_nodes = (1 - G_simplified_full.number_of_nodes() / G_detailed.number_of_nodes()) * 100
            reduction_edges = (1 - G_simplified_full.number_of_edges() / G_detailed.number_of_edges()) * 100
            print(f"Node reduction: {reduction_nodes:.1f}%")
            print(f"Edge reduction: {reduction_edges:.1f}%")
            
        except Exception as e:
            print(f"Error processing shapefile: {e}")
            print("Make sure the shapefile path is correct and contains LineString geometries.")
            
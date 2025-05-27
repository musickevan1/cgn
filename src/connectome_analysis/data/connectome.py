"""
Connectome construction utilities.
Build functional and structural connectivity matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
import warnings

# Import handling for optional packages
try:
    from nilearn.connectome import ConnectivityMeasure
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    ConnectivityMeasure = None
    print("Warning: nilearn not available. Limited connectivity options.")

class ConnectomeBuilder:
    """Build connectome matrices from neuroimaging data."""
    
    def __init__(self, 
                 connectivity_kinds: List[str] = ['correlation'],
                 standardize_connectomes: bool = True):
        """Initialize connectome builder."""
        self.connectivity_kinds = connectivity_kinds
        self.standardize_connectomes = standardize_connectomes
        
        # Initialize connectivity measures
        self.connectivity_measures = {}
        if NILEARN_AVAILABLE and ConnectivityMeasure is not None:
            for kind in connectivity_kinds:
                try:
                    self.connectivity_measures[kind] = ConnectivityMeasure(
                        kind=kind,
                        standardize=standardize_connectomes
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize {kind} connectivity: {e}")
        else:
            print("Warning: Using basic correlation only (nilearn not available)")
    
    def build_functional_connectome(self, 
                                  roi_time_series: np.ndarray,
                                  subject_id: str) -> Dict[str, np.ndarray]:
        """Build functional connectivity matrices."""
        connectomes = {}
        
        if NILEARN_AVAILABLE and self.connectivity_measures:
            for kind, measure in self.connectivity_measures.items():
                try:
                    # Fit and transform time series
                    connectivity_matrix = measure.fit_transform([roi_time_series])[0]
                    connectomes[kind] = connectivity_matrix
                    
                except Exception as e:
                    print(f"Warning: Could not compute {kind} connectivity for {subject_id}: {e}")
                    connectomes[kind] = np.full((roi_time_series.shape[1], roi_time_series.shape[1]), np.nan)
        else:
            # Fallback to basic correlation
            try:
                correlation_matrix = np.corrcoef(roi_time_series.T)
                connectomes['correlation'] = correlation_matrix
            except Exception as e:
                print(f"Warning: Could not compute correlation for {subject_id}: {e}")
                connectomes['correlation'] = np.full((roi_time_series.shape[1], roi_time_series.shape[1]), np.nan)
        
        return connectomes
    
    def compute_graph_metrics(self, 
                             connectivity_matrix: np.ndarray,
                             threshold: Optional[float] = None) -> Dict[str, Union[float, np.ndarray]]:
        """Compute graph theory metrics from connectivity matrix."""
        
        # Handle NaN values
        if np.isnan(connectivity_matrix).any():
            return {metric: np.nan for metric in 
                   ['global_efficiency', 'local_efficiency', 'modularity', 
                    'clustering', 'path_length', 'small_worldness']}
        
        # Apply threshold if specified
        if threshold is not None:
            adj_matrix = np.abs(connectivity_matrix) > threshold
        else:
            adj_matrix = np.abs(connectivity_matrix)
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adj_matrix)
        
        try:
            # Global metrics
            if threshold is not None:  # Binary graph
                global_efficiency = nx.global_efficiency(G)
                local_efficiency = nx.local_efficiency(G)
                clustering = nx.average_clustering(G)
                try:
                    if nx.is_connected(G):
                        path_length = nx.average_shortest_path_length(G)
                    else:
                        path_length = np.nan
                except:
                    path_length = np.nan
            else:  # Weighted graph
                global_efficiency = nx.global_efficiency(G)
                local_efficiency = nx.local_efficiency(G)
                clustering = nx.average_clustering(G, weight='weight')
                path_length = np.nan  # Not well-defined for weighted graphs
            
            # Modularity - Using compatible NetworkX methods
            modularity = np.nan
            try:
                # Try importing from the new location (NetworkX 3.0+)
                from networkx.algorithms.community.quality import modularity as nx_modularity
                import networkx.algorithms.community as nx_community
                partition = nx_community.louvain_communities(G)
                modularity = nx_modularity(G, partition)
            except ImportError:
                 try:
                     # Fallback to the old location (prior to NetworkX 3.0)
                     from networkx.algorithms.community import modularity as nx_modularity
                     import networkx.algorithms.community as nx_community
                     partition = nx_community.louvain_communities(G)
                     modularity = nx_modularity(G, partition)
                 except (AttributeError, ImportError):
                     pass  # Modularity will remain np.nan
                 except Exception:
                     pass
            except (AttributeError, Exception):
                pass # Handle other potential errors during modularity calculation
            
            # Small-worldness
            if not np.isnan(clustering) and not np.isnan(path_length):
                # Generate random graph for comparison
                n_nodes = len(G.nodes())
                n_edges = len(G.edges())
                if n_nodes > 1 and n_edges > 0:
                    p = n_edges / (n_nodes * (n_nodes - 1) / 2)
                    p = min(p, 1.0)  # Ensure probability is valid
                    random_G = nx.erdos_renyi_graph(n_nodes, p)
                    
                    random_clustering = nx.average_clustering(random_G)
                    try:
                        if nx.is_connected(random_G):
                            random_path_length = nx.average_shortest_path_length(random_G)
                        else:
                            random_path_length = np.nan
                    except:
                        random_path_length = np.nan
                    
                    if (not np.isnan(random_clustering) and not np.isnan(random_path_length) 
                        and random_clustering > 0 and random_path_length > 0):
                        small_worldness = (clustering / random_clustering) / (path_length / random_path_length)
                    else:
                        small_worldness = np.nan
                else:
                    small_worldness = np.nan
            else:
                small_worldness = np.nan
            
            return {
                'global_efficiency': float(global_efficiency),
                'local_efficiency': float(local_efficiency),
                'modularity': float(modularity) if not np.isnan(modularity) else np.nan,
                'clustering': float(clustering),
                'path_length': float(path_length) if not np.isnan(path_length) else np.nan,
                'small_worldness': float(small_worldness) if not np.isnan(small_worldness) else np.nan
            }
            
        except Exception as e:
            print(f"Warning: Could not compute graph metrics: {e}")
            return {metric: np.nan for metric in 
                   ['global_efficiency', 'local_efficiency', 'modularity', 
                    'clustering', 'path_length', 'small_worldness']}

class MultiModalConnectome:
    """Build multi-modal connectomes combining different data types."""
    
    def __init__(self):
        self.functional_builder = ConnectomeBuilder()
        
    def combine_modalities(self, 
                          functional_connectome: np.ndarray,
                          structural_connectome: Optional[np.ndarray] = None,
                          method: str = 'concatenate') -> np.ndarray:
        """Combine functional and structural connectivity information."""
        
        if structural_connectome is None:
            return functional_connectome
        
        if method == 'concatenate':
            # Stack upper triangular parts
            func_triu = functional_connectome[np.triu_indices_from(functional_connectome, k=1)]
            struct_triu = structural_connectome[np.triu_indices_from(structural_connectome, k=1)]
            return np.concatenate([func_triu, struct_triu])
        
        elif method == 'multiply':
            # Element-wise multiplication
            return functional_connectome * structural_connectome
        
        elif method == 'average':
            # Average of the two matrices
            return (functional_connectome + structural_connectome) / 2
        
        else:
            raise ValueError(f"Unknown combination method: {method}")

def extract_connectome_features(connectome: np.ndarray) -> np.ndarray:
    """Extract feature vector from connectivity matrix."""
    # Use upper triangular part (excluding diagonal)
    triu_indices = np.triu_indices_from(connectome, k=1)
    features = connectome[triu_indices]
    
    # Remove NaN values
    features = features[~np.isnan(features)]
    
    return features

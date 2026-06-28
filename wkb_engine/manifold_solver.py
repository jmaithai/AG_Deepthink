import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ManifoldSolver:
    def __init__(self, strand_map: Dict[str, Tuple[str, str]]):
        """
        Args:
            strand_map: Mapping of symbol -> (node_a, node_b)
        """
        self.strand_map = strand_map

    def build_design_matrix(self, active_symbols: List[str], active_nodes: List[str]) -> np.ndarray:
        """
        Builds the design matrix A_t for the system y = A_t * x.
        Rows: active_symbols + 1 (for zero-sum constraint).
        Cols: active_nodes.
        """
        M = len(active_symbols)
        N = len(active_nodes)
        node_to_idx = {node: i for i, node in enumerate(active_nodes)}
        
        A = np.zeros((M, N))
        for i, sym in enumerate(active_symbols):
            node_a, node_b = self.strand_map[sym]
            A[i, node_to_idx[node_a]] = 1.0
            if node_b is not None:
                A[i, node_to_idx[node_b]] = -1.0
                
        # Stack the conservation constraint row: sum(x) = 0
        constraint = np.ones((1, N))
        return np.vstack([A, constraint])

    def solve_pressures(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Solves least-squares pressures dynamically per bar.
        Only includes nodes that have active pricing data at each timestamp.
        """
        # Filter symbols that are in strand_map
        symbols = [c for c in prices_df.columns if c in self.strand_map]
        if not symbols:
            raise ValueError("No tradeable symbols matched strand_map.")
            
        prices_sub = prices_df[symbols].copy()
        log_prices = np.log(prices_sub)
        
        solved_records = []
        timestamps = prices_df.index
        
        # We solve row-by-row because active topology can change dynamically (e.g. market openings/closings)
        for t in timestamps:
            row_prices = log_prices.loc[t]
            
            # Determine active symbols at time t (non-NaN prices)
            valid_syms = [s for s in symbols if not np.isnan(row_prices[s])]
            if not valid_syms:
                # Fallback to zeros if no data at this timestamp
                solved_records.append({})
                continue
                
            # Determine active nodes for these symbols
            nodes_set = set()
            for s in valid_syms:
                node_a, node_b = self.strand_map[s]
                nodes_set.add(node_a)
                if node_b is not None:
                    nodes_set.add(node_b)
            active_nodes = sorted(list(nodes_set))
            
            if len(active_nodes) < 2:
                solved_records.append({})
                continue
                
            # Build design matrix A_t
            A_t = self.build_design_matrix(valid_syms, active_nodes)
            
            # Target vector y_t (prices + 0.0 constraint)
            y_t = np.append(row_prices[valid_syms].values, 0.0)
            
            # Solve least squares (guaranteed full rank due to dynamic node list pruning)
            x_t, _, rank, _ = np.linalg.lstsq(A_t, y_t, rcond=None)
            
            record = {node: val for node, val in zip(active_nodes, x_t)}
            solved_records.append(record)
            
        # Reconstruct pressures DataFrame with all discovered unique nodes
        pressures_df = pd.DataFrame(solved_records, index=timestamps).fillna(0.0)
        return pressures_df

    def solve_graph_wave_diffusion(
        self, 
        pressures_df: pd.DataFrame, 
        dtau_series: pd.Series, 
        c2: float = 0.5, 
        gamma: float = 0.1,
        coupling_k: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Integrates the Graph wave-diffusion equation:
        d2x/dtau2 + gamma * dx/dtau = -c2 * L * x + coupling_k * (x_obs - x)
        
        Args:
            pressures_df: Solved pressures (index=time, cols=nodes)
            dtau_series: Proper-time step sizes (index=time)
            c2: Wave speed squared
            gamma: Friction damping coefficient
            coupling_k: Coupling coefficient to drive the wave with observed pressures
        """
        nodes = list(pressures_df.columns)
        N = len(nodes)
        
        # Build weight matrix W from absolute correlation of pressures
        corr = pressures_df.corr().abs().values
        np.fill_diagonal(corr, 0.0)
        # Apply threshold to keep only strong relationships
        W = np.where(corr > 0.3, corr, 0.0)
        
        # Compute Degree matrix D and Laplacian L
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        # Standardize pressures to remove scale differences between assets (e.g. gold vs fiat)
        pressures_mean = pressures_df.mean().values
        pressures_std = pressures_df.std().values
        pressures_std = np.where(pressures_std == 0.0, 1.0, pressures_std)
        pressures_norm = (pressures_df.values - pressures_mean) / pressures_std
        
        # Initialize wave amplitudes (x) and velocities (v)
        wave_x = np.zeros((len(pressures_df), N))
        wave_v = np.zeros((len(pressures_df), N))
        
        # Set initial condition to solved normalized pressures
        wave_x[0] = pressures_norm[0]
        
        # Numerical integration using Euler-Cromer method with sub-stepping for stability
        sub_steps = 10
        for i in range(1, len(pressures_df)):
            dtau = dtau_series.iloc[i]
            dt_sub = dtau / sub_steps
            
            x_obs = pressures_norm[i]
            x_curr = wave_x[i - 1].copy()
            v_curr = wave_v[i - 1].copy()
            
            for _ in range(sub_steps):
                # Force includes the Laplacian coupling and the external observation driving term
                force = -c2 * np.dot(L, x_curr) + coupling_k * (x_obs - x_curr)
                v_curr += (force - gamma * v_curr) * dt_sub
                x_curr += v_curr * dt_sub
                
            wave_x[i] = x_curr
            wave_v[i] = v_curr
            
        wave_x_df = pd.DataFrame(wave_x, index=pressures_df.index, columns=[f"{n}_wave_x" for n in nodes])
        wave_v_df = pd.DataFrame(wave_v, index=pressures_df.index, columns=[f"{n}_wave_v" for n in nodes])
        
        # Compute wave energy flux: Flux = v * (L * x)
        wave_flux = np.zeros_like(wave_x)
        for i in range(len(pressures_df)):
            wave_flux[i] = wave_v[i] * np.dot(L, wave_x[i])
            
        wave_flux_df = pd.DataFrame(wave_flux, index=pressures_df.index, columns=[f"{n}_wave_flux" for n in nodes])
        
        return {
            "wave_x": wave_x_df,
            "wave_v": wave_v_df,
            "wave_flux": wave_flux_df
        }

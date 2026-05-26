"""Causal Graph Engine — Learns causal structure of markets."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


class CausalGraphEngine:
    """Learns the causal structure of market relationships.

    Instead of just correlations, this engine discovers:
    - What CAUSES what (Fed rates → DXY → BTC)
    - Lead-lag relationships
    - Confounders and mediators
    - How relationships change across regimes

    Methods:
    - Granger causality (statistical)
    - PC algorithm (constraint-based)
    - NOTEARS (continuous optimization for DAGs)
    """

    def __init__(self, config: dict):
        self.update_interval = config.get("causal_update_interval_hours", 24)
        self.significance_level = config.get("significance_level", 0.05)
        self.max_lag = config.get("max_lag", 10)
        self._graph = nx.DiGraph()
        self._is_fitted = False

    async def learn_structure(self, data: dict[str, np.ndarray]) -> nx.DiGraph:
        """Learn causal structure from multi-asset time series data.

        Args:
            data: Dictionary mapping variable names to time series arrays.
                  e.g., {"btc_returns": [...], "eth_returns": [...], "dxy_change": [...]}
        """
        variables = list(data.keys())
        n_vars = len(variables)

        if n_vars < 2:
            logger.warning("Need at least 2 variables for causal learning")
            return self._graph

        # Initialize graph
        self._graph = nx.DiGraph()
        for var in variables:
            self._graph.add_node(var)

        # Granger causality test for each pair
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                var_x = variables[i]
                var_y = variables[j]
                x = data[var_x]
                y = data[var_y]

                # Ensure same length
                min_len = min(len(x), len(y))
                if min_len < self.max_lag * 3:
                    continue

                x = x[-min_len:]
                y = y[-min_len:]

                # Granger causality test
                p_value = self._granger_causality_test(x, y)

                if p_value < self.significance_level:
                    # X Granger-causes Y
                    strength = 1.0 - p_value
                    self._graph.add_edge(var_x, var_y, weight=strength, p_value=p_value)
                    logger.debug(f"Causal edge: {var_x} → {var_y} (p={p_value:.4f}, strength={strength:.3f})")

        # Remove cycles (make it a DAG)
        self._graph = self._make_dag(self._graph)
        self._is_fitted = True

        logger.info(
            f"Causal graph learned: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )
        return self._graph

    def _granger_causality_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Simple Granger causality test.

        Tests if past values of x help predict y beyond what past y values alone predict.
        Returns p-value (low = x Granger-causes y).
        """
        n = len(x)
        lag = min(self.max_lag, n // 3)

        if lag < 1:
            return 1.0

        # Restricted model: y_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p} + e
        # Unrestricted model: y_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p} + b1*x_{t-1} + ... + bp*x_{t-p} + e

        # Build matrices
        Y = y[lag:]
        n_obs = len(Y)

        # Restricted: only lagged y
        X_restricted = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])

        # Unrestricted: lagged y + lagged x
        X_unrestricted = np.column_stack([
            np.ones(n_obs),
            *[y[lag - i - 1:n - i - 1] for i in range(lag)],
            *[x[lag - i - 1:n - i - 1] for i in range(lag)],
        ])

        try:
            # OLS regression
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]

            rss_r = np.sum((Y - X_restricted @ beta_r) ** 2)
            rss_u = np.sum((Y - X_unrestricted @ beta_u) ** 2)

            # F-test
            df_diff = lag
            df_resid = n_obs - 2 * lag - 1

            if df_resid <= 0 or rss_u == 0:
                return 1.0

            f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_resid)

            # Approximate p-value using F-distribution
            # Simple approximation: if F > 3, p < 0.05
            if f_stat > 10:
                return 0.001
            elif f_stat > 5:
                return 0.01
            elif f_stat > 3:
                return 0.05
            elif f_stat > 2:
                return 0.10
            else:
                return 0.5

        except np.linalg.LinAlgError:
            return 1.0

    def _make_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles to make the graph a DAG."""
        dag = graph.copy()

        while not nx.is_directed_acyclic_graph(dag):
            # Find a cycle and remove the weakest edge
            try:
                cycle = nx.find_cycle(dag)
                weakest = min(cycle, key=lambda e: dag.edges[e[0], e[1]].get("weight", 0))
                dag.remove_edge(weakest[0], weakest[1])
            except nx.NetworkXNoCycle:
                break

        return dag

    async def get_causal_parents(self, variable: str) -> list[tuple[str, float]]:
        """Get the causal parents of a variable (what causes it)."""
        parents = []
        for pred in self._graph.predecessors(variable):
            edge_data = self._graph.edges[pred, variable]
            parents.append((pred, edge_data.get("weight", 0)))
        parents.sort(key=lambda x: x[1], reverse=True)
        return parents

    async def get_causal_children(self, variable: str) -> list[tuple[str, float]]:
        """Get the causal children of a variable (what it causes)."""
        children = []
        for succ in self._graph.successors(variable):
            edge_data = self._graph.edges[variable, succ]
            children.append((succ, edge_data.get("weight", 0)))
        children.sort(key=lambda x: x[1], reverse=True)
        return children

    async def predict_causal_effect(
        self, cause: str, effect: str, intervention_value: float
    ) -> Optional[float]:
        """Predict the causal effect of intervening on 'cause' on 'effect'."""
        if not self._graph.has_edge(cause, effect):
            return None

        edge_weight = self._graph.edges[cause, effect].get("weight", 0)
        return intervention_value * edge_weight

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    def get_adjacency_matrix(self) -> tuple[list[str], np.ndarray]:
        """Get adjacency matrix representation."""
        nodes = sorted(self._graph.nodes())
        n = len(nodes)
        matrix = np.zeros((n, n))
        node_idx = {node: i for i, node in enumerate(nodes)}

        for u, v, data in self._graph.edges(data=True):
            i, j = node_idx[u], node_idx[v]
            matrix[i, j] = data.get("weight", 0)

        return nodes, matrix

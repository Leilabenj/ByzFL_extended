import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from typing import Optional, Dict, Any
from scipy.sparse.linalg import eigs




def build_graph_G(k,n):
    G = nx.random_regular_graph(k, n, seed=42)
    return G


# --- 3) Build Metropolis-Hastings mixing matrix W for undirected graphs ---
def metropolis_row(i, G):
    """Return (indices, weights) for row i of W."""
    deg_i = G.degree(i)
    nbrs = list(G.neighbors(i))
    weights = {}
    for j in nbrs:
        deg_j = G.degree(j)
        wij = 1.0 / (1.0 + max(deg_i, deg_j))
        weights[j] = wij
    # self-weight makes the row sum to 1
    wii = 1.0 - sum(weights.values())
    weights[i] = wii
    idx = np.fromiter(weights.keys(), dtype=int)
    vals = np.fromiter(weights.values(), dtype=float)
    return idx, vals

def build_metropolis_W(G):
    n = G.number_of_nodes()
    rows, cols, data = [], [], []
    for i in range(n):
        idx, vals = metropolis_row(i, G)
        rows.extend([i]*len(idx))
        cols.extend(idx.tolist())
        data.extend(vals.tolist())
    W = sp.csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(n, n))
    return W

def analyze_network(G: nx.Graph,
                    W: Optional[sp.csr_matrix] = None,
                    top_k_eigs: int = 6,
                    dense_threshold: int = 400) -> Dict[str, Any]:
    """
    Quick inspection of a decentralized learning graph + mixing matrix.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (assumed connected for the interpretations below).
    W : Optional[sp.csr_matrix]
        Row-stochastic mixing matrix. If None, it's built with Metropolis weights.
    top_k_eigs : int
        Number of largest-magnitude eigenvalues to report (>=2 recommended).
    dense_threshold : int
        Switch to dense eigendecomposition when n <= dense_threshold.

    Returns
    -------
    dict with keys:
        - 'W': csr_matrix
        - 'row_sums': np.ndarray
        - 'row_stochastic_ok': bool
        - 'eigs': np.ndarray (complex), sorted by |λ| desc
        - 'lambda1': complex
        - 'lambda2_abs': float
        - 'spectral_gap': float  (1 - |λ2|)
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if W is None:
        W = build_metropolis_W(G)

    # ---- Row-stochasticity check
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_stochastic_ok = np.allclose(row_sums, np.ones_like(row_sums), atol=1e-10)

    # ---- Eigen/spectral diagnostics
    # Use dense eigvals for small n (fast & exact), sparse eigs otherwise (approx top-k).
    if n <= dense_threshold:
        eigvals = np.linalg.eigvals(W.toarray())
        # sort by magnitude descending
        eig_sorted = eigvals[np.argsort(np.abs(eigvals))[::-1]]
    else:
        # sparse largest magnitude eigenvalues
        k = min(max(2, top_k_eigs), n - 1)  # ensure valid k
        vals, _ = eigs(W.T, k=k, which='LM')  # transpose is fine; spectrum same
        eig_sorted = vals[np.argsort(np.abs(vals))[::-1]]

    lambda1 = eig_sorted[0]
    lambda2_abs = float(np.abs(eig_sorted[1])) if eig_sorted.size > 1 else float("nan")
    spectral_gap = float(1.0 - lambda2_abs) if np.isfinite(lambda2_abs) else float("nan")

  
    print("\n=== Mixing matrix W checks ===")
    print(f"Row-stochastic: {row_stochastic_ok}")
    print(f"Row sums: min={row_sums.min():.12f} | max={row_sums.max():.12f}")

    # Show first few nonzero entries for first 5 rows
    preview_rows = min(5, n)
    mix_preview = []
    for i in range(preview_rows):
        row = W.getrow(i)
        for c, v in zip(row.indices.tolist(), row.data.tolist()):
            mix_preview.append({"row": i, "col": int(c), "W_ij": float(v)})
    mix_df = pd.DataFrame(mix_preview).sort_values(["row", "col"]).reset_index(drop=True)
    print("\nNonzero entries of W (rows 0–{}):".format(preview_rows-1))
    print(mix_df.head(20).to_string(index=False))

    # Eigenvalue table
    show_k = min(top_k_eigs, eig_sorted.size)
    eig_df = pd.DataFrame({
        "idx": np.arange(show_k),
        "lambda_real": np.real(eig_sorted[:show_k]),
        "lambda_imag": np.imag(eig_sorted[:show_k]),
        "|lambda|": np.abs(eig_sorted[:show_k])
    })
    print("\n=== Spectral diagnostics (sorted by |λ|) ===")
    print(eig_df.to_string(index=False))
    print(f"\nλ1 (should be ~1): {float(np.real(lambda1)):.12f} "
          f"(abs={np.abs(lambda1):.12f})")
    print(f"|λ2|: {lambda2_abs:.12f}")
    print(f"Spectral gap (1 - |λ2|): {spectral_gap:.12f}")

    return {
        "W": W,
        "row_sums": row_sums,
        "row_stochastic_ok": bool(row_stochastic_ok),
        "eigs": eig_sorted,
        "lambda1": lambda1,
        "lambda2_abs": lambda2_abs,
        "spectral_gap": spectral_gap,
    }


# --- 5) Compact per-node TopologyView you can pass to your Node ---
def topology_view(i, G, W):
    """
    Build a compact, per-node view of the topology for node i.

    Parameters
    ----------
    i : int
        Node id (assumed 0..n-1).
    G : networkx.Graph
        The full (undirected) graph.
    W : scipy.sparse.csr_matrix
        Row-stochastic mixing matrix aligned with node indices in G.

    Returns
    -------
    dict
        {
          "node_id": int,
          "neighbors": List[int],
          "degree": int,
          "mixing_row": List[Tuple[int, float]],  # includes (i, W_ii)
          "stochasticity": str,                   # "doubly" (undirected Metropolis) or "row"
        }

    Example
    -------
    >>> import networkx as nx
    >>> from scipy import sparse as sp
    >>> # 1) Build a graph and its Metropolis mixing matrix
    >>> G = nx.random_regular_graph(d=4, n=10, seed=0)
    >>> W = build_metropolis_W(G)  # your helper from graph.py
    >>>
    >>> # 2) Get the local view for node 0
    >>> view0 = topology_view(0, G, W)
    >>> view0["node_id"]
    0
    >>> view0["neighbors"]  # sorted neighbor IDs
    [*, *, *, *]  # example output, depends on the random graph
    >>> view0["degree"]
    4
    >>> view0["mixing_row"][:3]  # list of (col, weight), includes self-loop (0, W_00)
    [(0, ...), (nbr_j, ...), (nbr_k, ...)]
    >>> view0["stochasticity"]
    'doubly'
    >>>
    >>> # 3) Using it in a Node constructor (conceptual)
    >>> node_cfg = {
    ...     "node_id": view0["node_id"],
    ...     "neighbors": view0["neighbors"],
    ...     # convert mixing_row list to a dict if preferred:
    ...     "mixing_row": dict(view0["mixing_row"]),
    ...     "degree": view0["degree"],
    ...     "stochasticity": view0["stochasticity"],
    ... }
    >>> # Node(...) would then consume these fields without needing the full G/W
    """
    neighbors = sorted(list(G.neighbors(i)))
    # Extract sparse mixing row (indices and weights for non-zeros in row i)
    row = W.getrow(i)
    nz_cols = row.indices.tolist()
    nz_vals = row.data.tolist()
    mixing_row = list(zip(nz_cols, nz_vals))

    return {
        "node_id": i,
        "neighbors": neighbors,
        "degree": int(G.degree(i)),
        "mixing_row": mixing_row,  # list of (col, weight); includes self-loop
        "stochasticity": "doubly" if nx.is_regular(G) else "row",  # heuristic hint

    }
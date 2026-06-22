"""
Visualization utilities for AAlgoI pipelines.

Provides DAG visualization of algorithm selection pipelines
and algorithm performance charts.
"""

import logging

logger = logging.getLogger(__name__)

_MPL_AVAILABLE = False
_NX_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    plt = None

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    nx = None


def visualize_pipeline(
    pipeline_steps: list[str],
    save_path: str | None = "pipeline.png",
    title: str = "Pipeline DAG",
) -> str | None:
    if not _MPL_AVAILABLE or not _NX_AVAILABLE:
        logger.warning("matplotlib and networkx required for pipeline visualization")
        return None

    G = nx.DiGraph()
    for i, step in enumerate(pipeline_steps):
        G.add_node(step, layer=i)
        if i > 0:
            G.add_edge(pipeline_steps[i - 1], step)

    pos = nx.multipartite_layout(G, subset_key="layer")
    fig, ax = plt.subplots(figsize=(10, 3))
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue",
        edge_color="gray", arrows=True, ax=ax,
        node_size=3000, font_size=10,
    )
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return save_path
    return None


def plot_algorithm_comparison(
    algo_names: list[str],
    times_ms: list[float],
    save_path: str | None = "comparison.png",
) -> str | None:
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib required for comparison plots")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis([i / max(len(times_ms), 1) for i in range(len(times_ms))])
    ax.barh(range(len(algo_names)), times_ms, color=colors)
    ax.set_yticks(range(len(algo_names)))
    ax.set_yticklabels(algo_names)
    ax.set_xlabel("Execution Time (ms)")
    ax.set_title("Algorithm Performance Comparison")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return save_path
    return None

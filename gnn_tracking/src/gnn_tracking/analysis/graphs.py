import itertools
from collections import defaultdict
from typing import Iterable, NamedTuple, Sequence, Set

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch import Tensor as T
from torch_geometric.data import Data

from gnn_tracking.utils.graph_masks import get_good_node_mask


def shortest_path_length_catch_no_path(graph: nx.Graph, source, target) -> int | float:
    """Same as nx.shortest_path_length but return inf if no path exists"""
    try:
        return nx.shortest_path_length(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return float("inf")


def shortest_path_length_multi(
    graph: nx.Graph, sources: Iterable[int], targets: Iterable[int]
):
    """Shortest path for source to reach any of targets from any of the sources.
    If no connection exists, returns inf. If only target is source itself, returns 0.
    """
    if set(sources) == set(targets):
        return 0
    targets = set(targets) - set(sources)
    return min(
        [
            shortest_path_length_catch_no_path(graph, source=source, target=target)
            for source, target in itertools.product(sources, targets)
        ]
    )


def get_n_reachable(graph: nx.Graph, source: int, targets: Sequence[int]) -> int:
    """Get the number of targets that are reachable from source. The source node itself
    will not be counted!
    """
    targets = set(targets) - {source}
    return sum([nx.has_path(graph, source=source, target=target) for target in targets])


class TrackGraphInfo(NamedTuple):
    """Information about how well connected the hits of a track are in the graph.

    Here, "component" means connected component of the graph.
    "segment" means connected component of the graph that only contains hits of the
    track with the given particle ID.

    Attributes:
        pid: The particle ID of the track.
        n_hits: The number of hits in the track.
        n_segments: The number of segments of the track.
        n_hits_largest_segment: The number of hits in the largest segment of the track.
        distance_largest_segments: The shortest path length between the two largest
            segments
        n_hits_largest_component: The number of hits of the track of the biggest
            component of the track.
    """

    pid: int
    n_hits: int
    n_segments: int
    n_hits_largest_segment: int
    distance_largest_segments: int
    n_hits_largest_component: int


# A much faster way to get all connected component stats
# in one go:
# for c in nx.connected_components(gx):
#     for pid in data.particle_id:
#         node_indices = np.where(data.particle_id.cpu().numpy() == pid.item())[0]
#         assert len(node_indices) > 0
#         overlap = len(set(node_indices).intersection(set(c)))
#         if overlap > 0:
#             r[pid.item()].append(overlap)


def get_track_graph_info(
    graph: nx.Graph, particle_ids: np.ndarray, pid: int
) -> TrackGraphInfo:
    """Get information about how well connected the hits of a single particle are in the
    graph.

    Args:
        graph: networkx graph of the data
        particle_ids: The particle IDs of the hits.
        pid: Particle ID of the true track

    Returns:
        `TrackGraphInfo`
    """
    # IMPORTANT: This does not work with torch arrays! Graph will end up empty
    hits_for_pid = np.where(particle_ids == pid)[0]
    n_hits = len(hits_for_pid)
    assert n_hits > 0
    segment_subgraph = graph.subgraph(hits_for_pid)
    assert segment_subgraph.number_of_nodes() == n_hits
    segments: list[Sequence[int]] = sorted(  # type: ignore
        nx.connected_components(segment_subgraph), key=len, reverse=True
    )
    if len(segments) == 1:
        n_hits_largest_component = len(hits_for_pid)
    else:
        # We could also iterate over all PIDs, but that would be slower.
        # we already know that the segments are connected, so it's enough to
        # use one of the nodes from each one.
        component_sizes = [
            1 + get_n_reachable(graph, next(iter(segment)), hits_for_pid)
            for segment in segments
        ]
        n_hits_largest_component = max(component_sizes)
    distance_largest_segments = 0
    if len(segments) > 1:
        distance_largest_segments = shortest_path_length_multi(
            graph, sources=segments[0], targets=segments[1]
        )
        assert distance_largest_segments > 0
    n_hits_largest_segment = len(segments[0])
    assert sum(map(len, segments)) == n_hits
    assert n_hits >= n_hits_largest_component >= n_hits_largest_segment > 0, (
        n_hits,
        n_hits_largest_component,
        n_hits_largest_segment,
    )
    return TrackGraphInfo(
        pid=pid,
        n_hits=n_hits,
        n_segments=len(segments),
        n_hits_largest_segment=n_hits_largest_segment,
        distance_largest_segments=distance_largest_segments,
        n_hits_largest_component=n_hits_largest_component,
    )


def get_track_graph_info_from_data(
    data: Data,
    *,
    w: T | None = None,
    pt_thld=0.9,
    threshold: float | None = None,
    max_eta: float = 4.0,
) -> pd.DataFrame:
    """Get DataFrame of track graph information for every particle ID in the data.
    This function basically applies `get_track_graph_info` to every particle ID.

    Args:
        data: Data
        w: Edge weights. If None, no cut on edge weights
        pt_thld: pt threshold for particle IDs to consider
        threshold: Edge classification cutoff (if w is given)

    Returns:
        DataFrame with columns as in `TrackGraphInfo`
    """
    if w is not None:
        assert not torch.isnan(w).any()
        edge_mask = (w > threshold).squeeze()
        data_subgraph = data.edge_subgraph(edge_mask)
        if threshold <= 0:
            assert edge_mask.all()
            assert data_subgraph.num_edges == data.num_edges
    else:
        data_subgraph = data
    gx = torch_geometric.utils.convert.to_networkx(data_subgraph, to_undirected=True)
    assert gx.number_of_nodes() == data_subgraph.num_nodes, (
        gx.number_of_nodes(),
        data_subgraph.num_nodes,
    )
    # Can't check edge numbers because of directed vs undirected edges
    particle_ids = data.particle_id[
        get_good_node_mask(data, pt_thld=pt_thld, max_eta=max_eta)
    ].unique()
    results = []
    for pid in particle_ids:
        try:
            results.append(
                get_track_graph_info(
                    gx, data.particle_id.cpu().numpy(), pid.item()
                )._asdict()
            )
        except Exception as e:
            msg = "Error for PID"
            raise ValueError(msg, pid) from e
    return pd.DataFrame.from_records(results)


def summarize_track_graph_info(tgi: pd.DataFrame) -> dict[str, float]:
    """Summarize track graph information returned by
    `get_track_graph_info_from_data`.
    """
    return {
        "frac_segment100": sum((tgi.n_hits_largest_segment / tgi.n_hits) == 1)
        / len(tgi),
        "frac_component100": sum((tgi.n_hits_largest_component / tgi.n_hits) == 1)
        / len(tgi),
        "frac_segment50": sum((tgi.n_hits_largest_segment / tgi.n_hits) >= 0.50)
        / len(tgi),
        "frac_component50": sum((tgi.n_hits_largest_component / tgi.n_hits) >= 0.50)
        / len(tgi),
        "frac_segment75": sum((tgi.n_hits_largest_segment / tgi.n_hits) >= 0.75)
        / len(tgi),
        "frac_component75": sum((tgi.n_hits_largest_component / tgi.n_hits) >= 0.75)
        / len(tgi),
        "n_segments": tgi.n_segments.mean(),
        "frac_hits_largest_segment": (tgi.n_hits_largest_segment / tgi.n_hits).mean(),
        "frac_hits_largest_component": (
            tgi.n_hits_largest_component / tgi.n_hits
        ).mean(),
    }


class OrphanCount(NamedTuple):
    """Stats about the number of orphan nodes in a graph

    Attributes:
        n_orphan_correct: Number of orphan nodes that are actually bad nodes (low pt or
            noise)
        n_orphan_incorrect: Number of orphan nodes that are actually good nodes
        n_orphan_total: Total number of orphan nodes
    """

    n_orphan_correct: int
    n_orphan_incorrect: int
    n_orphan_total: int


def get_orphan_counts(data: Data, *, pt_thld=0.9, max_eta: float = 4.0) -> OrphanCount:
    """Count unmber of orphan nodes in a graph. See `OrphanCount` for details."""
    connected_nodes = data.edge_index.flatten().unique()
    orphan_mask = torch.zeros_like(data.particle_id, dtype=torch.bool)
    orphan_mask[connected_nodes] = False
    good_nodes_mask = get_good_node_mask(data, pt_thld=pt_thld, max_eta=max_eta)
    n_orphan_correct = torch.sum(orphan_mask & ~good_nodes_mask).item()
    n_orphan_incorrect = torch.sum(orphan_mask & good_nodes_mask).item()
    return OrphanCount(
        n_orphan_correct=n_orphan_correct,
        n_orphan_incorrect=n_orphan_incorrect,
        n_orphan_total=torch.sum(orphan_mask).item(),
    )


def get_basic_counts(
    data: Data, *, pt_thld: float = 0.9, max_eta: float = 4.0
) -> dict[str, int]:
    """Get basic counts of edges and nodes"""
    good_hits_mask = get_good_node_mask(data, pt_thld=pt_thld, max_eta=max_eta)
    good_edges_mask = (data.y == 0) & (good_hits_mask[data.edge_index[0, :]] > 0)

    return {
        "n_hits": data.num_nodes,
        "n_hits_noise": torch.sum(data.particle_id <= 0).item(),
        "n_hits_thld": torch.sum(good_hits_mask).item(),
        "n_edges": data.num_edges,
        "n_tracks": len(data.particle_id.unique()),
        "n_true_edges": torch.sum(data.y).item(),
        "n_true_edges_thld": torch.sum(good_edges_mask),
    }


def get_all_graph_construction_stats(
    data: Data, pt_thld=0.9, max_eta: float = 4.0
) -> dict[str, float]:
    """Evaluate graph construction performance for a single batch."""
    return (
        get_orphan_counts(data, pt_thld=pt_thld, max_eta=max_eta)._asdict()
        | summarize_track_graph_info(
            get_track_graph_info_from_data(data, pt_thld=pt_thld)
        )
        | get_basic_counts(data, pt_thld=pt_thld, max_eta=max_eta)
    )


def get_largest_segment_fracs(
    data: Data,
    *,
    pt_thld=0.9,
    n_particles_sampled=None,
    max_eta=4,
) -> np.ndarray:
    """A fast way to get the fraction of hits in the largest segment for each track.

    Args:
        data:
        pt_thld:
        n_particles_sampled: If not None, only consider a subsample of the particles.
            This speeds up calculation but introduces statistical fluctuations.
        max_eta: Maximum pseudorapidity

    Returns:
        Array of fractions.
    """
    # This implementation simply looks at the connected components for a graph
    # with all true edges stripped (so connected component = segment).
    basic_hit_mask = get_good_node_mask(data, pt_thld=pt_thld, max_eta=max_eta)
    unique_pids, counts = torch.unique(
        data.particle_id[basic_hit_mask], return_counts=True
    )
    pid2count = dict(zip(unique_pids.tolist(), counts.tolist()))
    if n_particles_sampled is not None:
        rand_perm = torch.randperm(len(unique_pids))
        unique_pids = unique_pids[rand_perm][:n_particles_sampled]
        basic_hit_mask &= torch.isin(data.particle_id, unique_pids)
    rdata = Data(
        edge_index=data.edge_index[:, data.y],
        particle_id=data.particle_id,
        num_nodes=len(data.particle_id),
    ).subgraph(basic_hit_mask)
    gx = nx.Graph()
    gx.add_edges_from(rdata.edge_index.T.detach().cpu().numpy())
    segments: list[Set[int]] = nx.connected_components(gx)
    pid_to_largest_segment = defaultdict(int)
    for segment in segments:
        pid = rdata.particle_id[next(iter(segment))].item()
        pid_to_largest_segment[pid] = max(
            pid_to_largest_segment[pid], len(segment) / pid2count[pid]
        )
    missing_pids = set(unique_pids.tolist()) - set(pid_to_largest_segment)
    for pid in missing_pids:
        pid_to_largest_segment[pid] = 1 / pid2count[pid]
    return np.array(list(pid_to_largest_segment.values()))


def get_cc_labels(edge_index: T, num_nodes: int) -> T:
    """Get labels for connected components of a graph."""
    gx = nx.Graph()
    gx.add_nodes_from(list(range(num_nodes)))
    gx.add_edges_from(edge_index.T.detach().cpu().numpy())
    components = nx.connected_components(gx)
    # todo: this is slow
    index_mapping = {
        node: index for index, node_set in enumerate(components) for node in node_set
    }
    return torch.tensor(
        [index_mapping[node] for node in gx.nodes()], device=edge_index.device
    )

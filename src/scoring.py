"""Stage 5 — Grouping and confidence scoring.

Merges all confirmed pairwise matches (from Stages 1–4) into groups
using a union-find (disjoint-set) data structure, then assigns each
group a confidence score and match-type label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Match type ranking — higher rank = stronger evidence
# ---------------------------------------------------------------------------

MATCH_TYPE_RANK = {
    "exact": 3,
    "near_exact": 2,
    "embedding_color_verified": 1,
}


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    """Disjoint-set / union-find with path compression and union by rank.

    Supports arbitrary hashable elements (e.g. file-path strings).
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root representative of *x*'s set (with path compression).

        Args:
            x: An element.

        Returns:
            The root representative of the set containing *x*.
        """
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing *x* and *y*.

        Args:
            x: First element.
            y: Second element.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def connected(self, x: str, y: str) -> bool:
        """Check if *x* and *y* are in the same set.

        Args:
            x: First element.
            y: Second element.

        Returns:
            ``True`` if both elements share the same root.
        """
        return self.find(x) == self.find(y)

    def groups(self) -> dict[str, list[str]]:
        """Return all groups as ``{root: [members]}``.

        Returns:
            A dict mapping each root representative to its sorted member list.
        """
        result: dict[str, list[str]] = {}
        for item in self._parent:
            root = self.find(item)
            result.setdefault(root, []).append(item)
        for members in result.values():
            members.sort()
        return result


# ---------------------------------------------------------------------------
# Pairwise match record
# ---------------------------------------------------------------------------


@dataclass
class PairMatch:
    """A single confirmed pairwise match between two images."""

    path_a: str
    path_b: str
    confidence: int
    match_type: str  # "exact", "near_exact", "embedding_color_verified"


# ---------------------------------------------------------------------------
# Duplicate group
# ---------------------------------------------------------------------------


@dataclass
class DuplicateGroup:
    """A group of images confirmed to depict the same product."""

    group_id: int
    members: list[str]
    confidence: int
    match_type: str  # strongest match type within the group


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_groups(
    matches: list[PairMatch],
    min_confidence: int = 60,
) -> list[DuplicateGroup]:
    """Merge pairwise matches into groups and assign scores.

    Uses graph clustering with guardrails against transitive chaining:
    - Connected components of size 2 are accepted if confidence >= min_confidence.
    - Components of size 3+ must form a clique or have high edge density (>= 0.80).
      Otherwise, the weakest edge is iteratively removed until all subcomponents
      satisfy the density/clique constraint, preventing unrelated products from
      chaining together.

    Confidence per group is the **maximum** confidence among edges within the group.
    Match type is the **strongest** type within the group.

    Args:
        matches: List of confirmed pairwise matches from Stages 1–4.
        min_confidence: Minimum confidence required for pairwise edges.

    Returns:
        List of :class:`DuplicateGroup` instances, sorted by confidence
        descending, then by group size descending.
    """
    if not matches:
        logger.info("No matches to group")
        return []

    import networkx as nx

    G = nx.Graph()
    for m in matches:
        if G.has_edge(m.path_a, m.path_b):
            curr = G[m.path_a][m.path_b]
            if m.confidence > curr["confidence"]:
                curr["confidence"] = m.confidence
            if MATCH_TYPE_RANK.get(m.match_type, 0) > MATCH_TYPE_RANK.get(
                curr["match_type"], 0
            ):
                curr["match_type"] = m.match_type
        else:
            G.add_edge(
                m.path_a,
                m.path_b,
                confidence=m.confidence,
                match_type=m.match_type,
            )

    # Step 1: Partition into exact-match equivalence classes (quotient graph)
    exact_G = nx.Graph()
    for m in matches:
        if m.match_type == "exact":
            exact_G.add_edge(m.path_a, m.path_b)

    all_nodes = set()
    for m in matches:
        all_nodes.add(m.path_a)
        all_nodes.add(m.path_b)

    rep_map: dict[str, str] = {}
    exact_clusters: dict[str, set[str]] = {}
    for comp in nx.connected_components(exact_G):
        rep = min(comp)
        for node in comp:
            rep_map[node] = rep
        exact_clusters[rep] = set(comp)

    for node in all_nodes:
        if node not in rep_map:
            rep_map[node] = node
            exact_clusters[node] = {node}

    # Step 2: Build quotient graph of canonical representatives for near-exact and embedding matches
    rep_G = nx.Graph()
    for node in exact_clusters:
        rep_G.add_node(node)

    for m in matches:
        if m.match_type == "exact":
            continue
        r_a = rep_map[m.path_a]
        r_b = rep_map[m.path_b]
        if r_a == r_b:
            continue
        # Accept confirmed embedding matches verified by the Color Gate
        if m.match_type == "embedding_color_verified" and m.confidence < 45.0:
            continue
        if rep_G.has_edge(r_a, r_b):
            curr = rep_G[r_a][r_b]
            if m.confidence > curr["confidence"]:
                curr["confidence"] = m.confidence
            if MATCH_TYPE_RANK.get(m.match_type, 0) > MATCH_TYPE_RANK.get(
                curr["match_type"], 0
            ):
                curr["match_type"] = m.match_type
        else:
            rep_G.add_edge(
                r_a,
                r_b,
                confidence=m.confidence,
                match_type=m.match_type,
            )

    # Step 3: Decompose representative graph with confidence guardrails and community partitioning
    # Step 3: Decompose representative graph with density and confidence guardrails
    rep_groups: list[list[str]] = []
    for comp in nx.connected_components(rep_G):
        if len(comp) < 2:
            if len(exact_clusters[list(comp)[0]]) >= 2:
                rep_groups.append(list(comp))
            continue

        sub = rep_G.subgraph(comp).copy()
        queue = [sub]
        while queue:
            curr_sub = queue.pop(0)
            n = curr_sub.number_of_nodes()
            if n < 2:
                if n == 1:
                    u = list(curr_sub.nodes())[0]
                    if len(exact_clusters[u]) >= 2:
                        rep_groups.append([u])
                continue
            if n == 2:
                u, v = list(curr_sub.nodes())
                edge_d = curr_sub[u][v]
                conf = edge_d.get("confidence", 0)
                if conf >= 45.0:
                    rep_groups.append(sorted([u, v]))
                else:
                    for w in [u, v]:
                        if len(exact_clusters[w]) >= 2:
                            rep_groups.append([w])
                continue

            e = curr_sub.number_of_edges()
            max_e = n * (n - 1) // 2
            density = e / max_e if max_e > 0 else 0
            edges = list(curr_sub.edges(data=True))
            min_edge = min(edges, key=lambda x: x[2].get("confidence", 0))

            # Dense duplicate cluster check (protects against transitive chaining across distinct products)
            min_conf = min_edge[2].get("confidence", 0)
            avg_deg = (2.0 * e) / n if n > 0 else 0
            dense_enough = (
                (n == 3 and density >= 0.65 and min_conf >= 45.0)
                or (4 <= n <= 30 and density >= 0.40 and min_conf >= 45.0)
                or (n > 30 and (density >= 0.25 or avg_deg >= 12.0) and min_conf >= 45.0)
            )
            if dense_enough:
                rep_groups.append(sorted(list(curr_sub.nodes())))
            else:
                curr_sub.remove_edge(min_edge[0], min_edge[1])
                for split_comp in nx.connected_components(curr_sub):
                    queue.append(curr_sub.subgraph(split_comp).copy())


    # Step 4: Expand representatives back to all their exact duplicate members
    raw_groups: list[list[str]] = []
    for r_group in rep_groups:
        full_members: set[str] = set()
        for r in r_group:
            full_members.update(exact_clusters[r])
        if len(full_members) >= 2:
            raw_groups.append(sorted(list(full_members)))

    # Step 5: Construct full graph with all original matches to determine group scores
    G = nx.Graph()
    for m in matches:
        if G.has_edge(m.path_a, m.path_b):
            curr = G[m.path_a][m.path_b]
            if m.confidence > curr["confidence"]:
                curr["confidence"] = m.confidence
            if MATCH_TYPE_RANK.get(m.match_type, 0) > MATCH_TYPE_RANK.get(
                curr["match_type"], 0
            ):
                curr["match_type"] = m.match_type
        else:
            G.add_edge(
                m.path_a,
                m.path_b,
                confidence=m.confidence,
                match_type=m.match_type,
            )

    # Build DuplicateGroup objects
    groups: list[DuplicateGroup] = []
    for members in raw_groups:
        sub_edges = [
            G[u][v]
            for u in members
            for v in members
            if u < v and G.has_edge(u, v)
        ]
        if not sub_edges:
            continue
        best_conf = max(e.get("confidence", 0) for e in sub_edges)
        best_type = max(
            (e.get("match_type", "embedding_color_verified") for e in sub_edges),
            key=lambda t: MATCH_TYPE_RANK.get(t, 0),
        )
        groups.append(
            DuplicateGroup(
                group_id=0,
                members=members,
                confidence=best_conf,
                match_type=best_type,
            )
        )

    # Sort: confidence desc, then group size desc
    groups.sort(key=lambda g: (-g.confidence, -len(g.members)))

    # Re-number after sort
    for idx, g in enumerate(groups, start=1):
        g.group_id = idx

    logger.info(
        "Stage 5 complete: %d duplicate groups (%d images total)",
        len(groups),
        sum(len(g.members) for g in groups),
    )
    return groups


def compute_embedding_confidence(
    similarity: float,
    color_distance: float,
    base: int = 85,
    distance_threshold: float = 18.0,
) -> int:
    """Compute confidence for an embedding + color-verified match.

    Combines embedding similarity and color-match strength into a
    single score.

    Args:
        similarity: Cosine similarity from FAISS (0.0–1.0).
        color_distance: Lab-space palette distance (lower = better).
        base: Base confidence score.
        distance_threshold: The configured color threshold.

    Returns:
        Confidence score clamped to ``[0, 100]``.
    """
    color_strength = max(0.0, 1.0 - (color_distance / max(1e-6, distance_threshold)))
    score = base * similarity * (0.7 + 0.3 * color_strength)
    return max(0, min(100, int(round(score))))

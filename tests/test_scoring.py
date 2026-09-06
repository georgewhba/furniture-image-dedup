"""Tests for scoring module (union-find grouping + confidence)."""

from __future__ import annotations

from src.scoring import (
    PairMatch,
    UnionFind,
    build_groups,
    compute_embedding_confidence,
)

# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------


class TestUnionFind:
    """Tests for the union-find data structure."""

    def test_self_find(self) -> None:
        """An element is its own root initially."""
        uf = UnionFind()
        assert uf.find("a") == "a"

    def test_union_and_connected(self) -> None:
        """Two unioned elements are connected."""
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.connected("a", "b") is True

    def test_not_connected(self) -> None:
        """Unrelated elements are not connected."""
        uf = UnionFind()
        uf.find("a")
        uf.find("b")
        assert uf.connected("a", "b") is False

    def test_transitive_union(self) -> None:
        """Union is transitive: a↔b and b↔c ⇒ a↔c."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.connected("a", "c") is True

    def test_groups(self) -> None:
        """Groups returns correct groupings."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        uf.find("e")  # singleton
        groups = uf.groups()
        # Should have 3 groups: {a,b}, {c,d}, {e}
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2, 2]

    def test_duplicate_union(self) -> None:
        """Unioning the same pair twice is idempotent."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("a", "b")
        groups = uf.groups()
        assert sum(len(v) for v in groups.values()) == 2


# ---------------------------------------------------------------------------
# build_groups
# ---------------------------------------------------------------------------


class TestBuildGroups:
    """Tests for build_groups()."""

    def test_empty_matches(self) -> None:
        """No matches → no groups."""
        assert build_groups([]) == []

    def test_single_exact_pair(self) -> None:
        """One exact pair → one group of size 2."""
        matches = [PairMatch("a.png", "b.png", confidence=100, match_type="exact")]
        groups = build_groups(matches)
        assert len(groups) == 1
        assert len(groups[0].members) == 2
        assert groups[0].confidence == 100
        assert groups[0].match_type == "exact"

    def test_transitive_grouping(self) -> None:
        """a↔b and b↔c → one group {a, b, c}."""
        matches = [
            PairMatch("a.png", "b.png", confidence=100, match_type="exact"),
            PairMatch("b.png", "c.png", confidence=90, match_type="near_exact"),
        ]
        groups = build_groups(matches)
        assert len(groups) == 1
        assert len(groups[0].members) == 3

    def test_highest_confidence_wins(self) -> None:
        """Group confidence is the maximum of its pairwise confidences."""
        matches = [
            PairMatch("a.png", "b.png", confidence=60, match_type="embedding_color_verified"),
            PairMatch("a.png", "c.png", confidence=100, match_type="exact"),
        ]
        groups = build_groups(matches)
        assert len(groups) == 1
        assert groups[0].confidence == 100

    def test_strongest_match_type_wins(self) -> None:
        """Group match type is the strongest type within the group."""
        matches = [
            PairMatch("a.png", "b.png", confidence=60, match_type="embedding_color_verified"),
            PairMatch("a.png", "c.png", confidence=95, match_type="near_exact"),
        ]
        groups = build_groups(matches)
        assert groups[0].match_type == "near_exact"

    def test_multiple_independent_groups(self) -> None:
        """Disconnected pairs form separate groups."""
        matches = [
            PairMatch("a.png", "b.png", confidence=100, match_type="exact"),
            PairMatch("c.png", "d.png", confidence=80, match_type="near_exact"),
        ]
        groups = build_groups(matches)
        assert len(groups) == 2

    def test_sorted_by_confidence_descending(self) -> None:
        """Groups are sorted by confidence descending."""
        matches = [
            PairMatch("a.png", "b.png", confidence=60, match_type="embedding_color_verified"),
            PairMatch("c.png", "d.png", confidence=100, match_type="exact"),
        ]
        groups = build_groups(matches)
        assert groups[0].confidence >= groups[1].confidence

    def test_group_ids_are_sequential(self) -> None:
        """Group IDs start at 1 and are sequential."""
        matches = [
            PairMatch("a.png", "b.png", confidence=100, match_type="exact"),
            PairMatch("c.png", "d.png", confidence=80, match_type="near_exact"),
            PairMatch("e.png", "f.png", confidence=60, match_type="embedding_color_verified"),
        ]
        groups = build_groups(matches)
        ids = [g.group_id for g in groups]
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# compute_embedding_confidence
# ---------------------------------------------------------------------------


class TestComputeEmbeddingConfidence:
    """Tests for embedding confidence scoring."""

    def test_perfect_match(self) -> None:
        """Similarity=1.0 and distance=0 → high confidence."""
        conf = compute_embedding_confidence(1.0, 0.0, base=80, distance_threshold=25.0)
        assert conf == 80

    def test_lower_similarity_lower_confidence(self) -> None:
        """Lower similarity → lower confidence."""
        conf_high = compute_embedding_confidence(0.99, 0.0, base=80)
        conf_low = compute_embedding_confidence(0.86, 0.0, base=80)
        assert conf_high > conf_low

    def test_higher_color_distance_lower_confidence(self) -> None:
        """Higher color distance → lower confidence."""
        conf_close = compute_embedding_confidence(0.95, 5.0, base=80, distance_threshold=25.0)
        conf_far = compute_embedding_confidence(0.95, 24.0, base=80, distance_threshold=25.0)
        assert conf_close > conf_far

    def test_clamped_to_0_100(self) -> None:
        """Confidence is clamped to [0, 100]."""
        conf = compute_embedding_confidence(0.1, 100.0, base=80, distance_threshold=25.0)
        assert 0 <= conf <= 100

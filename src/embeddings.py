"""Stage 3 — Visual embedding extraction and similarity search.

Uses a pretrained CLIP model (via ``open_clip``) to extract a semantic
embedding per image, then builds a FAISS index for efficient nearest-
neighbor search.  Candidates from this stage must **always** pass through
the color-verification gate (Stage 4) before being reported.

Index type rationale
--------------------
- Below ``faiss_brute_force_threshold`` (default 1 000): ``IndexFlatIP``
  (brute-force inner product).  At small scale the clustering overhead of
  IVF costs more than it saves.
- Above that threshold: ``IndexIVFFlat`` with ``nlist`` clusters and
  ``nprobe`` search probes.  Provides ~10× speedup at 10 K images and
  ~30× at 100 K, with negligible recall loss when ``nprobe`` is ≥ 10.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


class EmbeddingExtractor:
    """Wraps an open_clip CLIP model for batched image embedding extraction.

    Args:
        model_name: open_clip architecture name (e.g. ``"ViT-B-32"``).
        pretrained: Pretrained weight tag (e.g. ``"laion2b_s34b_b79k"``).
        device: PyTorch device string.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        import open_clip  # lazy — heavy import

        if torch is None:
            raise ImportError(
                "PyTorch is required for embedding extraction. Install with: pip install torch"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and torch is not None:
            num_threads = min(14, os.cpu_count() or 8)
            torch.set_num_threads(num_threads)
            logger.info("Configured PyTorch CPU with %d threads", num_threads)

        local_model = Path(__file__).resolve().parent.parent / "models" / "open_clip_model.safetensors"
        if local_model.is_file():
            arch = "ViT-B-32-quickgelu" if "quickgelu" not in model_name else model_name
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=str(local_model)
            )
            logger.info("Loaded local CLIP model weights from %s on %s", local_model, self.device)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            logger.info(
                "CLIP model loaded: %s / %s on %s", model_name, pretrained, self.device
            )

        self.model = self.model.to(self.device).eval()
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """Output embedding dimension (lazily determined)."""
        if self._dim is None:
            # Probe with a dummy image
            dummy = Image.new("RGB", (224, 224))
            emb = self.extract_single(dummy)
            self._dim = emb.shape[0]
        return self._dim

    def extract_single(self, image: Image.Image) -> np.ndarray:
        """Extract a normalized embedding for a single image.

        Args:
            image: PIL Image (any mode — converted to RGB internally).

        Returns:
            1-D float32 numpy array, L2-normalized.
        """
        img = image.convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).flatten()

    def extract_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Extract normalized embeddings for a batch of images.

        Args:
            images: List of PIL Images.
            batch_size: Number of images per forward pass.
            show_progress: Show a tqdm progress bar.

        Returns:
            Shape ``(N, D)`` float32 numpy array, each row L2-normalized.
        """
        all_embeddings: list[np.ndarray] = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings", unit="batch")

        for start in iterator:
            batch = images[start : start + batch_size]
            tensors = torch.stack(
                [self.preprocess(img.convert("RGB")) for img in batch]
            ).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------


class FaissIndex:
    """Wraps a FAISS index for nearest-neighbor embedding search.

    Automatically chooses between brute-force (``IndexFlatIP``) and IVF
    (``IndexIVFFlat``) based on the number of vectors.

    Args:
        dimension: Embedding vector dimension.
        nlist: Number of IVF clusters (ignored for brute-force).
        nprobe: Number of clusters to visit per query.
        brute_force_threshold: Use brute-force below this vector count.
        use_gpu: Use FAISS GPU resources if available.
    """

    def __init__(
        self,
        dimension: int,
        nlist: int = 100,
        nprobe: int = 10,
        brute_force_threshold: int = 1000,
        use_gpu: bool = False,
    ) -> None:
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.brute_force_threshold = brute_force_threshold
        self.use_gpu = use_gpu
        self._index: faiss.Index | None = None
        self._is_trained = False
        self._n_vectors = 0

    def build(self, embeddings: np.ndarray) -> None:
        """Build the index from a matrix of embeddings.

        Args:
            embeddings: Shape ``(N, D)`` float32 array, L2-normalized.
        """
        if faiss is None:
            raise ImportError(
                "FAISS is required for similarity search. Install with: pip install faiss-cpu"
            )

        n, d = embeddings.shape
        if d != self.dimension:
            msg = f"Expected dim {self.dimension}, got {d}"
            raise ValueError(msg)
        self._n_vectors = n

        if n < self.brute_force_threshold:
            logger.info(
                "Building brute-force index (IndexFlatIP) for %d vectors", n
            )
            self._index = faiss.IndexFlatIP(d)
        else:
            actual_nlist = min(self.nlist, n // 10)  # can't have more clusters than n/10
            actual_nlist = max(actual_nlist, 1)
            logger.info(
                "Building IVF index (nlist=%d, nprobe=%d) for %d vectors",
                actual_nlist,
                self.nprobe,
                n,
            )
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, actual_nlist)
            self._index.nprobe = self.nprobe
            self._index.train(embeddings)

        self._index.add(embeddings)
        self._is_trained = True
        logger.info("FAISS index built with %d vectors", n)

    def query(
        self,
        embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.85,
    ) -> list[tuple[int, float]]:
        """Find nearest neighbors for a single query embedding.

        Args:
            embedding: 1-D float32 array, L2-normalized.
            k: Maximum number of neighbors to return.
            threshold: Minimum cosine similarity (inner product on
                normalized vectors).

        Returns:
            List of ``(index, similarity)`` tuples, sorted by similarity
            descending.  Only results above *threshold* are returned.
        """
        if self._index is None:
            raise RuntimeError("Index not built — call build() first")

        query_vec = embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self._index.search(query_vec, k)

        results: list[tuple[int, float]] = []
        for sim, idx in zip(similarities[0], indices[0], strict=True):
            if idx == -1:
                continue
            if sim >= threshold:
                results.append((int(idx), float(sim)))

        return results

    def query_all(
        self,
        embeddings: np.ndarray,
        k: int = 10,
        threshold: float = 0.85,
    ) -> list[list[tuple[int, float]]]:
        """Batch query for all embeddings.

        Args:
            embeddings: Shape ``(N, D)`` float32 array.
            k: Max neighbors per query.
            threshold: Minimum similarity.

        Returns:
            List of neighbor lists, one per query.
        """
        if self._index is None:
            raise RuntimeError("Index not built — call build() first")

        similarities, indices = self._index.search(embeddings, k)

        results: list[list[tuple[int, float]]] = []
        for i in range(len(embeddings)):
            neighbors: list[tuple[int, float]] = []
            for sim, idx in zip(similarities[i], indices[i], strict=True):
                if idx == -1 or idx == i:
                    continue  # skip self-matches
                if sim >= threshold:
                    neighbors.append((int(idx), float(sim)))
            results.append(neighbors)

        return results

    def save(self, path: Path) -> None:
        """Serialize the index to disk.

        Args:
            path: File path for the saved index.
        """
        if faiss is None:
            raise ImportError(
                "FAISS is required to save index. Install with: pip install faiss-cpu"
            )
        if self._index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self._index, str(path))
        logger.info("FAISS index saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously saved index from disk.

        Args:
            path: Path to the saved index file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if faiss is None:
            raise ImportError(
                "FAISS is required to load index. Install with: pip install faiss-cpu"
            )
        if not path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self._index = faiss.read_index(str(path))
        self._is_trained = True
        self._n_vectors = self._index.ntotal
        logger.info("FAISS index loaded from %s (%d vectors)", path, self._n_vectors)


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------


def find_embedding_candidates(
    file_paths: list[str],
    embeddings: np.ndarray,
    config: Config,
) -> list[dict]:
    """Build a FAISS index and find all similarity candidates.

    Args:
        file_paths: Ordered list of file paths corresponding to rows
            in *embeddings*.
        embeddings: Shape ``(N, D)`` float32 array, L2-normalized.
        config: Pipeline configuration.

    Returns:
        De-duplicated list of candidate dicts, each containing:

        - ``pair``: tuple of two file paths
        - ``similarity``: cosine similarity score
    """
    index = FaissIndex(
        dimension=config.embedding_dimension,
        nlist=config.faiss_nlist,
        nprobe=config.faiss_nprobe,
        brute_force_threshold=config.faiss_brute_force_threshold,
        use_gpu=config.faiss_use_gpu,
    )
    index.build(embeddings)

    all_neighbors = index.query_all(
        embeddings,
        k=config.similarity_k,
        threshold=config.similarity_threshold,
    )

    # De-duplicate: ensure each pair appears only once
    seen_pairs: set[tuple[str, str]] = set()
    candidates: list[dict] = []

    for i, neighbors in enumerate(all_neighbors):
        for idx, sim in neighbors:
            pair = tuple(sorted([file_paths[i], file_paths[idx]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                candidates.append(
                    {
                        "pair": pair,
                        "similarity": sim,
                    }
                )

    logger.info(
        "Stage 3 complete: %d embedding candidates from %d images",
        len(candidates),
        len(file_paths),
    )
    return candidates

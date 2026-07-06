"""Native C12 retriever scaffold.

This package is intentionally independent from C7-B6 fixed prediction pools.
It may read C7-B6 train artifacts only as teacher / hard-negative evidence.
"""

from .scaffold import (
    C12SplitLoader,
    HardNegativeSampler,
    NativeRetrieverBatchBuilder,
    QueryTypeAwareRouter,
    TVRFeatureStore,
)

__all__ = [
    "C12SplitLoader",
    "HardNegativeSampler",
    "NativeRetrieverBatchBuilder",
    "QueryTypeAwareRouter",
    "TVRFeatureStore",
]

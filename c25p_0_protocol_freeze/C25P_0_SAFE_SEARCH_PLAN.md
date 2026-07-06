# Safe Search Plan

Use bounded walks with explicit max depth and global limits. Use `du --max-depth` for large directories. Do not recursively list whole disks. Do not hash videos, frame trees, feature matrices, checkpoints, or C24H score caches. Do not read frame/video bytes except tiny pilot probes explicitly requested by command-line limits.

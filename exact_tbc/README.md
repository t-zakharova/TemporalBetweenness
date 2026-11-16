# Exact temporal betweenness — Zhang et al.

Original C++/OpenMP implementation accompanying
“Efficient Exact and Approximate Betweenness Centrality Computation
for Temporal Graphs” (WWW 2024).

## Source and attribution

- Authors' repository: https://github.com/ZJUT-DBLab/TBC
- Publication page: https://raids-lab.com/publications/zhang2024efficient/
- Pinned commit: `a480804f78baf1d4da8f98ef6f58a85ace700fbc`
- Upstream directory: `exact tbc computation/`
- Local source directory: `zhang_tbc/`

The seven source files were downloaded directly from the pinned GitHub
revision without modifications. Their Git blob hashes were checked
against GitHub. The accompanying LICENSE was also verified.

This is an upstream source snapshot. It does not establish which
implementation or configuration generated the published CLGNN labels.

## Input and preprocessing

Input is a whitespace-separated edge list without a header:

```text
source target timestamp
```

For example:

```text
1 2 1
2 3 2
3 4 3
```

The loader:

- Keeps the direction of each event; reverse events are not added.
- Removes self-loops and duplicate `(source, target, timestamp)` events.
- Sorts outgoing events by timestamp.
- Replaces timestamps with their ordered ranks.
- Stops reading at the first empty line.

Timestamp ranking is performed by the loader independently of the
`needCompress` setting, which controls graph compression.

## Current build status

The unmodified snapshot contains a source-level build blocker:
`Computer.h` refers to `DataGraph::time_to_orgin_time`, but that member
is absent from `DataGraph.h`.

In addition, the original `main.cpp` has an empty `filenames` vector,
and both computation flags are false.

A runnable working copy therefore requires a separately reviewed
compatibility fix and configuration. No compilation or solver execution
was performed as part of this import.

Some path-count accumulators use `int`; overflow requires attention
before using this implementation for graphs with many shortest paths.

## Upstream build and run interface

The authors specify a 64-bit Linux environment and a C++ compiler
with OpenMP support.

After preparing a runnable working copy and selecting the graph and
settings, run from its source directory:

```bash
g++ -O3 -fopenmp -o run main.cpp
./run
```

These commands describe the upstream interface; the pristine snapshot
requires the build issue above to be resolved first.

With uncompressed shortest computation enabled, the program prints
`node_id score` lines alongside loading and timing messages.
The scores are raw, without normalization.

## License and modifications

The upstream code is distributed under Apache License 2.0; see
`zhang_tbc/LICENSE`.

Retain the license and existing attribution notices when redistributing
the code. Modified upstream files must carry prominent change notices.
No NOTICE file was present in the checked upstream root or exact-solver
directory.

The imported source files remain unchanged. Future adaptations and
their validation should be documented separately.

from collections import defaultdict
from fractions import Fraction
from pathlib import Path


OFFICE = Path(__file__).resolve().parents[2] / "test_data" / "sp_office.txt"
CUTOFF = 145780


def read_office():
    by_time = defaultdict(set)
    nodes = set()

    with OFFICE.open() as graph:
        for line in graph:
            u, v, time = map(int, line.split())
            if time <= CUTOFF:
                nodes.update((u, v))
                if u != v:
                    by_time[time].add((u, v))

    batches = [by_time[time] for time in sorted(by_time)]
    return sorted(nodes), batches


def shortest(batches, source, removed=None):
    # best[v] = (minimum hops to v so far, number of such paths)
    best = {source: (0, 1)}

    for edges in batches:
        updates = {}
        for u, v in edges:
            if u == removed or v == removed or u not in best:
                continue

            hops, count = best[u]
            candidate = hops + 1
            previous = updates.get(v, best.get(v))

            if previous is None or candidate < previous[0]:
                updates[v] = (candidate, count)
            elif candidate == previous[0]:
                updates[v] = (candidate, previous[1] + count)

        best.update(updates)

    return best


def reference_tbc(nodes, batches):
    scores = {v: Fraction(0) for v in nodes}

    for source in nodes:
        original = shortest(batches, source)

        for v in original:
            if v == source:
                continue
            avoiding = shortest(batches, source, removed=v)

            for target, (length, count) in original.items():
                if target == source or target == v:
                    continue

                other = avoiding.get(target)
                surviving = 0
                if other is not None and other[0] == length:
                    surviving = other[1]

                scores[v] += Fraction(count - surviving, count)

    return scores


if __name__ == "__main__":
    nodes, batches = read_office()
    event_count = sum(len(edges) for edges in batches)
    if (event_count, len(nodes)) != (2000, 81):
        raise ValueError(f"Unexpected Office slice: {event_count} events, {len(nodes)} nodes")

    for node, score in reference_tbc(nodes, batches).items():
        print(node, float(score))

# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Maximum-cardinality, one-to-one event matching and micro metrics."""
import statistics


def candidate_pairs(references, episodes, max_window):
    return {(i, j) for i, episode in enumerate(episodes) for j, ref in enumerate(references)
            if any(abs(prefix["time"] - ref["time"]) <= max_window for prefix in episode["prefixes"])}


def validate_pairs(pairs, expected, episodes):
    if not isinstance(pairs, list) or len(pairs) != len(expected):
        raise ValueError("Judge must return exactly the requested pairs")
    actual = set()
    for pair in pairs:
        i, j, k = pair["episode_index"], pair["reference_index"], pair["first_correct_prefix_index"]
        if type(i) is not int or type(j) is not int or (i, j) not in expected or (i, j) in actual:
            raise ValueError("Unknown, duplicate or invalid judge pair")
        if k is not None and (type(k) is not int or not 0 <= k < len(episodes[i]["prefixes"])):
            raise ValueError("Invalid first-correct-prefix index")
        actual.add((i, j))
    if actual != expected:
        raise ValueError("Incomplete judge coverage")


def match(references, episodes, window, pairs=None, no_early=False, not_before=0.):
    edges = {i: [] for i in range(len(episodes))}
    deltas = {}
    if pairs is None:
        candidates = [(i, j, episode["time"]) for i, episode in enumerate(episodes) for j in range(len(references))]
    else:
        candidates = [(p["episode_index"], p["reference_index"],
                       episodes[p["episode_index"]]["prefixes"][p["first_correct_prefix_index"]]["time"])
                      for p in pairs if p["first_correct_prefix_index"] is not None]
    for i, j, timestamp in candidates:
        delta = timestamp - references[j]["time"]
        within_window = (0 <= delta <= window) if no_early else (abs(delta) <= window)
        if timestamp >= not_before and within_window:
            edges[i].append(j)
            deltas[i, j] = delta
    for i in edges:
        edges[i].sort(key=lambda j: (abs(deltas[i, j]), j))
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in sorted(edges, key=lambda i: (len(edges[i]), i)):
        augment(i, set())
    # Cardinality is optimal; mean latency is not an optimal-cost assignment.
    return dict(tp=len(owner), fp=len(episodes) - len(owner), fn=len(references) - len(owner),
                deltas=[deltas[i, j] for j, i in owner.items()])


def aggregate(items):
    result = {key: sum(item[key] for item in items) for key in ("tp", "fp", "fn")}
    tp, fp, fn = (result[k] for k in ("tp", "fp", "fn"))
    deltas = [d for item in items for d in item["deltas"]]
    result.update(precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else 0.,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                  mean_abs_delta=statistics.mean(abs(d) for d in deltas) if deltas else None,
                  median_delta=statistics.median(deltas) if deltas else None)
    return result

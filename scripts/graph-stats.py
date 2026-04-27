from datasets import load_dataset
import pandas as pd

def compute_dependency_stats(df: pd.DataFrame, cycle_log_path: str = "cycles.log", n_rows: int | None = None):
    # Preprocess: uuid -> object_links
    graph = df["object_links"].to_dict()

    # Memoization cache
    cache = {}

    # Cycle logging
    cycle_log_buffer = []

    def dfs(uuid, path):
        # Cycle detection
        if uuid in path:
            cycle_start = path.index(uuid)
            cycle = path[cycle_start:] + [uuid]
            cycle_log_buffer.append("Cycle detected: " + " -> ".join(cycle) + "\n")
            return {
                "unique_nodes": set(),
                "max_depth": 0,
                "branch_sum": 0,
                "node_count": 0
            }

        # Cached result
        if uuid in cache:
            return cache[uuid]

        path.append(uuid)

        object_links = graph.get(uuid, [])
        current_branching = len(object_links)

        all_unique = set()
        max_depth = 0
        branch_sum = current_branching
        node_count = 1

        for slot in object_links:
            for dep in slot:
                if dep not in graph:
                    continue

                result = dfs(dep, path)

                all_unique.add(dep)
                all_unique.update(result["unique_nodes"])

                max_depth = max(max_depth, 1 + result["max_depth"])
                branch_sum += result["branch_sum"]
                node_count += result["node_count"]

        path.pop()

        result = {
            "unique_nodes": all_unique,
            "max_depth": max_depth,
            "branch_sum": branch_sum,
            "node_count": node_count
        }

        cache[uuid] = result
        return result

    # Run per node
    rows = []

    for uuid in df.index if n_rows is None else df.index[:n_rows]:
        result = dfs(uuid, [])

        unique_nodes = result["unique_nodes"]
        branch_sum = result["branch_sum"]
        node_count = result["node_count"]

        avg_branching = branch_sum / node_count if node_count > 0 else 0

        rows.append({
            "uuid": uuid,
            "branching_factor_avg": avg_branching,
            "max_depth": result["max_depth"],
            "dependency_mass": len(unique_nodes),
            "unique_nodes_visited": unique_nodes,
        })

    with open(cycle_log_path, "w") as f:
        f.write("\n".join(cycle_log_buffer))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    ds = load_dataset('offendo/math-atlas', split='train')
    df = ds.to_pandas().set_index('uuid')
    dd = compute_dependency_stats(df)
    dd.to_json('math-atlas-dependency-stats.json')

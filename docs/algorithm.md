# Quadric Error Metrics (QEM) Algorithm

## Mathematical Background

The QEM algorithm, introduced by Garland & Heckbert (1997), simplifies triangle meshes by iteratively collapsing edges while minimizing geometric error. Each vertex $v$ is associated with a symmetric $4\times4$ quadric matrix $Q$, representing the sum of squared distances from $v$ to the planes of its adjacent faces. For a plane $ax + by + cz + d = 0$, the quadric is $K_p = [a, b, c, d]^T [a, b, c, d]$.

## Edge Collapse Strategy

1. **Compute Q matrices** for all vertices.
2. **Select valid pairs** (typically mesh edges).
3. **For each pair $(v_1, v_2)$:**
   - Compute $Q = Q_1 + Q_2$.
   - Find optimal contraction target $\bar{v}$ minimizing $\bar{v}^T Q \bar{v}$.
   - The contraction cost is this minimum error.
4. **Build a heap** of all pairs, keyed by cost.
5. **Iteratively:**
   - Remove the lowest-cost pair.
   - Contract the edge, update mesh connectivity and quadrics.
   - Update costs for affected pairs.

## Implementation Details

- **Quadrics:** Computed from face planes using vertex adjacency.
- **Optimal Position:** $\bar{v}$ is found by solving $Q' \bar{v} = [0,0,0,1]$, where $Q'$ is $Q$ with the last row replaced by $[0,0,0,1]$.
- **Heap:** Python's `heapq` is used to efficiently select the minimum-cost edge.
- **Edge Contraction:** Updates vertex positions, removes faces, and updates adjacency and quadrics.
- **Numerical Stability:** If $Q'$ is singular, the midpoint is used.
- **Data Structures:**
  - `Mesh` class holds vertices, faces, adjacency, and normals.
  - Main logic in `quadric_edge_collapse/quadric_edge_collapse_tri.py`.

## Complexity Analysis

- **Initialization:** $O(n + m)$ for $n$ vertices, $m$ faces.
- **Heap Operations:** Each contraction is $O(\log e)$ for $e$ edges.
- **Edge Contractions:** Up to $O(n)$ contractions, each updating $O(d)$ neighbors ($d$ = degree).
- **Overall:** $O(n \log n + m)$ for typical meshes.

## References
- Garland, M., & Heckbert, P. S. (1997). Surface Simplification Using Quadric Error Metrics. SIGGRAPH.
- [Original paper](https://mgarland.org/files/papers/quadrics.pdf)

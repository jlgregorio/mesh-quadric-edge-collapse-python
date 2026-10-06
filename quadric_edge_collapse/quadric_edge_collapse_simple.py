import numpy as np

from .mesh import Mesh


def quadric_edge_collapse_decimation(mesh, target_vertex_count):
    """Mesh simplification using a quadric based edge-collapse strategy."""


    while mesh.vertices_number > target_vertex_count:

        # 1. Compute vertex quadrics
        Q_matrices = compute_initial_quadrics(mesh)

        # 2. Select all valid pairs
        valid_pairs = mesh.edges

        # 3. Compute v_bar and cost for each valid pair
        best_pair = None
        best_v_bar = None
        best_cost = np.inf

        for pair in valid_pairs:

            v_1, v_2 = pair
            v_bar, cost = compute_pair_contraction(
                Q_matrices[v_1],
                Q_matrices[v_2],
                mesh.vertices[v_1],
                mesh.vertices[v_2]
            )

        # 4. Select cheapest edge
            if cost < best_cost:
                best_cost = cost
                best_v_bar = v_bar[:3] # v_bar in homogeneous coordinates [x, y, z, 1]
                best_pair = pair

        # 5. Collapse edge
        mesh = collapse_edge(mesh, best_pair, best_v_bar)

    return mesh


def compute_initial_quadrics(mesh):
    """Compute Q matrix for each vertex of the initial mesh"""

    # Cartersian coordinates (a, b, c, d) of planes formed by the faces of the mesh
    dists = - np.sum(mesh.faces_normals * mesh.faces_centers, axis=1)
    planes_coords = np.hstack([mesh.faces_normals, dists[:, None]])

    # The error quadric Q for each vertex is the sum of its fundamental quadrics
    Q_matrices = np.zeros((mesh.vertices_number, 4, 4))
    for i, faces in enumerate(mesh.vertices_faces):
        Q_matrices[i] = planes_coords[faces].T @ planes_coords[faces]
    
    return Q_matrices
        

def compute_pair_contraction(Q_1, Q_2, p_1, p_2):
    """Optimal contraction position and cost for an edge."""

    # Approximation matrix Q_bar = Q_0 + Q_1
    Q_bar = Q_1 + Q_2
    # Optimal position of v is the one that minimizes v.T @ Q @ v
    # which is found by solving Q' @ v = [0, 0, 0, 1]
    Q_prime = Q_bar.copy()
    Q_prime[3, :] = [0., 0., 0., 1.]
    try:
        v_h = np.linalg.solve(Q_prime, [0., 0., 0., 1.])
    # Use midpoint if matrix is ill-conditioned
    except np.linalg.LinAlgError:
        midpoint = 0.5 * (p_1 + p_2)
        v_h = np.array([midpoint[0], midpoint[1], midpoint[2], 1.])
    # Cost of contracting a pair
    cost = max(0.0, float(v_h @ Q_bar @ v_h))

    return v_h[:3], cost

def collapse_edge(mesh, pair, v_bar):
    
    v_1, v_2 = pair

    new_vertices = mesh.vertices.copy()
    new_faces = mesh.faces.copy()

    new_vertices[v_1] = v_bar

    new_faces[new_faces==v_2] = v_1

    valid = (
        (new_faces[:,0] != new_faces[:,1]) &
        (new_faces[:,1] != new_faces[:,2]) &
        (new_faces[:,2] != new_faces[:,0])
    )

    new_faces = new_faces[valid]

    used = np.unique(new_faces)

    mapping = np.full(len(new_vertices), -1, dtype=int)
    mapping[used] = np.arange(len(used))

    new_vertices = new_vertices[used]
    new_faces = mapping[new_faces]

    return Mesh(new_vertices, new_faces)

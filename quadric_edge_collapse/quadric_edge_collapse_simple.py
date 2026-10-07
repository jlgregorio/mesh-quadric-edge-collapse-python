import numpy as np

from .mesh import Mesh


class QuadricEdgeCollapseSimplifier:
    """
    Simplifies a mesh using a quadric-based edge-collapse strategy.

    Parameters
    ----------
    mesh : Mesh
        The input mesh to be simplified.
    target_vertex_count : int
        The desired number of vertices after simplification.
    """
    def __init__(self, mesh: 'Mesh', target_vertex_count: int):
        self.mesh = mesh
        self.target_vertex_count = target_vertex_count

    def simplify(self) -> 'Mesh':
        """
        Perform the simplification process.

        Returns
        -------
        Mesh
            The simplified mesh.
        """
        mesh = self.mesh
        while mesh.vertices_number > self.target_vertex_count:
            Q_matrices = self._compute_initial_quadrics(mesh)
            valid_pairs = mesh.edges
            best_pair = None
            best_v_bar = None
            best_cost = np.inf
            for pair in valid_pairs:
                v_1, v_2 = pair
                v_bar, cost = self._compute_pair_contraction(
                    Q_matrices[v_1],
                    Q_matrices[v_2],
                    mesh.vertices[v_1],
                    mesh.vertices[v_2]
                )
                if cost < best_cost:
                    best_cost = cost
                    best_v_bar = v_bar[:3]
                    best_pair = pair
            mesh = self._collapse_edge(mesh, best_pair, best_v_bar)
        return mesh



    def _compute_initial_quadrics(self, mesh: 'Mesh') -> np.ndarray:
        """
        Compute the quadric error matrix Q for each vertex of the initial mesh.

        Parameters
        ----------
        mesh : Mesh
            The input mesh.

        Returns
        -------
        Q_matrices : np.ndarray
            Array of shape (num_vertices, 4, 4) containing the quadric matrices for each vertex.
        """
        dists = -np.sum(mesh.faces_normals * mesh.faces_centers, axis=1)
        planes_coords = np.hstack([mesh.faces_normals, dists[:, None]])
        Q_matrices = np.zeros((mesh.vertices_number, 4, 4))
        for i, faces in enumerate(mesh.vertices_faces):
            Q_matrices[i] = planes_coords[faces].T @ planes_coords[faces]
        return Q_matrices

        

    def _compute_pair_contraction(self, Q_1: np.ndarray, Q_2: np.ndarray, p_1: np.ndarray, p_2: np.ndarray) -> tuple:
        """
        Compute the optimal contraction position and cost for an edge.

        Parameters
        ----------
        Q_1 : np.ndarray
            Quadric matrix for vertex v_1.
        Q_2 : np.ndarray
            Quadric matrix for vertex v_2.
        p_1 : np.ndarray
            Position of vertex v_1 (length 3).
        p_2 : np.ndarray
            Position of vertex v_2 (length 3).

        Returns
        -------
        v_h : np.ndarray
            Optimal contraction position in homogeneous coordinates (length 3).
        cost : float
            Cost of contracting the pair.
        """
        Q_bar = Q_1 + Q_2
        Q_prime = Q_bar.copy()
        Q_prime[3, :] = [0., 0., 0., 1.]
        try:
            v_h = np.linalg.solve(Q_prime, [0., 0., 0., 1.])
        except np.linalg.LinAlgError:
            midpoint = 0.5 * (p_1 + p_2)
            v_h = np.array([midpoint[0], midpoint[1], midpoint[2], 1.])
        cost = max(0.0, float(v_h @ Q_bar @ v_h))
        return v_h[:3], cost


    def _collapse_edge(self, mesh: 'Mesh', pair: tuple, v_bar: np.ndarray) -> 'Mesh':
        """
        Collapse the edge defined by a pair of vertices and update the mesh.

        Parameters
        ----------
        mesh : Mesh
            The input mesh.
        pair : tuple
            Pair of vertex indices (v_1, v_2) to collapse.
        v_bar : np.ndarray
            New vertex position for the collapsed edge (length 3).

        Returns
        -------
        Mesh
            The updated mesh after edge collapse.
        """
        v_1, v_2 = pair
        new_vertices = mesh.vertices.copy()
        new_faces = mesh.faces.copy()
        new_vertices[v_1] = v_bar
        new_faces[new_faces == v_2] = v_1
        valid = (
            (new_faces[:, 0] != new_faces[:, 1]) &
            (new_faces[:, 1] != new_faces[:, 2]) &
            (new_faces[:, 2] != new_faces[:, 0])
        )
        new_faces = new_faces[valid]
        used = np.unique(new_faces)
        mapping = np.full(len(new_vertices), -1, dtype=int)
        mapping[used] = np.arange(len(used))
        new_vertices = new_vertices[used]
        new_faces = mapping[new_faces]
        return Mesh(new_vertices, new_faces)


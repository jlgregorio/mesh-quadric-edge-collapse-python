import numpy as np

class Mesh:
    """A basic class for handling triangular meshes."""

    def __init__(self, vertices, faces):
        """Args:
            vertices: the vertices of the mesh (n x 3).
            faces: the faces of the mesh (m x 3).
        """

        self.vertices = vertices
        self.faces = faces
        # Flags for lazy deletion
        self.vertex_alive = np.ones(self.vertex_number, dtype=bool)
        self.face_alive = np.ones(self.face_number, dtype=bool)
        # Properties
        self.vertex_faces = self._build_vertex_faces()
        self.vertex_neighbors = self._build_vertex_neighbors()
        self.edge_faces = self._build_edge_faces()
    
    @property
    def vertex_number(self) -> int:
        return len(self.vertices)

    @property
    def face_number(self) -> int:
        return len(self.faces)

    @property
    def alive_face_number(self):
        return np.count_nonzero(self.face_alive)

    @property
    def alive_vertex_number(self):
        return np.count_nonzero(self.vertex_alive)

    @staticmethod
    def _edge(v_1, v_2) -> tuple:
        """Canonical edge representation."""
        return (v_1, v_2) if v_1 < v_2 else (v_2, v_1)

    def _build_vertex_neighbors(self):
        """Vertices connected by an edge for each vertex."""
        
        neighbors = [set() for _ in range(self.vertex_number)]
        
        for face_id, (v_0, v_1, v_2) in enumerate(self.faces):
            
            if not self.face_alive[face_id]:
                continue
            
            neighbors[v_0].update((v_1, v_2))
            neighbors[v_1].update((v_0, v_2))
            neighbors[v_2].update((v_0, v_1))
        
        return neighbors
    
    def _build_vertex_faces(self):
        """Faces that contain each vertex."""

        vf = [set() for _ in range(self.vertex_number)]
        
        for face_id, (v_0, v_1, v_2) in enumerate(self.faces):

            if not self.face_alive[face_id]:
                continue

            vf[v_0].add(face_id)
            vf[v_1].add(face_id)
            vf[v_2].add(face_id)

        return vf

    def _build_edge_faces(self) -> dict:
        """Faces adjacent to each edge."""

        edge_faces = {}
        
        for face_id, (v_0, v_1, v_2) in enumerate(self.faces):

            if not self.face_alive[face_id]:
                continue
        
            for edge in (self._edge(v_0, v_1), self._edge(v_1, v_2), self._edge(v_2, v_0)):
                
                edge_faces.setdefault(edge, set()).add(face_id)
        
        return edge_faces

    def incident_faces(self, vertex):
        return self.vertex_faces[vertex]

    def neighbor_vertices(self, vertex):
        return self.vertex_neighbors[vertex]

    def adjacent_faces(self, edge):
        return self.edge_faces.get(edge, [])

    def is_boundary_edge(self, edge):
        return len(self.adjacent_faces(edge)) == 1

    def is_manifold_edge(self, edge):
        return len(self.adjacent_faces(edge)) <= 2

    def collapse_edge(self, v_keep, v_remove):
        """Collapse edge (v_keep, v_remove) into v_keep."""

        if not self.vertex_alive[v_keep]:
            raise ValueError(f"Vertex {v_keep} is dead.")

        if not self.vertex_alive[v_remove]:
            raise ValueError(f"Vertex {v_remove} is dead.")

        affected_vertices = (
            {v_keep, v_remove}
            | self.vertex_neighbors[v_keep]
            | self.vertex_neighbors[v_remove]
        )

        # Rewire faces
        incident_faces = list(self.vertex_faces[v_remove])
        
        for face_id in incident_faces:

            if not self.face_alive[face_id]:
                continue

            old_face = self.faces[face_id].copy()

            self._remove_face_connectivity(face_id, old_face)

            new_face = np.where(old_face==v_remove, v_keep, old_face)

            # Degenerate triangle
            if len(set(new_face)) < 3:

                self.face_alive[face_id] = False
                continue
                
            self.faces[face_id] = new_face

            self._add_face_connectivity(face_id, new_face)

        # Kill vertex
        self.vertex_alive[v_remove] = False
        self.vertex_neighbors[v_remove].clear()

        # Vertex no longer participates in topology
        self.vertex_faces[v_remove].clear()

        # Recompute local neighbors only
        self._recompute_neighbors(affected_vertices)

        return affected_vertices

    def _add_face_connectivity(self, face_id, face):
        """Add the contribution of a face to connectivity."""

        v_0, v_1, v_2 = face

        # Vertex -> faces
        self.vertex_faces[v_0].add(face_id)
        self.vertex_faces[v_1].add(face_id)
        self.vertex_faces[v_2].add(face_id)

        # Edge -> faces
        for edge in (self._edge(v_0, v_1), self._edge(v_1, v_2), self._edge(v_2, v_0)):
            self.edge_faces.setdefault(edge, set()).add(face_id)

    def _remove_face_connectivity(self, face_id, face):
        """Remove the contribution of a face from connectivity."""

        v_0, v_1, v_2 = face

        # Vertex -> faces
        self.vertex_faces[v_0].discard(face_id)
        self.vertex_faces[v_1].discard(face_id)
        self.vertex_faces[v_2].discard(face_id)

        # Edge -> faces
        for edge in (self._edge(v_0, v_1), self._edge(v_1, v_2), self._edge(v_2, v_0)):

            if edge not in self.edge_faces:
                continue

            self.edge_faces[edge].discard(face_id)

            if not self.edge_faces[edge]:
                del self.edge_faces[edge]

    def _recompute_neighbors(self, vertices):
        """Recompute neighbors."""

        for v in vertices:

            if not self.vertex_alive[v]:
                self.vertex_neighbors[v] = set()
                continue

            neighbors = set()

            for face_id in self.vertex_faces[v]:

                face = self.faces[face_id]

                for u in face:
                    if u != v:
                        neighbors.add(u)

            self.vertex_neighbors[v] = neighbors

    def compact(self):
        """Remove dead vertices and faces and renumber indices."""

        # Vertex mapping
        alive_vertices = np.flatnonzero(self.vertex_alive)
        old_to_new = np.full(self.vertex_number, -1, dtype=np.int64)
        old_to_new[alive_vertices] = np.arange(len(alive_vertices))

        # Compact vertices
        new_vertices = self.vertices[alive_vertices]

        # Compact faces
        alive_faces = np.flatnonzero(self.face_alive)

        new_faces = self.faces[alive_faces]

        # Remap vertex indices
        new_faces = old_to_new[new_faces]

        # Safety check
        if np.any(new_faces < 0):
            raise RuntimeError("A live face references a deleted vertex.")

        # Replace mesh data
        self.vertices = new_vertices
        self.faces = new_faces

        # Reset alive flags
        self.vertex_alive = np.ones(len(self.vertices), dtype=bool)
        self.face_alive = np.ones(len(self.faces), dtype=bool)

        # Rebuild connectivity
        self.vertex_faces = self._build_vertex_faces()
        self.vertex_neighbors = self._build_vertex_neighbors()
        self.edge_faces = self._build_edge_faces()

        return old_to_new

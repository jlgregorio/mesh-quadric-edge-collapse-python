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
    
    @property
    def vertices_number(self):
        """The number of vertices of the mesh (n)."""

        return len(self.vertices)

    @property
    def vertices_adjacency(self):
        """Vertices connected by an edge for each vertex."""
        
        neighbors = [[] for _ in range(self.vertices_number)]
        
        for v_1, v_2 in self.edges:
            neighbors[v_1].append(v_2)
            neighbors[v_2].append(v_1) 
        
        return neighbors
    
    @property
    def vertices_faces(self):
        """Faces that contain each vertex."""

        vf = [list() for _ in range(len(self.vertices))]
        
        for i, f in enumerate(self.faces):
            vf[f[0]].append(i)
            vf[f[1]].append(i)
            vf[f[2]].append(i)

        return vf

    @property
    def faces_number(self):
        """The number of faces of the mesh (m)."""

        return len(self.faces)
    
    @property
    def faces_normals(self):
        """The unit normal vector for each face."""
        
        v1 = self.faces_coords[:, 1] - self.faces_coords[:, 0]
        v2 = self.faces_coords[:, 2] - self.faces_coords[:, 0]
        face_normals = np.cross(v1, v2).astype(np.float64)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        face_normals /= norms
        return np.nan_to_num(face_normals)

    @property
    def faces_area(self):
        """The area of each face."""

        v1 = self.faces_coords[:, 1] - self.faces_coords[:, 0]
        v2 = self.faces_coords[:, 2] - self.faces_coords[:, 0]

        # The norm of the vector resulting from the cross product corresponds
        # to the double of the triangle area.
        return .5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    
    @property
    def faces_coords(self):
        """The coordinates of each face."""

        return self.vertices[self.faces]

    @property
    def faces_centers(self):
        """The center of each face."""

        return self.faces_coords.mean(axis=1)
    
    @property
    def faces_edges(self):
        """The edges of each face."""

        return self.faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    
    @property
    def edges(self):
        """Unique unoriented edges."""

        return np.unique(np.sort(self.faces_edges, axis=1), axis=0)
    

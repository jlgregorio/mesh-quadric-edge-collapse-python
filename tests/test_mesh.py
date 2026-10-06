import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tempfile

import numpy as np

from quadric_edge_collapse.mesh import Mesh
from utils.off_format import load_off, save_off


class TestMesh(unittest.TestCase):
    def setUp(self):
        # Simple tetrahedron mesh
        self.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.faces = np.array([
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3]
        ])
        self.mesh = Mesh(self.vertices, self.faces)

    def test_vertices_faces_numbers(self):
        self.assertEqual(self.mesh.vertices_number, 4)
        self.assertEqual(self.mesh.faces_number, 4)

    def test_adjacency_and_edges(self):
        edges = self.mesh.edges
        self.assertTrue(np.all(edges >= 0))
        self.assertTrue(np.all(edges < 4))
        self.assertEqual(len(edges), 6)
        adj = self.mesh.vertices_adjacency
        self.assertEqual(len(adj), 4)

    def test_faces_normals_and_area(self):
        normals = self.mesh.faces_normals
        self.assertEqual(normals.shape, (4, 3))
        areas = self.mesh.faces_area
        self.assertEqual(areas.shape, (4,))
        self.assertTrue(np.all(areas > 0))

    def test_faces_coords_and_centers(self):
        coords = self.mesh.faces_coords
        self.assertEqual(coords.shape, (4, 3, 3))
        centers = self.mesh.faces_centers
        self.assertEqual(centers.shape, (4, 3))

    def test_off_io(self):
        with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.off') as tmp:
            save_off(tmp.name, self.mesh)
            tmp.close()
            v, f = load_off(tmp.name)
            os.remove(tmp.name)
        self.assertEqual(v.shape, (4, 3))
        self.assertEqual(f.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()

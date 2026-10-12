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
        self.assertEqual(self.mesh.vertex_number, 4)
        self.assertEqual(self.mesh.face_number, 4)

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

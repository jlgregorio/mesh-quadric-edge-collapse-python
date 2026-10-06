import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_off
from quadric_edge_collapse import Mesh, quadric_edge_collapse_decimation
from quadric_edge_collapse.quadric_edge_collapse_simple import compute_initial_quadrics

class TestCollapseBunny(unittest.TestCase):
    def test_bunny_target(self):
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent
        DATA_DIR = ROOT / "data"
        target = 8140
        vertices, faces = load_off(DATA_DIR / "stanford_bunny.off")
        mesh = Mesh(vertices, faces)
        collapsed_mesh = quadric_edge_collapse_decimation(mesh, target)
        self.assertEqual(collapsed_mesh.vertices_number, target)

class TestCollapseSynthetic(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        self.faces = np.array([
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3]
        ], dtype=int)
        self.mesh = Mesh(self.vertices, self.faces)

    def test_simplification(self):
        collapsed = quadric_edge_collapse_decimation(self.mesh, 3)
        self.assertEqual(collapsed.vertices_number, 3)
        self.assertTrue(collapsed.faces_number <= 4)

    def test_quadric_error_update(self):
        Q = compute_initial_quadrics(self.mesh)
        self.assertEqual(Q.shape, (4, 4, 4))

if __name__ == "__main__":
    unittest.main()

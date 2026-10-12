import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_off
from quadric_edge_collapse import Mesh, QuadricEdgeCollapseDecimator


class TestCollapseBunny(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent
        DATA_DIR = ROOT / "data"
        self.vertices, self.faces = load_off(DATA_DIR / "stanford_bunny.off")
    
    def test_bunny_high_target(self):
        mesh = Mesh(self.vertices, self.faces)
        target = int(0.9 * mesh.vertex_number)
        decimator = QuadricEdgeCollapseDecimator(mesh, target)
        collapsed_mesh = decimator.simplify()
        self.assertEqual(collapsed_mesh.vertex_number, target)

    def test_bunny_mid_target(self):
        mesh = Mesh(self.vertices, self.faces)
        target = int(0.5 * mesh.vertex_number)
        decimator = QuadricEdgeCollapseDecimator(mesh, target)
        collapsed_mesh = decimator.simplify()
        self.assertEqual(collapsed_mesh.vertex_number, target)

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
        simplifier = QuadricEdgeCollapseDecimator(self.mesh, 3)
        collapsed = simplifier.simplify()
        self.assertEqual(collapsed.vertex_number, 3)
        self.assertTrue(collapsed.face_number <= 4)

    def test_quadric_error_update(self):
        simplifier = QuadricEdgeCollapseDecimator(self.mesh, 3)
        simplifier._compute_initial_quadrics()
        self.assertEqual(simplifier.vertex_quadrics.shape, (4, 4, 4))

if __name__ == "__main__":
    unittest.main()

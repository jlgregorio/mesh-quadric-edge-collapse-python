import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quadric_edge_collapse import Mesh, quadric_edge_collapse_decimation
from utils import load_off


class TestCollapse(unittest.TestCase):

    def test_bunny_target(self):
        
        target = 5000

        vertices, faces = load_off("./data/stanford_bunny.off")
        mesh = Mesh(vertices, faces)
        collapsed_mesh = quadric_edge_collapse_decimation(mesh, target)

        self.assertEqual(collapsed_mesh.vertices_number, target)


if __name__ == "__main__":
    unittest.main()


from quadric_edge_collapse import Mesh, QuadricEdgeCollapseDecimator
from utils import load_off, save_off

if __name__=="__main__":
    
    # Load mesh
    vertices, faces = load_off("./data/stanford_bunny.off")
    mesh = Mesh(vertices, faces)

    # Simplify mesh
    decimator = QuadricEdgeCollapseDecimator(mesh, 1000)
    collapsed_mesh = decimator.simplify()

    # Save new mesh
    save_off("bunny_simplified.off", collapsed_mesh)

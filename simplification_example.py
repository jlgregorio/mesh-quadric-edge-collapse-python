
from quadric_edge_collapse import Mesh, QuadricEdgeCollapseSimplifier
from utils import load_off, save_off

if __name__=="__main__":
    
    # Load mesh
    vertices, faces = load_off("./data/stanford_bunny.off")
    mesh = Mesh(vertices, faces)

    # Simplify mesh
    simplifier = QuadricEdgeCollapseSimplifier(mesh, 2000)
    collapsed_mesh = simplifier.simplify()

    # Save new mesh
    save_off("bunny_simplified.off", collapsed_mesh)

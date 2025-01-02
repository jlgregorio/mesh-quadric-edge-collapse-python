
from quadric_edge_collapse import Mesh, quadric_edge_collapse_decimation
from utils import load_off, save_off

if __name__=="__main__":
    
    # Load mesh
    vertices, faces = load_off("./data/stanford_bunny.off")
    mesh = Mesh(vertices, faces)

    # Simplify mesh
    collapsed_mesh = quadric_edge_collapse_decimation(mesh, 2000)

    # Save new mesh
    save_off("bunny_simplified.off", collapsed_mesh)

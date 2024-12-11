import heapq

import numpy as np

from .mesh import Mesh


def quadric_edge_collapse_decimation(mesh, target_vertex_count):
    """Mesh simplification using a quadric based edge-collapse strategy."""
    
    vertices, faces = mesh.vertices, mesh.faces

    # 1. Compute the Q matrices for all the initial vertices
    Q_matrices = compute_quadrics(mesh)

    # 2. Select all valid pairs
    valid_pairs = mesh.edges

    # 3. Compute the optimal contraction target v_bar and the contraction cost 
    # for each valid pair
    heap_cost = []
    
    for pair in valid_pairs:
         
        # Approximation matrix Q_bar = Q_0 + Q_1
        Q_bar = Q_matrices[pair].sum(axis=0)
        # Optimal position of v is the one that minimizes v.T @ Q @ v
        # which is found by solving Q' @ v = [0, 0, 0, 1]
        Q_prime = np.eye(4)
        Q_prime[:3, :] = Q_bar[:3, :]
        
        try:
            v_bar = np.linalg.inv(Q_prime) @ np.array([0., 0., 0., 1.])
        
        except np.linalg.LinAlgError:

            print("Ill-conditioned matrix!")
            v_bar = mesh.vertices[pair].mean(axis=0)
            v_bar = np.concatenate([v_bar, np.array([1])])

        # Cost of contracting a pair
        contraction_cost = v_bar.T @ Q_bar @ v_bar # negative values?!
        
        # Put all together
        # (v_bar is expressed in homogeneous coordinates [x, y, z, 1], 
        # we just want [x, y, z])
        heap_cost.append((contraction_cost, [*v_bar[:3]], [*pair]))

    # 4. Place all the pairs in a heap keyed on cost 
    # (with the minimum cost pair at the top)
    heapq.heapify(heap_cost)

    # 5. Iteratively remove the pair (v1, v2) of least cost from the heap, 
    # contract this pair, and update the costs of all valid pairs involving v1
    vertices_mask = np.ones(mesh.vertices_number, dtype=bool)
    faces_mask = np.ones(mesh.faces_number, dtype=bool)

    vertices_adjacency = mesh.vertices_adjacency
    vertices_faces = mesh.vertices_faces
    
    while np.sum(vertices_mask) > target_vertex_count:
        
        # Check if there are still edges to collapse
        if not len(heap_cost):
            print("No more edges to collapse!")
            break

        _, v_bar, (id_v_1, id_v_2) = heapq.heappop(heap_cost)


        # Ignore already collapsed edges
        if not vertices_mask[id_v_1] or not vertices_mask[id_v_2]:
            continue

        #  
        shared_vertices = [id_v for id_v in vertices_adjacency[id_v_1]
                           if id_v in vertices_adjacency[id_v_2]]
        
        v_2_exclusive_neighbors = [id_v for id_v in vertices_adjacency[id_v_2]
                                   if id_v != id_v_1 and id_v not in shared_vertices]
        
        shared_faces = [id_v for id_v in vertices_faces[id_v_1]
                        if id_v in vertices_faces[id_v_2]]
        v_2_exclusive_faces = [f for f in vertices_faces[id_v_2]
                               if f not in shared_faces]

        # Ignore non-manifold cases
        if len(shared_vertices) != 2:
            continue
        
        # Ignore boundary edges
        elif len(shared_faces) != 2:
            continue
        
        # Contract the pair
        else:
            
            # Replace v_1 (coordinates) by v_bar
            vertices[id_v_1] = v_bar

            # "Delete" v_2
            vertices_mask[id_v_2] = False

            # "Delete" faces shared by v_1 and v_2
            faces_mask[shared_faces] = False

            # Update v_1 neighbors: first remove v_2...
            vertices_adjacency[id_v_1].remove(id_v_2)
            # ... then add v_2 neighbors not initially shared with v_1
            vertices_adjacency[id_v_1].extend(v_2_exclusive_neighbors)

            # Update v_2 face adjacency
            # vertices_faces[id_v_2] = [] # Useful?
            # Update v_2 neighbors
            # vertices_adjacency[id_v_2] = [] # Useful?
            
            # Update v_2 neighbors connectivity (replace v_2 by v1)
            for id_v in v_2_exclusive_neighbors:
                vertices_adjacency[id_v] = [id_v_1 if id_v==id_v_2 else id_v
                                            for id_v in vertices_adjacency[id_v]]

            
            # Update v_1 face adjacency
            vertices_faces[id_v_1].extend(v_2_exclusive_faces)
            for f in shared_faces:
                 vertices_faces[id_v_1].remove(f)

            
            # Update faces containing v_2 (replace v_2 by v_1)
            for id_f in vertices_faces[id_v_1]:
                faces[id_f] = [id_v_1 if id_v==id_v_2 else id_v
                               for id_v in faces[id_f]]

            # Update the cost of all valid pairs involving v_1
            # TODO: Recompute Q1?
            
            for id_v in vertices_adjacency[id_v_1]:
                
                pair = [id_v_1, id_v]

                # Approximation matrix Q_bar = Q_0 + Q_1
                Q_bar = Q_matrices[pair].sum(axis=0)
                # Q_bar = Q_1 + Q_matrices[id_v]

                # Re-compute costs and target for new pairs
                Q_prime = np.eye(4)
                Q_prime[:3, :] = Q_bar[:3, :]
                
                try:
                    v_bar = np.linalg.inv(Q_prime) @ np.array([0., 0., 0., 1.])
                
                except np.linalg.LinAlgError:

                    print("Ill-conditioned matrix!")
                    v_bar = mesh.vertices[pair].mean(axis=0)
                    v_bar = np.concatenate([v_bar, np.array([1])])

                contraction_cost = v_bar.T @ Q_bar @ v_bar
                
                heapq.heappush(heap_cost, (contraction_cost, [*v_bar[:3]], [*pair]))


    # Rebuild mesh
    new_vertices = vertices[vertices_mask]
    new_vertices_id = np.cumsum(vertices_mask) - 1
    
    new_faces = faces[faces_mask]
    new_faces = new_vertices_id[new_faces.reshape(-1, 1)].reshape(-1, 3)
    
    collapsed_mesh = Mesh(new_vertices, new_faces)

    return collapsed_mesh


def compute_quadrics(mesh):
    """Compute Q matrix for each vertex"""

    # Cartersian coordinates (a, b, c, d) of planes formed by the faces of the mesh
    dists = - np.sum(mesh.faces_normals * mesh.faces_centers, axis=1)
    planes_coords = np.hstack([mesh.faces_normals, dists[:, None]])

    # The error quadric Q for each vertex is the sum of its fundamental quadrics
    Q_matrices = np.zeros((mesh.vertices_number, 4, 4))
    for i, faces in enumerate(mesh.vertices_faces):
        Q_matrices[i] = planes_coords[faces].T @ planes_coords[faces]
    
    return Q_matrices
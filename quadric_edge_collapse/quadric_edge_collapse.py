import heapq

import numpy as np

from .mesh import Mesh


class QuadricEdgeCollapseDecimator:
    """Simplifies a mesh using a quadric-based edge-collapse strategy."""

    def __init__(self, mesh, target_vertex_count):

        self.mesh = mesh
        self.target_vertex_count = target_vertex_count

        self.vertex_quadrics = np.empty((self.mesh.vertex_number, 4, 4))
        self.edge_heap = []
        self.edge_costs = {}
        self.edge_target_position = {}
        self.heap_counter = 0

    def _compute_initial_quadrics(self):
        """Compute the quadric error matrix Q for each vertex of the initial mesh."""

        face_coords = self.mesh.vertices[self.mesh.faces]
        
        # Normal [a, b, c]
        u_1 = face_coords[:, 1] - face_coords[:, 0]
        u_2 = face_coords[:, 2] - face_coords[:, 0]
        face_normals = np.cross(u_1, u_2)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = np.nan_to_num(face_normals)
        
        # Distance from plane to origin d
        d = -np.sum(face_normals * face_coords[:, 0], axis=1)
       
        # Plane coordinates [a, b, c, d]
        planes = np.column_stack((face_normals, d))
        
        # Face quadrics
        face_quadrics = planes[:, :, None] @ planes[:, None, :]
        
        # Vertex quadric is the sum of its faces quadrics
        self.vertex_quadrics.fill(0.)
        for vertex_id, incident_faces in enumerate(self.mesh.vertex_faces):

            if not incident_faces:
                self.vertex_quadrics[vertex_id].fill(0.)
                continue
                
            self.vertex_quadrics[vertex_id] = face_quadrics[list(incident_faces)].sum(axis=0)
        

    def _compute_edge_contraction(self, edge):
            """Compute the optimal contraction position and cost for an edge."""
            
            v_1, v_2 = edge
            
            # Compute quadric
            Q_bar = self.vertex_quadrics[v_1] + self.vertex_quadrics[v_2]
            Q_prime = Q_bar.copy()
            Q_prime[3, :] = [0., 0., 0., 1.]
            
            # Optimal contraction target
            try:
                v_h = np.linalg.solve(Q_prime, [0., 0., 0., 1.])
            
            # If fails select the best option between vertices or midpoint
            except np.linalg.LinAlgError:
                p_1 = self.mesh.vertices[v_1]
                p_2 = self.mesh.vertices[v_2]
                candidates = [
                     np.append(p_1, 1.0), # vertex 1
                     np.append(p_2, 1.0), # vertex 2
                     np.append((p_1 + p_2)/2, 1.0) # midpoint
                ]
                costs = [float(v @ Q_bar @ v) for v in candidates]
                v_h = candidates[np.argmin(costs)]
            
            # The error of contraction target is the cost
            cost = max(0.0, float(v_h @ Q_bar @ v_h))
            
            return v_h[:3], cost

    def _initialize_heap(self):

        # Reset previous values if any
        self.edge_heap.clear()
        self.edge_target_position.clear()
        self.heap_counter = 0
        
        for edge in self.mesh.edge_faces:
            v_bar, cost = self._compute_edge_contraction(edge)

            self.edge_costs[edge] = cost
            self.edge_target_position[edge] = v_bar

            heapq.heappush(self.edge_heap, (cost, self.heap_counter, edge))

            self.heap_counter += 1

    def _is_collapse_valid(self, edge):

        # Reject boundary edge (belonging to one triangular face only)
        if len(self.mesh.edge_faces.get(edge, [])) != 2:
            return False
        # Reject non-manifold geometry
        v_1, v_2 = edge
        common = (
            self.mesh.vertex_neighbors[v_1] & self.mesh.vertex_neighbors[v_2]
        )
        if len(common) != 2:
            return False

        return True

    def _update_region(self, affected_vertices):
        """Recompute collapse costs for all edges incident to affected_vertices."""

        updated_edges = set()

        for v in affected_vertices:

            if not self.mesh.vertex_alive[v]:
                continue

            for neighbor in self.mesh.vertex_neighbors[v]:

                if not self.mesh.vertex_alive[neighbor]:
                    continue

                edge = self.mesh._edge(v, neighbor)

                if edge in updated_edges:
                    continue

                updated_edges.add(edge)

                # Edge may have disappeared
                if edge not in self.mesh.edge_faces:
                    continue

                if not self._is_collapse_valid(edge):
                    continue

                position, cost = self._compute_edge_contraction(edge)

                self.edge_costs[edge] = cost
                self.edge_target_position[edge] = position

                heapq.heappush(self.edge_heap, (cost, self.heap_counter, edge))

                self.heap_counter += 1

    def _collapse(self, edge):
        """Collapse edge and update quadrics."""

        v_keep, v_remove = edge

        # Neighborhood before collapse
        affected_vertices = (
            {v_keep, v_remove}
            | self.mesh.vertex_neighbors[v_keep]
            | self.mesh.vertex_neighbors[v_remove]
        )

        # Optimal position
        self.mesh.vertices[v_keep] = self.edge_target_position[edge]

        # Quadric merge
        self.vertex_quadrics[v_keep] += self.vertex_quadrics[v_remove]

        # Reset quadric for removed vertex
        self.vertex_quadrics[v_remove].fill(0.0)

        # Topological collapse
        self.mesh.collapse_edge(v_keep, v_remove)

        # Add neighbors of kept vertex
        affected_vertices |= self.mesh.vertex_neighbors[v_keep]

        # Remove obsolete cache entries
        self.edge_costs.pop(edge, None)
        self.edge_target_position.pop(edge, None)

        return affected_vertices


    def simplify(self):
        """Simplify mesh."""

        self._compute_initial_quadrics()
        self._initialize_heap()

        while self.mesh.alive_vertex_number > self.target_vertex_count:

            # No more edges to collapse
            if not self.edge_heap:
                break

            cost, _, edge = heapq.heappop(self.edge_heap)

            # Edge disappeared
            if edge not in self.mesh.edge_faces:
                continue

            # Outdated heap entry
            current_cost = self.edge_costs.get(edge)

            if current_cost is None:
                continue

            if abs(cost - current_cost) > 1e-12:
                continue

            # Topology check
            if not self._is_collapse_valid(edge):
                continue

            # Collapse
            affected_vertices = self._collapse(edge)

            # Recompute nearby costs
            self._update_region(affected_vertices)

        # Clean mesh before returning (dead vertices and faces)
        self.mesh.compact()

        return self.mesh

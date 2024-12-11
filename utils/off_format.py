import numpy as np

def load_off(filename):
    """A very basic reader for OFF files"""

    vertices, faces = [], []

    with open(filename, 'r', encoding="utf-8") as file:

        # First line is optional and contains the letters "OFF"
        first_line = file.readline().strip()
        if first_line.startswith("OFF"):
            # Regular case
            if len(first_line)==3:
                second_line = file.readline().strip()
            # Case where the first and second lines are mixed
            else:
                second_line = first_line[3:]
        else:
            # Case where the first line is ommitted
            second_line = first_line
        
        # Second line contains the number of vertices, faces, and edges (optional)
        numbers = list(map(int, second_line.split()))
        n_v = numbers[0]
        n_f = numbers[1]
        
        # Following n_v lines contain the list of vertices
        # Each line contains the x, y, z coordinates of a vertex
        for _ in range(n_v):
            line = file.readline()
            values = list(map(float, line.strip().split()))
            vertices.append(values[:3])

        # Following n_f lines contain the list of faces
        # Each line contains the number of vertices of the face followed by the
        # indexes of the vertices (zero indexing) and RGB values (optional)
        for _ in range(n_f):
            line = file.readline()
            values = list(map(int, line.strip().split()))
            n = values.pop(0)
            if n==3: #triangle
                faces.append(values[:3])
            elif n==4: # split quad into 2 triangles
                faces.append(values[:3])
                faces.append(values[1:4])

    return np.array(vertices), np.array(faces)


def save_off(filename, mesh):
    """A very basic writer for OFF files"""

    with open(filename, 'w', encoding="utf-8") as file:
        
        # Header
        file.write(f"OFF\n{mesh.vertices_number} {mesh.faces_number} 0 \n")

        # Vertices
        for vertex in mesh.vertices:
            file.write(" ".join(map(str, vertex)) + "\n")

        # Faces
        for face in mesh.faces:
            file.write("3 " + " ".join(map(str, face)) + "\n")


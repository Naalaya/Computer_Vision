# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.array([ 0.16257299, -0.370805  , -1.09232295,  1.62570095,
#               -1.62570095,  1.09232295,  0.370805  , -0.16257299])
# y = np.array([-1.71022499, -0.81153202, -0.52910602, -0.36958599,
#                0.369587  ,  0.52910602,  0.81153202,  1.71022499])
# z = np.array([ 0.22068501, -1.48456001,  1.23566902,  0.469576  ,
#               -0.469576  , -1.23566902,  1.48456001, -0.22068501])
#
# faces = ([[3, 0, 1],[6, 7, 4],[3, 6, 2],[0, 2, 4],[1, 4, 7],[6, 3, 5],
#           [1, 5, 3],[4, 2, 6],[2, 0, 3],[4, 1, 0],[7, 5, 1],[5, 7, 6]])
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# ax.plot_trisurf(x,y,z, triangles = faces)
# plt.show()

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Homogeneous point we want to convert

point_3d = np.array([2, 2, -10, 1])

# Type of the projection we want

# projection_type = “orthographic”

# Coordinates of the view volume

left = -3

right = 3

bottom = -3

top = 3

near = 5

far = 20

# Creating camera matrix

rotation_matrix = np.array([

    [1, 0, 0, 0],

    [0, 1, 0, 0],

    [0, 0, 1, 0],

    [0, 0, 0, 1]

])

translation_matrix = np.array([

    [1, 0, 0, 0],

    [0, 1, 0, 0],

    [0, 0, 1, 0],

    [0, 0, 0, 1]

])

camera_matrix = rotation_matrix @ translation_matrix


# Projection Matrix

def orthographic_projection(left, right, bottom, top, near, far):
    op_matrix = np.array(
        [[2 / (right - left), 0, 0, -(right + left) / (right - left)],
         [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
         [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
    return op_matrix


def perspective_projection(left, right, bottom, top, near, far):
    pp_matrix = np.array([

        [(2 * near) / (right - left), 0, (right + left) / (right - left), 0],

        [0, (2 * near) / (top - bottom), (top + bottom) / (top - bottom), 0],

        [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],

        [0, 0, -1, 0]

    ])

    return pp_matrix


# ViewPort Matrix

nx = 600

ny = 600

viewport_matrix = np.array([

    [nx / 2, 0, 0, (nx - 1) / 2],

    [0, ny / 2, 0, (ny - 1) / 2],

    [0, 0, 0.5, 0.5],

])

# Choosing projection matrix associated with projection type
projection_matrix = perspective_projection(left, right, bottom, top, near, far)

# Applying the matrices in the described order.

point_after_CM = camera_matrix @ point_3d

point_after_PM = projection_matrix @ point_after_CM

# Normalization of the projected point

point_after_PM /= point_after_PM[3]

point_after_VP = viewport_matrix @ point_after_PM

cube_vertices = np.array([

    [-1, -1, -1, 1],  # Vertex 0

    [1, -1, -1, 1],  # Vertex 1

    [1, 1, -1, 1],  # Vertex 2

    [-1, 1, -1, 1],  # Vertex 3

    [-1, -1, 1, 1],  # Vertex 4

    [1, -1, 1, 1],  # Vertex 5

    [1, 1, 1, 1],  # Vertex 6

    [-1, 1, 1, 1]  # Vertex 7

])

# Translate cube vertices to center at (0, 0, -10)

translation_vector = np.array([0, 0, -10, 0])

cube_vertices = cube_vertices + translation_vector

cube_edges = [

    [0, 1], [1, 2], [2, 3], [3, 0],

    [4, 5], [5, 6], [6, 7], [7, 4],

    [0, 4], [1, 5], [2, 6], [3, 7]

]

# Create a figure and 3D subplot

fig = plt.figure(figsize=(10, 6))

ax3d = fig.add_subplot(121, projection='3d')

ax2d = fig.add_subplot(122)


# Plot the cube in 3D

for edge in cube_edges:
    ax3d.plot(cube_vertices[edge, 0], cube_vertices[edge, 1], cube_vertices[edge, 2], color='blue')
    # Transformed cube vertices after camera and projection matrices
    cube_after_CM = camera_matrix @ cube_vertices.T
    cube_after_PM = projection_matrix @ cube_after_CM
    cube_after_PM /= cube_after_PM[3]
    cube_after_VP = viewport_matrix @ cube_after_PM

    # Plot the projected cube in 2D

    for edge in cube_edges:
        start_idx, end_idx = edge

        start_point = cube_after_VP[:2, start_idx]

        end_point = cube_after_VP[:2, end_idx]

        ax2d.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

        # Set labels and title

        ax3d.set_xlabel('X')

        ax3d.set_ylabel('Y')

        ax3d.set_zlabel('Z')

        ax3d.set_title('3D Cube Projection')

        ax2d.set_xlabel('X')

        ax2d.set_ylabel('Y')

        ax2d.set_xlim(0, nx)

        ax2d.set_ylim(0, ny)

        ax2d.set_title('2D Projection on Screen')
        plt.tight_layout()
plt.show()



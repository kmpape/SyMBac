from typing import List, Tuple, Union

import numpy as np


def circ(
        angle: Union[np.array, List[float]], 
        radius: float,
        x_shift: float = 0.0, 
        y_shift: float = 0.0, 
        remove_start_end: bool=False,
    ) -> Tuple[np.array, np.array]:
    y = radius * np.cos(angle) + y_shift
    x = radius * np.sin(angle) + x_shift
    return (x[1:-1], y[1:-1]) if (remove_start_end and len(angle) > 2) else (x, y)

def centroid(vertices: np.array) -> np.array:
    return np.mean(vertices,axis=0)

def wall(
        x_start: float, 
        x_end: float, 
        y_shift: float,
        resolution: int,
        remove_start_end: bool=False,
    ) -> Tuple[np.array, np.array]:
    x = np.linspace(x_start, x_end, resolution)
    y = np.ones(resolution)*y_shift
    return (x[1:-1], y[1:-1]) if (remove_start_end and resolution > 2) else (x, y)

def get_vertices(
        cell_length: float, 
        cell_width: float, 
        angle: float, 
        resolution: int,
    ) -> List[Tuple[float, float]]:
    """Generates coordinates for a cell centered around (0,0)

    Parameters
    ----------
    cell_length : float
        The length of the cell. 
    cell_width : float
        Total thickness of the cell, defines the poles too.
    angle : float
        Angle in radians to rotate the cell by (counter-clockwise)
    resolution : int
        Number of points defining the cell wall geometry
    Returns
    -------
    list of lists containing cell x and y coords
    """
    assert cell_length >= cell_width
    straight_length = cell_length - cell_width
    left_x, left_y = circ(angle=np.linspace(np.pi,2*np.pi, resolution), radius=cell_width/2, x_shift=-straight_length/2, remove_start_end=True)
    right_x, right_y = circ(angle=np.linspace(0,np.pi, resolution), radius=cell_width/2, x_shift=+straight_length/2, remove_start_end=True)
    top_x, top_y = wall(x_start=-straight_length/2, x_end=straight_length/2, y_shift=cell_width/2, resolution=resolution)
    bottom_x, bottom_y = wall(x_start=-straight_length/2, x_end=straight_length/2, y_shift=-cell_width/2, resolution=resolution)
    coordinates = np.column_stack((
        np.concatenate((np.flip(left_x), bottom_x, np.flip(right_x), np.flip(top_x))), 
        np.concatenate((np.flip(left_y), bottom_y, np.flip(right_y), np.flip(top_y)))
    ))
    rotation_matrix_T =  np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    # Shape is symmetrix in X and Y so centroid will be (0, 0)
    return np.dot(coordinates, rotation_matrix_T).tolist()  # TODO Is this tolist needed?

# def circ(
#         angle: Union[np.array, List[float]], 
#         radius: float,
#         x_shift: float = 0.0, 
#         y_shift: float = 0.0, 
#     ) -> Tuple[np.array, np.array]:
#     y = radius * np.cos(angle) + radius
#     x = radius * np.sin(angle) + x_shift + radius
#     return x, y

# def wall(
#         thickness: float, 
#         start: float, 
#         end: float, 
#         t_or_b: bool, 
#         resolution: int,
#     ) -> Tuple[np.array, np.array]:
#     """Generates the straight part of the cell wall's coordinates (all but the poles)

#     Parameters
#     ----------
#     thickness : float
#         The distance from the top cell wall to the bottom cell wall
#     start : float
#         The start X coordinate of the cell wall
#     end : float
#         The end X coordinate of the cell wall
#     t_or_b: int
#         0 for top wall
#         1 for bottom wall
#     resolution : int
#         Number of points defining the cell wall geometry
#     Returns
#     -------
#     tuple(Numpy Array, Numpy Array)
#         return[0] is the wall's x coordinates
#         return[0] is the wall's y coordiantes
        
#     Examples
#     --------
#     Create two cell walls of length 10, 3 apart

#     >>> import numpy as np
#     >>> import matplotlib.pyplot as plt
#     >>> top_wall = wall(3,0,10,0,20)
#     >>> bottom_wall = top_wall = wall(3,0,10,1,20)
#     >>> plt.plot(walls[0], walls[1])
#     >>> plt.plot(walls[0], walls[1])
#     >>> plt.show()
#     """
#     wall_x = np.linspace(start, end, resolution)
#     wall_y = np.ones(resolution)*thickness * t_or_b +thickness
#     return wall_x, wall_y

# def get_vertices(
#         cell_length: float, 
#         cell_width: float, 
#         angle: float, 
#         resolution: int,
#     ) -> List[Tuple[float, float]]:
#     """Generates coordinates for a cell centered around (0,0)

#     Parameters
#     ----------
#     cell_length : float
#         The length of the cell. 
#     cell_width : float
#         Total thickness of the cell, defines the poles too.
#     angle : float
#         Angle in radians to rotate the cell by (counter-clockwise)
#     resolution : int
#         Number of points defining the cell wall geometry
#     Returns
#     -------
#     list of lists containing cell x and y coords
    
#     Examples
#     --------
#     Create a cell of length 10+4 rotated by 1 radian with a resolution of 20:
    
#     >>> import numpy as np
#     >>> import matplotlib.pyplot as plt
#     >>> verts = get_vertices(10,4,1,20)
#     >>> verts_y = [y[0] for y in verts]
#     >>> verts_x = [x[1] for x in verts]
#     >>> plt.plot(verts_x,verts_y)
#     """
#     assert cell_length >= cell_width
#     w_half = cell_width/2
#     left_wall = circ(angle=np.linspace(np.pi,2*np.pi, resolution), x_shift=0, radius=w_half)
#     right_wall = circ(angle=np.linspace(0,np.pi, resolution), x_shift=cell_length - w_half*2, radius=w_half)

#     top_wall_xy = wall(w_half, w_half, cell_length - w_half, 1, resolution)
#     bottom_wall_xy = wall(w_half, w_half, cell_length - w_half, -1, resolution)

#     coordinates = [[left_wall[0][x] - cell_length/2, left_wall[1][x] - w_half/2] for x in reversed(range(len(left_wall[0])))] + \
#             [[bottom_wall_xy[0][x] - cell_length/2, bottom_wall_xy[1][x]- w_half/2] for x in (range(len(bottom_wall_xy[0])))] + \
#             [[right_wall[0][x] - cell_length/2, right_wall[1][x]- w_half/2] for x in reversed(range(len(right_wall[0])))] + \
#             [[top_wall_xy[0][x] - cell_length/2, top_wall_xy[1][x]- w_half/2] for x in reversed(range(len(top_wall_xy[0])))]
#     coordinates = np.array(coordinates)
#     cell_centroid = centroid(coordinates)
#     centered_verts = coordinates - cell_centroid
#     centered_verts = centered_verts.tolist()

#     rotated = np.zeros((len(centered_verts),2))
#     for x in range(len(centered_verts)):
#         rotated[x] = rotate(cell_centroid, (centered_verts[x][0],centered_verts[x][1]), angle)
#     centered_verts = rotated - centroid(rotated)

#     return centered_verts.tolist()

# def centroid(vertices):
#     """Return the centroid of a list of vertices 
    
#     Parameters
#     ----------
#     vertices : list(tuple)
#         A list of tuples containing x,y coordinates.

#     """
#     return np.sum(vertices,axis=0)/len(vertices)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pymunk
import numpy as np

from SyMBac.cell_geometry import CellGeometry

def segment_creator(
        local_xy1, 
        local_xy2, 
        global_xy, 
        thickness: int, 
        friction: float=0.0,#
    ) -> Tuple[pymunk.Body, pymunk.Segment]:
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1,local_xy2,thickness)
    segment_body.position = global_xy
    segment_shape.friction = 0
    return segment_body, segment_shape

def trench_creator(size,trench_length, global_xy, space):
    size = int(np.ceil(size/1.5))  # TODO What is this? This will bug width any max-width parameter in trenches
    segments = []
    for x in range(size):
        segment = segment_creator((x,0),(0,size-x),global_xy,1)
        segments.append(segment)

    for x in range(size):
        segment = segment_creator((size-x,0),(size,size-x),(global_xy[0]+size/2, global_xy[1]),1)
        segments.append(segment)
    for z in segments:
        for s in z:
            space.add(s)

    left_wall = segment_creator((0,0),(0,trench_length),global_xy,1)
    right_wall = segment_creator((size,0),(size,trench_length),(global_xy[0]+size/2, global_xy[1]),1)
    barrier_thickness = 1
    left_barrier = segment_creator((0,0),(0,trench_length),(global_xy[0]-barrier_thickness, global_xy[1]),barrier_thickness)
    right_barrier = segment_creator((size,0),(size,trench_length),(global_xy[0]+size/2+barrier_thickness, global_xy[1]),barrier_thickness)
    walls = [left_wall, right_wall, left_barrier, right_barrier]
    for z in walls:
        for s in z:
            space.add(s)

def trench_creator2(
        width: float, 
        length: float, 
        position: Tuple[float, float], 
        space: pymunk.Space, 
        friction: float=0, 
        thickness=1,
    ) -> List[int]:
    assert length >= width/2
    body_ids = []
    coords = np.concatenate((
        np.array([[0, length]]), # Left wall
        CellGeometry.make_circ(
            angle_start=-np.pi, 
            angle_end=0, 
            radius=width/2, 
            x_shift=width/2, 
            y_shift=width/2,
            resolution=13), # Bottom wall
        np.array([[width, length]]), # Right wall
    ))
    for irow in range(coords.shape[0]-1):
        body, shape = segment_creator(
            local_xy1=(coords[irow,0], coords[irow,1]),
            local_xy2=(coords[irow+1,0], coords[irow+1,1]),
            global_xy=position,
            thickness=thickness,
            friction=friction,
        )
        space.add(body, shape)
        body_ids.append(body.id)
    return body_ids

def get_trench_segments(space):
    """
    A function which extracts the rigid body trench objects from the pymunk space object. Space object should be passed
    from the return value of the run_simulation() function

    Returns
    -------
    List of trench segment properties, later used to draw the trench.
    """
    trench_shapes = []
    for shape, body in zip(space.shapes, space.bodies):
        if body.body_type == 2:
            trench_shapes.append(shape)

    trench_segment_props = []
    for x in trench_shapes:
        trench_segment_props.append([x.bb, x.area, x.a, x.b])

    trench_segment_props = pd.DataFrame(trench_segment_props)
    trench_segment_props.columns = ["bb", "area", "a", "b"]
    main_segments = trench_segment_props.sort_values("area", ascending=False).iloc[0:2]
    return main_segments
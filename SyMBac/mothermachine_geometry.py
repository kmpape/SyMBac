from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pymunk
import numpy as np

from SyMBac.cell_geometry import CellGeometry
from SyMBac.cell import Cell2


class MothermachinePart(Enum):
    MOTHERMACHINE = auto()
    TRENCH = auto()
    CHANNEL = auto()
    UNKNOWN = auto()


class ChannelFlowDirection(Enum):
    LEFT = -1
    RIGHT = 1


class Mothermachine:
    def __init__(
            self,
            space: pymunk.Space,
            trench_width: float,
            trench_length: float,
            trench_spacing: float,
            num_trenches: int,
            channel_width: float,
            segment_thickness: int=1,
            left_margin: Optional[float]=None,
            right_margin: Optional[float]=None,
            flow_direction: Optional[ChannelFlowDirection] = ChannelFlowDirection.RIGHT,
    ):
        self.space: pymunk.Space = space
        self.t_width: float = trench_width
        self.t_length: float = trench_length
        self.t_spacing: float = trench_spacing
        self.num_t: int = num_trenches
        self.c_width: float = channel_width
        self.segment_thickness: float = segment_thickness
        self.left_margin: float = trench_spacing if left_margin is None else left_margin
        self.right_margin: float = trench_spacing if right_margin is None else right_margin
        self.flow_direction: ChannelFlowDirection = flow_direction

        self._trench_ids: List[int] = list(range(2*self.num_t)) # First half = bottom trenches
        self._body_ids: Dict[int, Optional[List[int]]] = {_id: None for _id in self._trench_ids}
        self._trench_positions: Dict[int, Optional[Tuple[float, float]]] = {_id: None for _id in self._trench_ids}

        self._build()

    def get_bounding_box(self, which: MothermachinePart, trench_id: Optional[int] = None) -> pymunk.BB:
        if which == MothermachinePart.MOTHERMACHINE:
            left = 0
            right = self.left_margin + self.right_margin + self.num_t * self.t_width + (self.num_t - 1) * self.t_spacing
            top = 2 * self.t_length + self.c_width
            bottom = 0
        elif which == MothermachinePart.CHANNEL:
            left = 0
            right = self.left_margin + self.right_margin + self.num_t * self.t_width + (self.num_t - 1) * self.t_spacing
            top = self.t_length + self.c_width
            bottom = self.t_length
        else:
            if not self._is_valid_id(trench_id=trench_id):
                raise ValueError(f"Invalid trench_id={trench_id} provided. Must be in [0,{2*self.num_t-1}]")
            bottom, left = self._trench_positions[trench_id]
            right = left + self.t_width
            top = bottom + self.t_length           

        return pymunk.BB(left=left, right=right, top=top, bottom=bottom)
    
    def get_flow_force(self) -> Tuple[float, float]:
        mag = 1
        mean_angle = np.pi * 0.5 if self.flow_direction == ChannelFlowDirection.LEFT else 0.0
        angle = np.random.uniform(low=mean_angle-np.pi/6, high=mean_angle+np.pi/6)
        return (mag * np.cos(angle), mag * np.sin(angle))
    
    def get_trench_ids(self):
        return self._trench_ids
    
    def get_trench_position(self, trench_id: int) -> Tuple[float, float]:
        if not self._is_valid_id(trench_id=trench_id):
            raise ValueError(f"Invalid trench_id={trench_id} provided. Must be in [0,{2*self.num_t-1}]")
        return self._trench_positions[trench_id]
    
    def is_top_trench(self, trench_id: int) -> bool:
        return trench_id >= self.num_t
    
    def where(self, cell: Cell2, use_centroid: bool=True) -> MothermachinePart:
        bb = self.get_bounding_box(which=MothermachinePart.MOTHERMACHINE)
        if cell.is_out_of_bounds(x_min=bb.left, x_max=bb.right, y_min=bb.bottom, y_max=bb.top, use_centroid=use_centroid):
            return MothermachinePart.UNKNOWN
        bb = self.get_bounding_box(which=MothermachinePart.CHANNEL)
        if not cell.is_out_of_bounds(x_min=bb.left, x_max=bb.right, y_min=bb.bottom, y_max=bb.top, use_centroid=use_centroid):
            return MothermachinePart.CHANNEL
        return MothermachinePart.TRENCH # Assuming that cells are spawned and staying in trenches or channels
    
    def _build(self):
        _ = Mothermachine.create_and_add_segment(
            space=self.space,
            local_xy1=(0, 0),
            local_xy2=(self.left_margin, 0),
            global_xy=(0, self.t_length),
            thickness=self.segment_thickness,
        )
        _ = Mothermachine.create_and_add_segment(
            space=self.space,
            local_xy1=(0, 0),
            local_xy2=(self.left_margin, 0),
            global_xy=(0, self.t_length + self.c_width),
            thickness=self.segment_thickness,
        )
        for i in range(self.num_t):
            x_pos_i = i * (self.t_width + self.t_spacing) + self.left_margin
            # Bottom trench and wall
            bottom_pos_i = (x_pos_i, 0)
            self._body_ids[i] = Mothermachine.create_and_add_bottom_trench(
                space=self.space,
                width=self.t_width,
                length=self.t_length,
                position=bottom_pos_i,
                thickness=self.segment_thickness,
            )
            self._trench_positions[i] = bottom_pos_i
            _ = Mothermachine.create_and_add_segment(
                space=self.space,
                local_xy1=(self.t_width, self.t_length),
                local_xy2=(self.t_width+self.t_spacing if i < self.num_t-1 else self.t_width+self.right_margin, self.t_length),
                global_xy=bottom_pos_i,
                thickness=self.segment_thickness,
            )
            # Top trench and wall
            top_pos_i = (x_pos_i, self.t_length + self.c_width)
            self._body_ids[i+self.num_t] = Mothermachine.create_and_add_top_trench(
                space=self.space,
                width=self.t_width,
                length=self.t_length,
                position=top_pos_i,
                thickness=self.segment_thickness,
            )
            self._trench_positions[i+self.num_t] = top_pos_i
            _ = Mothermachine.create_and_add_segment(
                space=self.space,
                local_xy1=(self.t_width, 0),
                local_xy2=(self.t_width+self.t_spacing if i < self.num_t-1 else self.t_width+self.right_margin, 0),
                global_xy=top_pos_i,
                thickness=self.segment_thickness,
            )
    
    def _is_valid_id(self, trench_id: int) -> bool:
        return trench_id in self._trench_ids

    @staticmethod
    def create_and_add_top_trench(
        width: float, 
        length: float, 
        position: Tuple[float, float], # Corresponds to bottom left corner
        space: pymunk.Space, 
        friction: float=0, 
        thickness: float=1,
        half_circle_resolution: int=13,
    ) -> List[int]:
        assert length >= width/2
        body_ids = []
        coords = np.concatenate((
            np.array([[0, 0]]), # Left wall
            CellGeometry.make_circ(
                angle_start=np.pi, 
                angle_end=0, 
                radius=width/2, 
                x_shift=width/2, 
                y_shift=length-width/2,
                resolution=half_circle_resolution), # Bottom wall
            np.array([[width, 0]]), # Right wall
        ))
        for irow in range(coords.shape[0]-1):
            body_ids.append(Mothermachine.create_and_add_segment(
                space=space,
                local_xy1=(coords[irow,0], coords[irow,1]),
                local_xy2=(coords[irow+1,0], coords[irow+1,1]),
                global_xy=position,
                thickness=thickness,
                friction=friction,
            ))
        return body_ids

    @staticmethod
    def create_and_add_bottom_trench(
        width: float, 
        length: float, 
        position: Tuple[float, float], # Corresponds to bottom left corner
        space: pymunk.Space, 
        friction: float=0, 
        thickness: float=1,
        half_circle_resolution: int=13,
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
                resolution=half_circle_resolution), # Bottom wall
            np.array([[width, length]]), # Right wall
        ))
        for irow in range(coords.shape[0]-1):
            body_ids.append(Mothermachine.create_and_add_segment(
                space=space,
                local_xy1=(coords[irow,0], coords[irow,1]),
                local_xy2=(coords[irow+1,0], coords[irow+1,1]),
                global_xy=position,
                thickness=thickness,
                friction=friction,
            ))
        return body_ids

    @staticmethod
    def create_segment(
        local_xy1: Tuple[float, float], 
        local_xy2: Tuple[float, float], 
        global_xy: Tuple[float, float], 
        thickness: float=1, 
        friction: float=0.0,
    ) -> Tuple[pymunk.Body, pymunk.Segment]:
        segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        segment_shape = pymunk.Segment(segment_body, local_xy1,local_xy2,thickness)
        segment_body.position = global_xy
        segment_shape.friction = friction
        return segment_body, segment_shape

    @staticmethod
    def create_and_add_segment(
        space: pymunk.Space,
        local_xy1: Tuple[float, float], 
        local_xy2: Tuple[float, float], 
        global_xy: Tuple[float, float], 
        thickness: float=1, 
        friction: float=0.0,
    ) -> int:
        body, shape = Mothermachine.create_segment(
            local_xy1=local_xy1,
            local_xy2=local_xy2,
            global_xy=global_xy,
            thickness=thickness,
            friction=friction,
        )
        space.add(body, shape)
        return body.id

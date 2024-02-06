from dataclasses import dataclass, field
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pymunk
from SyMBac import cell_geometry

# For reproducible results TODO move this to package __init__
SEED = 1
random.seed(SEED)

# TODO Move these constants to config
DEFAULT_CELL_MASS = 1.0
DEFAULT_CELL_FRICTION = 0.0

def create_growth_rate_callable(
        growth_rate_constant: float=0.01,
        min_uniform: float=0.5,
        max_uniform: float=1.3, 
    ) -> Callable[[], float]:
    def get_random_growth_rate() -> float:
        return growth_rate_constant * random.uniform(min_uniform, max_uniform)
    return get_random_growth_rate

def create_max_length_callable(
        min_uniform: float=0.9,
        max_uniform: float=1.1,  
    ) -> Callable[[float], float]:
    def get_random_max_length(max_length: float) -> float:
        return max_length * random.uniform(min_uniform, max_uniform)
    return get_random_max_length

def create_width_callable(
        max_width: float,
        min_uniform: float=0.9,
        max_uniform: float=1.1,  
    ) -> Callable[[float], float]:
    def get_random_width(width: float) -> float:
        return min(width * random.uniform(min_uniform, max_uniform), max_width)
    return get_random_width

def create_lysis_callable(
        lysis_probability: float = 0,  
    ) -> Callable[[], bool]:
    def get_lysis_event() -> bool:
        return random.uniform(0, 1) < lysis_probability
    return get_lysis_event


class CellIDFactory:
    def __init__(self, start_id: int=0):
        self._counter: int = start_id

    def get_next_id(self) -> int:
        next_id = self._counter
        self._counter += 1
        return next_id


@dataclass
class CellCurvatureProperties:
    freq_modif: float = field(default_factory=lambda: random.uniform(0.9, 1.1))
    amp_modif: float = field(default_factory=lambda: random.uniform(0.9, 1.1))
    phase_modif: float = field(default_factory=lambda: random.uniform(-1, 1))
    phase_mult: float = 20

    def tolist(self) -> List[float]:
        return [self.freq_modif, self.amp_modif, self.phase_modif, self.phase_mult]


@dataclass
class CellProperties:
    length: float
    width: float
    angle: float
    centroid: Tuple[float, float]
    pinching_sep: float

    def tolist(self, curve_props: CellCurvatureProperties):
        """This function returns properties in original SyMBac order."""
        return [self.length, self.width, self.angle, self.centroid,
                curve_props.freq_modif, curve_props.amp_modif, curve_props.phase_modif, curve_props.phase_mult,
                self.pinching_sep]


class Cell:
    def __init__(
        self,
        length: float,
        width: float,
        position: Tuple[float, float],
        angle: float,
        max_length: float,
        id_factory: CellIDFactory,
        self_id: int,
        get_width_daughter: Callable[[float], float],
        parent_id:  Optional[int] = None,
        daughter_ids:  List[int] = [],
        pinching_sep: float = 0,
        get_lysis_event: Callable[[], bool] = create_lysis_callable(),
        get_max_length_daughter: Callable[[float], float] = create_max_length_callable(),
        get_random_growth_rate: Callable[[], float] = create_growth_rate_callable(),
        mother_above_daughter: bool = False,
        division_ratio: float = 0.5,
    ):
        self.max_length: float = max_length
        self.geometry = cell_geometry.CellGeometry(length=length, width=width)
        pymunk_body_shape = Cell.make_pymunk_cell(
                vertices=self.geometry.get_vertices().tolist(),
                angle=angle,
                position=position,
        )
        self.body: pymunk.Body = pymunk_body_shape[0]
        self.shape: pymunk.Poly = pymunk_body_shape[1]
        self.id_factory: CellIDFactory = id_factory
        self.self_id: int = self_id
        self.parent_id: int = parent_id
        self.daughter_ids: List[int] = daughter_ids
        self.pinching_sep: float = pinching_sep
        self.get_lysis_event: Callable[[], bool] = get_lysis_event
        self.get_width_daughter: Callable[[float], float] = get_width_daughter
        self.get_max_length_daughter: Callable[[], float] = get_max_length_daughter
        self.get_random_growth_rate: Callable[[], float] = get_random_growth_rate
        self.division_sign: int = 1 if mother_above_daughter else -1
        self.division_ratio: float = division_ratio

        self._is_alive: bool = True
        self._curvature_properties: CellCurvatureProperties = CellCurvatureProperties()
        self._timeseries_properties: Dict[int, CellProperties] = {}

    def apply_force(self, force_global: Tuple[float, float]):
        force_local = pymunk.Vec2d(x=force_global[0], y=force_global[1]).rotated(-self.shape.body.angle)
        point_local = pymunk.Vec2d(x=np.random.uniform(low=-self.get_length()/2, high=self.get_length()/2), y=0)
        self.body.apply_force_at_local_point(force=force_local, point=point_local)

    def divide(self) -> 'Cell':
        width_daughter = self.get_width_daughter(self.get_width())
        max_length_daughter = self.get_max_length_daughter(self.max_length)

        length_old = self.get_length()
        length_mother = (length_old-self.get_width()) * (1-self.division_ratio) + self.get_width()
        length_daughter = max((length_old-self.get_width()) * self.division_ratio + self.get_width(), width_daughter) # Otherwise, CellGeometry assertion will fail

        position_old = self.body.position
        position_mother = (position_old[0] + self.division_sign * length_mother/2 *  np.cos(self.get_angle()),
                           position_old[1] + self.division_sign * length_mother/2 *  np.sin(self.get_angle()))
        position_daugher = (position_old[0] - self.division_sign * length_daughter/2 *  np.cos(self.get_angle()),
                            position_old[1] - self.division_sign * length_daughter/2 *  np.sin(self.get_angle()))

        self.grow(new_length=length_mother, new_pinching_sep=0.0)
        self.translate(new_position=position_mother)

        daughter_id = self.id_factory.get_next_id()
        self.daughter_ids.append(daughter_id)

        return Cell(
            length=length_daughter,
            width=width_daughter,
            position=position_daugher,
            angle=self.get_angle(),
            max_length=max_length_daughter,
            id_factory=self.id_factory,
            self_id=daughter_id,
            parent_id=self.self_id,
            daughter_ids=[],
            pinching_sep=0,
            get_lysis_event=self.get_lysis_event,
            get_width_daughter=self.get_width_daughter,
            get_max_length_daughter=self.get_max_length_daughter,
            get_random_growth_rate=self.get_random_growth_rate,
            mother_above_daughter=True if self.division_sign > 1 else False,
            division_ratio=self.division_ratio,
        )
    
    def get_angle(self) -> float:
        return self.body.angle
    
    def get_centroid(self) -> Tuple[float, float]:
        return tuple(self.get_position() + self.geometry.get_centroid(angle=self.get_angle()))  # TODO this sums np.array and Tuple[float, float]
    
    def get_curvature_properties(self, as_list: bool=False) -> Union[CellCurvatureProperties, List[float]]:
        return self._curvature_properties.tolist() if as_list else self._curvature_properties

    def get_length(self) -> float:
        return self.geometry.get_length()
    
    def get_pinching_sep(self) -> float:
        return self.pinching_sep
    
    def get_position(self) -> Tuple[float, float]:
        return self.body.position
    
    def get_properties_list(self, time_index: int) -> List[Union[float, Tuple[float, float]]]:
        assert time_index in self._timeseries_properties
        return self._timeseries_properties[time_index].tolist(curve_props=self._curvature_properties)
    
    def get_vertex_list(self, global_coordinates: bool=True) -> List[Tuple[float, float]]:
        if global_coordinates:
            return [self._to_global_coordinates(vertex) for vertex in self.shape.get_vertices()]
        else:
            return [(vertex[0], vertex[1]) for vertex in self.shape.get_vertices()]
    
    def get_width(self) -> float:
        return self.geometry.get_width()
    
    def grow(self, new_length: Optional[float]=None, new_pinching_sep: Optional[float]=None):
        if new_length is not None:
            self._is_alive = not self.get_lysis_event()
            self._update_vertices(new_length=self.get_length())
        if self.is_alive():
            new_length = self.get_length() * (1 + self.get_random_growth_rate()) if new_length is None else new_length
            self._update_vertices(new_length=new_length)
            self._update_pinching_sep(pinching_sep=new_pinching_sep)
        else:
            self.shape.color = (0, 255, 0, 255)

    def has_properties(self, time_index: int) -> bool:
        return time_index in self._timeseries_properties

    def is_alive(self) -> bool:
        return self._is_alive

    def is_dividing(self) -> bool: 
        return self.is_alive() and self.get_length() > self.max_length
    
    def is_out_of_bounds(self, bb: Union[Tuple[float,float,float], pymunk.BB], use_centroid: bool=False) -> bool:
        if isinstance(bb, pymunk.BB):
            x_min, x_max, y_min, y_max = (bb.left, bb.right, bb.bottom, bb.top)
        else:
            x_min, x_max, y_min, y_max = bb
        if use_centroid:
            c = self.get_centroid()
            return c[0] < x_min or c[0] > x_max or c[1] < y_min or c[1] > y_max
        else:
            return any([v[0] < x_min or v[0] > x_max or v[1] < y_min or v[1] > y_max for v in self.get_vertex_list()])
    
    def record_timeseries_properties(self, timestep: int):
        assert timestep not in self._timeseries_properties
        self._timeseries_properties[timestep] = CellProperties(
            length=self.get_length(),
            width=self.get_width(),
            angle=self.get_angle(),
            centroid=self.get_centroid(),
            pinching_sep=self.pinching_sep,
        )

    def translate(self, new_position: Tuple[float, float]):
        self.body.position = new_position

    def _update_vertices(self, new_length: float):
        self.geometry.update_length(new_length=new_length)  # Update local vertex and centroid coordinates
        self.shape.unsafe_set_vertices(self.geometry.get_vertices().tolist())

    def _update_pinching_sep(self, pinching_sep: Optional[float]=None):
        self.pinching_sep = min(max(0, self.get_length() - self.max_length + self.get_width()), self.get_width() - 2) if pinching_sep is None else pinching_sep

    @staticmethod
    def _to_global_coordinates(vertex: pymunk.Vec2d, shape: pymunk.Poly):
        return vertex.rotated(shape.body.angle) + shape.body.position

    @staticmethod
    def _to_local_coordinates(vertex: pymunk.Vec2d, shape: pymunk.Poly):
        return vertex.rotated(-shape.body.angle) - shape.body.position
    
    @staticmethod
    def make_pymunk_cell(
            vertices: List[Tuple[float, float]],
            angle: Optional[float]=None,
            position: Optional[Tuple[float, float]]=None,
        ) -> Tuple[pymunk.Body, pymunk.Poly]:

        shape = pymunk.Poly(None, vertices)
        shape.friction = DEFAULT_CELL_FRICTION
        moment = pymunk.moment_for_poly(DEFAULT_CELL_MASS, shape.get_vertices())  # TODO what is the difference between shape.get_vertices and our vertices?
        body = pymunk.Body(DEFAULT_CELL_MASS, moment)
        body.position = body.position if position is None else position
        body.angle = body.angle if angle is None else angle
        shape.body = body
        
        return body, shape

    

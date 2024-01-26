from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pymunk
from SyMBac import cell_geometry


def create_growth_rate_callable(
        dt: float=0.01,
        growth_rate_constant: float=1,
        min_uniform: float=0.5,
        max_uniform: float=1.3, 
    ) -> Callable[[], float]:
    def get_random_growth_rate() -> float:
        return growth_rate_constant * dt * np.random.uniform(min_uniform, max_uniform)
    return get_random_growth_rate

def create_max_length_callable(
        min_uniform: float=0.9,
        max_uniform: float=1.1,  
    ) -> Callable[[float], float]:
    def get_random_max_length(max_length: float) -> float:
        return max_length * np.random.uniform(min_uniform, max_uniform)
    return get_random_max_length

def create_width_callable(
        max_width: float,
        min_uniform: float=0.9,
        max_uniform: float=1.1,  
    ) -> Callable[[float], float]:
    def get_random_width(width: float) -> float:
        return min(width * np.random.uniform(min_uniform, max_uniform), max_width)
    return get_random_width


class CellIDFactory:
    def __init__(self, start_id: int=0):
        self._counter: int = start_id

    def get_next_id(self) -> int:
        next_id = self._counter
        self._counter += 1
        return next_id


class Cell:
    def __init__(self, cell_factory):
        self.cell_factory = cell_factory
        self.cell_id = self.cell_factory.increment_counter()

    def get_cell_id(self):
        return self.cell_id


class Cell2:
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
        mass: float = 1,
        friction: float = 0,
        lysis_p: float = 0,
        pinching_sep: float = 0,
        get_max_length_daughter: Callable[[float], float] = create_max_length_callable(),
        get_random_growth_rate: Callable[[], float] = create_growth_rate_callable(),
        mother_above_daughter: bool = False,
        division_ratio: float = 0.5,
    ):
        self.max_length: float = max_length
        self.mass: float = mass
        self.friction: float = friction
        self.geometry = cell_geometry.CellGeometry(length=length, width=width)
        pymunk_body_shape = Cell2.make_pymunk_cell(
                vertices=self.geometry.get_vertices().tolist(),
                mass=self.mass,
                angle=angle,
                position=position,
                friction=self.friction,
        )
        self.body: pymunk.Body = pymunk_body_shape[0]
        self.shape: pymunk.Poly = pymunk_body_shape[1]
        self.id_factory: CellIDFactory = id_factory
        self.self_id: int = self_id
        self.parent_id: int = parent_id
        self.daughter_ids: List[int] = daughter_ids
        self.lysis_p: float = lysis_p
        self.pinching_sep: float = pinching_sep
        self.get_width_daughter: Callable[[float], float] = get_width_daughter
        self.get_max_length_daughter: Callable[[], float] = get_max_length_daughter
        self.get_random_growth_rate: Callable[[], float] = get_random_growth_rate
        self.division_sign: int = 1 if mother_above_daughter else -1
        self.division_ratio: float = division_ratio

    def apply_force(self, force_global: Tuple[float, float]):
        force_local = pymunk.Vec2d(x=force_global[0], y=force_global[1]).rotated(-self.shape.body.angle)
        point_local = pymunk.Vec2d(x=np.random.uniform(low=-self.get_length()/2, high=self.get_length()/2), y=0)
        self.body.apply_force_at_local_point(force=force_local, point=point_local)

    def divide(self) -> 'Cell2':
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

        return Cell2(
            length=length_daughter,
            width=width_daughter,
            position=position_daugher,
            angle=self.get_angle(),
            max_length=max_length_daughter,
            id_factory=self.id_factory,
            self_id=daughter_id,
            parent_id=self.self_id,
            daughter_ids=[],
            mass=self.body.mass,
            friction=self.shape.friction,
            lysis_p=self.lysis_p,
            pinching_sep=0,
            get_width_daughter=self.get_width_daughter,
            get_max_length_daughter=self.get_max_length_daughter,
            get_random_growth_rate=self.get_random_growth_rate,
            mother_above_daughter=True if self.division_sign > 1 else False,
            division_ratio=self.division_ratio,
        )
    
    def get_angle(self) -> float:
        return self.body.angle
    
    def get_centroid(self) -> np.array:
        return self.get_position() + self.geometry.get_centroid(angle=self.get_angle())
    
    def get_length(self) -> float:
        return self.geometry.get_length()
    
    def get_position(self) -> Tuple[float, float]:
        return self.body.position
    
    def get_vertex_list(self, global_coordinates: bool=True) -> List[Tuple[float, float]]:
        if global_coordinates:
            return [self._to_global_coordinates(vertex) for vertex in self.shape.get_vertices()]
        else:
            return [(vertex[0], vertex[1]) for vertex in self.shape.get_vertices()]
    
    def get_width(self) -> float:
        return self.geometry.get_width()
    
    def grow(self, new_length: Optional[float]=None, new_pinching_sep: Optional[float]=None):
        new_length = self.get_length() * (1 + self.get_random_growth_rate()) if new_length is None else new_length
        self._update_vertices(new_length=new_length)
        self._update_pinching_sep(pinching_sep=new_pinching_sep)

    def is_dividing(self) -> bool: 
        return self.get_length() > self.max_length
    
    def is_out_of_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float, use_centroid: bool=False) -> bool:
        if use_centroid:
            c = self.get_centroid()
            return c[0] < x_min or c[0] > x_max or c[1] < y_min or c[1] > y_max
        else:
            return any([v[0] < x_min or v[0] > x_max or v[1] < y_min or v[1] > y_max for v in self.get_vertex_list()])
    
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
            mass: float,
            angle: Optional[float]=None,
            position: Optional[Tuple[float, float]]=None,
            friction: float=0.0,
        ) -> Tuple[pymunk.Body, pymunk.Poly]:

        shape = pymunk.Poly(None, vertices)
        shape.friction = friction
        moment = pymunk.moment_for_poly(mass, shape.get_vertices())  # TODO what is the difference between shape.get_vertices and our vertices?
        body = pymunk.Body(mass, moment)
        body.position = body.position if position is None else position
        body.angle = body.angle if angle is None else angle
        shape.body = body
        
        return body, shape

class Cell:
    """
    Cells are the agents in the simulation. This class allows for instantiating `Cell` object.

    .. note::
       Typically the user will not need to call this class, as it will be handled by :meth:`SyMBac.cell_simulation`,
       specifically all cell setup
       happens when instantiating a simulation using :meth:`SyMBac.simulation.Simulation`
    """
    def __init__(
        self,
        length: float,
        width: float,
        resolution: int,
        position: Tuple[float, float],
        angle: float,
        space: pymunk.Space,
        max_length: float,
        max_length_mean: float,
        max_length_var: float,
        width_var: float,
        width_mean: float,
        mass: float = 0.000001,
        friction: float = 0,
        ID: Optional[int] = None,
        parent:  Optional[int] = None,
        daughter:  Optional[int] = None,
        lysis_p: float = 0,
        pinching_sep: float = 0,
        get_random_growth_rate: Callable[[], float] = create_growth_rate_callable(),
        mother_above_daughter: bool = False,
    ):
        
        """
        Initialising a cell

        For info about the Pymunk objects, see the API reference. http://www.pymunk.org/en/latest/pymunk.html Cell class has been tested and works with pymunk version 6.0.0

        Parameters
        ----------
        length : float
            Cell's length
        width : float
            Cell's width
        resolution : int
            Number of points defining cell's geometry
        position : (float, float)
            x,y coords of cell centroid
        angle : float
            rotation in radians of cell (counterclockwise)
        space : pymunk.space.Space
            The pymunk space of the cell
        dt : float
            Timestep the cell experiences every iteration
        growth_rate_constant : float
            The cell grows by a function of dt*growth_rate_constant depending on its growth model
        max_length : float
            The maximum length a cell reaches before dividing
        max_length_mean : float
            should be the same as max_length for reasons unless doing advanced simulations
        max_length_var : float
            The variance defining a normal distribution around max_length
        width_var : float
            The variance defining a normal distribution around width
        width_mean : float
            For reasons should be set equal to width unless using advanced features
        body : pymunk.body.Body
            The cell's pymunk body object
        shape : pymunk.shapes.Poly
            The cell's pymunk body object
        ID : int
            A unique identifier for each cell. At the moment just a number from 0 to 100_000_000 and cross fingers that we get no collisions. 
            
        """
        self.length: float = length
        self.width: float = width
        self.resolution: int = resolution
        self.position: Tuple[float, float] = position
        self.angle: float = angle
        self.space: pymunk.Space = space
        self.max_length: float = max_length
        self.max_length_mean: float = max_length_mean
        self.max_length_var: float = max_length_var
        self.width_var: float = width_var
        self.width_mean: float = width_mean
        self.mass: float = mass
        self.friction: float = friction
        pymunk_body_shape = Cell.make_pymunk_cell(
                vertices=self.calculate_vertex_list(),
                mass=self.mass,
                angle=self.angle,
                position=self.position,
                friction=self.friction,
        )
        self.body: pymunk.Body = pymunk_body_shape[0]
        self.shape: pymunk.Shape = pymunk_body_shape[1]
        self.ID: int = np.random.randint(0,100_000_000) if ID is None else ID
        self.lysis_p: float = lysis_p
        self.parent: int = parent
        self.daughter: int = daughter
        self.pinching_sep: float = pinching_sep
        self.get_random_growth_rate: Callable[[], float] = get_random_growth_rate
        self.division_sign: int = 1 if mother_above_daughter else -1
        

    def create_pm_cell(self) -> Union[Tuple[pymunk.Body, pymunk.Shape], Dict[str, Union[int, float, Tuple, Callable, None]]]:
        """
        Creates a pymunk (pm) cell object, and places it into the pymunk space given when initialising the cell. If the
        cell is dividing, then two cells will be created. Typically this function is called for every cell, in every
        timestep to update the entire simulation.

        .. note::
           The return type of this function is dependent on the value returned by :meth:`SyMBac.cell.Cell.is_dividing()`.
           This is not good, and will be changed in a future version.

        Returns
        -------
        dict or (pymunk.body, pymunk.shape)

           If :meth:`SyMBac.cell.Cell.is_dividing()` returns `True`, then a dictionary of values for the daughter cell
           is returned. A daughter can then be created. E.g:

           >>> daughter_details = cell.create_pm_cell()
           >>> daughter = Cell(**daughter_details)

           If :meth:`SyMBac.cell.Cell.is_dividing()` returns `False`, then only a tuple containing (pymunk.body, pymunk.shape) will be returned.
        """

        if self.is_dividing():
            len_ratio = 0.3
            length_old = self.length
            self.length = (length_old-self.width) * (1-len_ratio) + self.width
            daughter_length = (length_old-self.width) * len_ratio + self.width
            self.pinching_sep = 0
            x_mother = self.position[0] - self.division_sign * self.length/2 *  np.sin(self.angle - np.pi/2)  # TODO sign here
            y_mother = self.position[1] + self.division_sign * self.length/2 *  np.cos(self.angle - np.pi/2)
            x_daughter = self.position[0] + self.division_sign * daughter_length/2 *  np.sin(self.angle - np.pi/2)
            y_daughter = self.position[1] - self.division_sign * daughter_length/2 *  np.cos(self.angle - np.pi/2)

            self.body, self.shape = Cell.make_pymunk_cell(
                vertices=self.calculate_vertex_list(),
                mass=self.mass,
                angle=self.angle,
                position=[x_mother, y_mother],
                friction=self.friction,
            )

            daughter_details = {
                "length": daughter_length,
                "width": np.random.normal(self.width_mean,self.width_var),
                "resolution": self.resolution,
                "position": [x_daughter, y_daughter],
                "angle": self.angle*np.random.uniform(0.95,1.05),
                "space": self.space,
                "max_length": np.random.normal(self.max_length_mean,self.max_length_var),
                "max_length_mean": self.max_length_mean,
                "max_length_var": self.max_length_var,
                "width_var": self.width_var,
                "width_mean": self.width_mean,
                "mass": self.mass,
                "friction": self.friction,
                "ID": None,
                "parent": self.parent,
                "daughter": None,
                "lysis_p": self.lysis_p,
                "pinching_sep": 0,
                "get_random_growth_rate": self.get_random_growth_rate,
                "mother_above_daughter": True if self.division_sign > 0 else False,
            } 
            return daughter_details
        else:
            self.body, self.shape =  Cell.make_pymunk_cell(
                vertices=self.calculate_vertex_list(),
                mass=self.mass,
                angle=self.angle,
                position=self.position,
                friction=self.friction,
            )
            return self.body, self.shape
    
    @staticmethod
    def make_pymunk_cell(
            vertices: List[Tuple[float, float]],
            mass: float,
            angle: Union[float, None]=None,
            position: Union[Tuple[float, float], None]=None,
            friction: float=0.0,
        ) -> Tuple[pymunk.Body, pymunk.Shape]:

        shape = pymunk.Poly(None, vertices)
        shape.friction = friction
        moment = pymunk.moment_for_poly(mass, shape.get_vertices())  # TODO what is the difference between shape.get_vertices and our vertices?
        body = pymunk.Body(mass, moment)
        body.position = body.position if position is None else position
        body.angle = body.angle if angle is None else angle
        shape.body = body
        
        return body, shape


    def is_dividing(self) -> bool: # This needs to be made constant or a cell can divide in one frame and not another frame
        """
        Checks whether a cell is dividing by comparing its current length to its max length (defined when instnatiated).

        Returns
        -------
        output : bool
            `True` if ``self.length > self.max_length``, else `False`.
        """
        return self.length > self.max_length


    def update_length(self):
        """
        A method, typically called every timepoint to update the length of the cell according to ``self.length = self.length + self.growth_rate_constant*self.dt*self.length``.

        Contains additional logic to control the amount of cell pinching happening according to the difference between
        the maximum length and the current length.

        Returns
        -------
        None
        """
        self.length = self.length * (1 + self.get_random_growth_rate())  # TODO simplify this using one constant callable
        self.pinching_sep = max(0, self.length - self.max_length + self.width)
        self.pinching_sep = min(self.pinching_sep, self.width - 2)

    def update_position(self):
        """
        A method, typically called every timepoint to keep synchronised the cell position (``self.position`` and ``self.angle``)
        with the position of the cell's corresponding body in the pymunk space (``self.body.position`` and ``self.body.angle``).

        Returns
        -------
        None
        """
        self.position = self.body.position
        self.angle = self.body.angle

    def update_parent(self, parent: int):
        """
        Parameters
        ----------
        parent : :class:`SyMBac.cell.Cell`
           The SyMBac cell object to assign as the parent to the current cell.

        Returns
        -------
        None
        """
        self.parent = parent

    def get_angle(self) -> float:
        """
        Gets the angle of the cell's pymunk body.

        Returns
        -------
        angle : float
           The cell's angle in radians.
        """
        return self.body.angle
    
    def calculate_vertex_list(self) -> List[List[float]]:
        return cell_geometry.get_vertices(
            cell_length=self.length,
            cell_width=self.width,
            angle=0,#self.angle, 
            resolution=self.resolution,
            )

    def get_vertex_list(self) -> List[List[float]]:
        """
        Calculates the vertex list (a set of x,y coordinates) which parameterise the outline of the cell

        Returns
        -------
        vertices : list(tuple(float, float))
           A list of vertices, each in a tuple, where the order is `(x, y)`. The coordinates are relative to the pymunk
           space in which the cell exists.
        """
        vertices = []
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position #.rotated(self.shape.body.angle)
            vertices.append((x,y))
        return vertices

    def get_centroid(self) -> Tuple[float, float]:
        """
        Calculates the centroid of the cell from the vertices.

        Returns
        -------
        centroid : float
            The cell's centroid in coordinates relative to the pymunk space which the cell exists in.
        """
        vertices = self.get_vertex_list()
        return cell_geometry.centroid(vertices)

    

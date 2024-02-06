from copy import deepcopy
import random
from typing import Callable, List, Tuple, Union

import numpy as np
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
from tqdm.auto import tqdm

from SyMBac.cell import Cell, CellIDFactory, create_width_callable
from SyMBac.config import ConfigCell, ConfigMothermachine, ConfigSimulation
from SyMBac.trench_geometry import trench_creator, get_trench_segments
from SyMBac.mothermachine_geometry import Mothermachine, MothermachinePart


def run_simulation(trench_length, trench_width, cell_max_length, cell_width, sim_length, pix_mic_conv, gravity,
                   phys_iters, max_length_var, width_var, save_dir, lysis_p=0, show_window = True, streamlit_mode = False):
    """
    Runs the rigid body simulation of bacterial growth based on a variety of parameters. Opens up a Pyglet window to
    display the animation in real-time. If the simulation looks bad to your eye, restart the kernel and rerun the
    simulation. There is currently a bug where if you try to rerun the simulation in the same kernel, it will be
    extremely slow.

    Parameters
    ----------

    trench_length : float
        Length of a mother machine trench (micron)
    trench_width : float
        Width of a mother machine trench (micron)
    cell_max_length : float
        Maximum length a cell can reach before dividing (micron)
    cell_width : float
        the average cell width in the simulation (micron)
    pix_mic_conv : float
        The micron/pixel size of the image
    gravity : float
        Pressure forcing cells into the trench. Typically left at zero, but can be varied if cells start to fall into
        each other or if the simulation behaves strangely.
    phys_iters : int
        Number of physics iterations per simulation frame. Increase to resolve collisions if cells are falling into one
        another, but decrease if cells begin to repel one another too much (too high a value causes cells to bounce off
        each other very hard). 20 is a good starting point
    max_length_var : float
        Variance of the maximum cell length
    width_var : float
        Variance of the maximum cell width
    save_dir : str
        Location to save simulation output
    lysis_p : float
        probability of cell lysis

    Returns
    -------
    cell_timeseries : lists
        A list of parameters for each cell, such as length, width, position, angle, etc. All used in the drawing of the
        scene later
    space : a pymunk space object
        Contains the rigid body physics objects which are the cells.
    """

    space = create_space()
    space.gravity = 0, gravity  # arbitrary units, negative is toward trench pole
    #space.iterations = 1000
    #space.damping = 0
    #space.collision_bias = 0.0017970074436457143*10
    space.collision_slop = 0.
    dt = 1 / 20  # time-step per frame
    pix_mic_conv = 1 / pix_mic_conv  # micron per pixel
    scale_factor = pix_mic_conv * 3  # resolution scaling factor

    trench_length = trench_length * scale_factor
    trench_width = trench_width * scale_factor
    trench_creator(trench_width, trench_length, (35, 0), space)  # Coordinates of bottom left corner of the trench

    cell1 = Cell(
        length=cell_max_length * scale_factor,
        width=cell_width * scale_factor,
        resolution=60,
        position=(20 + 35, 10),
        angle=0.8,
        space=space,
        dt= dt,
        growth_rate_constant=1,
        max_length=cell_max_length * scale_factor,
        max_length_mean=cell_max_length * scale_factor,
        max_length_var=max_length_var * np.sqrt(scale_factor),
        width_var=width_var * np.sqrt(scale_factor),
        width_mean=cell_width * scale_factor,
        parent=None,
        lysis_p=lysis_p
    )

    if show_window:

        window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
        options = DrawOptions()
        options.shape_outline_color = (10,20,30,40)
        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options)

        # key press event
        @window.event
        def on_key_press(symbol, modifier):

            # key "E" get press
            if symbol == pyglet.window.key.E:
                # close the window
                window.close()

    #global cell_timeseries
    #global x

    #try:
    #    del cell_timeseries
    #except:
    #    pass
    #try:
    #    del x
    #except:
    #    pass

    x = [0]
    cell_timeseries = []
    cells = [cell1]
    if show_window:
        pyglet.clock.schedule_interval(step_and_update, interval=dt, cells=cells, space=space, phys_iters=phys_iters,
                                       ylim=trench_length, cell_timeseries=cell_timeseries, x=x, sim_length=sim_length,
                                       save_dir=save_dir)
        pyglet.app.run()
    else:
        if streamlit_mode:
            import streamlit as st
            progress_text = "Simulation running"
            my_bar = st.progress(0, text=progress_text)
        for _ in tqdm(range(sim_length+2)):
            step_and_update(
                dt=dt, cells=cells, space=space, phys_iters=phys_iters, ylim=trench_length,
                cell_timeseries=cell_timeseries, x=x, sim_length=sim_length, save_dir=save_dir
            )
            if streamlit_mode:
                my_bar.progress((_)/sim_length, text=progress_text)

    # window.close()
    # phys_iters = phys_iters
    # for x in tqdm(range(sim_length+250),desc="Simulation Progress"):
    #    cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length*1.1, cell_timeseries = cell_timeseries, x=x, sim_length = sim_length, save_dir = save_dir)
    #    if x > 250:
    #        cell_timeseries.append(deepcopy(cells))
    return cell_timeseries, space

def create_space():
    """
    Creates a pymunk space

    :return pymunk.Space space: A pymunk space
    """

    space = pymunk.Space(threaded=False)
    #space.threads = 2
    return space

def run_simulation2(
        cfg_sim: ConfigSimulation,
        cfg_cell: ConfigCell,
        cfg_mm: ConfigMothermachine,
        old_version: bool = False,
    ) -> Tuple[List[List[Cell]], pymunk.Space, Mothermachine]:

    space = pymunk.Space(threaded=False)
    space.gravity = 0, 0.0
    space.collision_slop = 0.0
    dt = 1 / 20  # TODO why hard-coded?
    scale_factor = 1 / cfg_sim.pix_mic_conv  # TODO why factor 3?

    trench_length = cfg_mm.trench_length * scale_factor
    trench_width = cfg_mm.trench_width * scale_factor
    channel_width = cfg_mm.channel_width * scale_factor
    trench_spacing = cfg_mm.trench_spacing * scale_factor

    cell_max_length = cfg_cell.max_length * scale_factor
    cell_max_width = cfg_cell.max_width * scale_factor
    cell_init_width = cfg_cell.init_width * scale_factor

    mothermachine = Mothermachine(
        space=space,
        trench_width=trench_width,
        trench_length=trench_length,
        trench_spacing=trench_spacing,
        num_trenches=cfg_mm.num_trenches,
        channel_width=channel_width,
    )

    cell_id_factory = CellIDFactory()
    cells = []
    for i in mothermachine.get_trench_ids():
        cell_length = cell_max_length * 0.9
        trench_position = mothermachine.get_trench_position(trench_id=i)
        cell_position_x = trench_position[0] + trench_width * 0.5
        if mothermachine.is_top_trench(trench_id=i):
            cell_position_y = trench_position[1] + trench_length - cell_length * 0.5
            mother_above_daughter = True
        else:
            cell_position_y = trench_position[1] + cell_length * 0.5
            mother_above_daughter = False
        cell_pos = (cell_position_x, cell_position_y)
        cell_width = min(cell_max_width, 
                         cell_init_width * random.uniform(cfg_cell.init_width_min_uniform,
                                                          cfg_cell.init_width_max_uniform))
        # TODO create all callables from config here
        cell = Cell(
            length=cell_length,
            width=cell_width,
            position=cell_pos,
            id_factory=cell_id_factory,
            self_id = cell_id_factory.get_next_id(),
            angle=np.pi/2,
            max_length=cell_max_length,
            get_width_daughter=create_width_callable(max_width=cell_max_width),
            mother_above_daughter=mother_above_daughter,
        )
        cells.append(cell)
        cell.shape.color = (255, 0, 0, 255)
        space.add(cell.body, cell.shape)

    bb_mothermachine = mothermachine.get_bounding_box(which=MothermachinePart.MOTHERMACHINE)
    sim_progress = [0]
    cell_timeseries = []
    if cfg_sim.show_window:
        window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
        options = DrawOptions()
        options.shape_outline_color = (10,20,30,40)
        scale_x = window.width / (bb_mothermachine.right - bb_mothermachine.left)
        scale_y = window.height / (bb_mothermachine.top - bb_mothermachine.bottom)
        scale_window = min(scale_x, scale_y)
        window.view = window.view.scale((scale_window, scale_window, 1))

        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options)
        
        @window.event
        def on_key_press(symbol, modifier):
            if symbol == pyglet.window.key.E:
                window.close()

        pyglet.clock.schedule_interval_for_duration(
            step_and_update3, 
            interval=dt, 
            duration=cfg_sim.sim_length, 
            cells=cells, 
            space=space, 
            phys_iters=cfg_sim.num_physics_iter,
            cell_timeseries=cell_timeseries, 
            sim_progress=sim_progress, 
            sim_length=cfg_sim.sim_length,
            mothermachine=mothermachine
        )
        pyglet.app.run()
    else:
        for _ in tqdm(range(cfg_sim.sim_length)):
            step_and_update3(
                dt=dt, 
                cells=cells, 
                space=space, 
                phys_iters=cfg_sim.num_physics_iter,
                cell_timeseries=cell_timeseries, 
                sim_progress=sim_progress, 
                sim_length=cfg_sim.sim_length,
                mothermachine=mothermachine
            )
    if old_version:
        return cell_timeseries, space, mothermachine
    else:
        return cells, space, mothermachine


def step_and_update3(
        dt: float, 
        cells: List[Cell], 
        space: pymunk.Space, 
        phys_iters: int, 
        cell_timeseries: List[List[Cell]],
        sim_progress: List[int],
        sim_length: int,
        mothermachine: Mothermachine,
    ) -> Tuple[List[Cell]]:

    # Update cell growth, lysis and divisions
    for i in range(len(cells)):
        cell = cells[i]
        if mothermachine.where(cell=cell, use_centroid=True) == MothermachinePart.UNKNOWN:
            pass
            # cell.body.sleep()  # TODO this yields a seg fault when embedded in a callback
        else:
            cell.grow()
            if cell.is_dividing():
                daughter = cell.divide()
                cells.append(daughter)
                space.add(daughter.body, daughter.shape)
                if mothermachine.where(cell=daughter, use_centroid=True) == MothermachinePart.CHANNEL:
                    daughter.apply_force(mothermachine.get_flow_force())
            if mothermachine.where(cell=cell, use_centroid=True) == MothermachinePart.CHANNEL:
                cell.apply_force(mothermachine.get_flow_force())

    # Run physics solver and resolve collisions
    for _ in range(phys_iters):
        space.step(dt)

    # Record timeseries properties
    for cell in cells:
        cell.record_timeseries_properties(timestep=sim_progress[0])

    cell_timeseries.append(deepcopy(cells)) # TODO remove this
    if sim_progress[0] == sim_length-1: # TODO This is suboptimal. Should not need to terminate like this.
        pyglet.app.exit()
        return (cells)
    sim_progress[0] += 1
    return (cells)

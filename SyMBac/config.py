from dataclasses import dataclass


@dataclass
class ConfigCell:
    max_length: float
    "Max. length of the cell before it divides."
    max_width: float
    "Max. width possible."
    init_width: float
    "Width at initialisation computed from min(init_width*random.uniform(min_uniform, max_uniform), max_width)."
    growth_rate_constant: float
    "Determines growth rate of cell."

    lysis_probability: float = 0
    "Probability of lysis at every time step."

    division_ratio: float = 0.5
    "At division, length of daughter computed from length_mother*division_ratio."

    max_length_min_uniform: float = 0.9
    "At division, max_length computed from growth_rate_constant*random.uniform(min_uniform, max_uniform)."
    max_length_max_uniform: float = 1.1
    "At division, max_length computed from growth_rate_constant*random.uniform(min_uniform, max_uniform)."

    width_min_uniform: float = 0.9
    "At division, max_width computed from min(width * random.uniform(min_uniform, max_uniform), max_width)."
    width_max_uniform: float = 1.1
    "At division, max_width computed from min(width * random.uniform(min_uniform, max_uniform), max_width)."

    init_width_min_uniform: float = 0.8
    "Width at initialisation computed from min(init_width*random.uniform(min_uniform, max_uniform), max_width)."
    init_width_max_uniform: float = 1.2
    "Width at initialisation computed from min(init_width*random.uniform(min_uniform, max_uniform), max_width)."
    

TEST_CONFIG_CELL = ConfigCell(
    max_length=30.0,
    max_width=7.5,
    init_width=7,
    growth_rate_constant=0.01,
)


@dataclass
class ConfigMothermachine:
    trench_length: float
    "Length of a trench."
    trench_width: float
    "Width of a trench."
    trench_spacing: float
    "Spacing between trenches."
    channel_width: float
    "Width of media channel."
    num_trenches: int
    "Number of trenches on one side of the media channel."

TEST_CONFIG_MOTHERMACHINE = ConfigMothermachine(
    trench_length=92.0,
    trench_width=32.0,
    trench_spacing=92.0,
    channel_width=184.0,
    num_trenches=2,
)

@dataclass
class ConfigSimulation:
    pix_mic_conv: float
    "TODO Pixel conversion factor."
    sim_length: int
    "Total length of simulation."
    show_window: bool
    "Show window during physics simulation."
    num_physics_iter: int
    "Number of physics solver iterations at each timestep."
    save_dir: str
    "TODO change this to Path."
    resize_amount: float
    "TODO How does this differ from pix_mic_conv?"
    offset: float
    "TODO Why is this neccesary? Why only one offset for one direction?"

TEST_CONFIG_SIMULATION = ConfigSimulation(
    pix_mic_conv=0.065,
    sim_length=20,
    show_window=True,
    num_physics_iter=10,
    save_dir="/home/tmp/",
    resize_amount=1,
    offset=30.0,
)
import os
import numpy as np
from itertools import product

# ============================================================
# CONFIGURATION
# ============================================================

MAIN_FOLDER = "generated_cases_linspace"

# Number of levels for each continuous parameter
N_LEVELS = 5

# ------------------------------------------------------------
# IMPORTANT:
# Set these to your actual ISLAB slab dimensions.
# Units: metres
# ------------------------------------------------------------

X_LENGTH_MIN = 3.5
X_LENGTH_MAX = 4.5

Y_WIDTH_MIN = 3.5
Y_WIDTH_MAX = 4.5

# Mesh density
NX_MIN = 2
NX_MAX = 8

NY_MIN = 2
NY_MAX = 8

# Number of mesh-density levels
N_MESH_LEVELS = 4

# ------------------------------------------------------------
# Material / foundation / loading ranges
# ------------------------------------------------------------

E_MIN = 27600          # MPa
E_MAX = 41400          # MPa

THICKNESS_MIN = 150    # mm
THICKNESS_MAX = 380    # mm

ALPHA_MIN = 6.9e-6     # /deg C
ALPHA_MAX = 12e-6      # /deg C

K_MIN = 27             # MPa/m
K_MAX = 300            # MPa/m

TEMP_MIN = 0           # deg C
TEMP_MAX = 50          # deg C

PRESSURE_MIN = 0.7     # MPa
PRESSURE_MAX = 1.0     # MPa


# ============================================================
# PRESSURE PATCH
# ============================================================
#
# Instead of independently generating X1, X2, Y1, Y2,
# we define:
#
#   pressure patch length
#   pressure patch width
#   patch centre position
#
# This guarantees the patch remains INSIDE the slab.
#
# Modify these according to your tyre/contact model.
# ============================================================

PRESSURE_LENGTH_MIN = 0.20   # m
PRESSURE_LENGTH_MAX = 0.60   # m

PRESSURE_WIDTH_MIN = 0.20    # m
PRESSURE_WIDTH_MAX = 0.60    # m


# ============================================================
# TEMPLATE
# ============================================================

template = """*GENERAL

v2.2

Test1

0 10 0

1 1 1 0.1

0



*GEOMETRY

1 1

{NX}

{NY}

{X_COORDS}

{Y_COORDS}



*LAYER

1 0



21 {E} 0.15 {THICKNESS} {ALPHA} 23600



*FOUNDATION

51 {KVALUE} 1



*TEMPERATURE

1 1 {TEMP} 1



*PRESSURE

{PRESSURE} {X1} {X2} {Y1} {Y2}

"""


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(MAIN_FOLDER, exist_ok=True)


# ============================================================
# LINSPACE PARAMETERS
# ============================================================

E_VALUES = np.linspace(
    E_MIN,
    E_MAX,
    N_LEVELS
)

THICKNESS_VALUES = np.linspace(
    THICKNESS_MIN,
    THICKNESS_MAX,
    N_LEVELS
)

ALPHA_VALUES = np.linspace(
    ALPHA_MIN,
    ALPHA_MAX,
    N_LEVELS
)

K_VALUES = np.linspace(
    K_MIN,
    K_MAX,
    N_LEVELS
)

TEMP_VALUES = np.linspace(
    TEMP_MIN,
    TEMP_MAX,
    N_LEVELS
)

PRESSURE_VALUES = np.linspace(
    PRESSURE_MIN,
    PRESSURE_MAX,
    N_LEVELS
)

X_LENGTH_VALUES = np.linspace(
    X_LENGTH_MIN,
    X_LENGTH_MAX,
    N_LEVELS
)

Y_WIDTH_VALUES = np.linspace(
    Y_WIDTH_MIN,
    Y_WIDTH_MAX,
    N_LEVELS
)

NX_VALUES = np.linspace(
    NX_MIN,
    NX_MAX,
    N_MESH_LEVELS,
    dtype=int
)

NY_VALUES = np.linspace(
    NY_MIN,
    NY_MAX,
    N_MESH_LEVELS,
    dtype=int
)


# ============================================================
# GENERATE PRESSURE PATCH
# ============================================================

def generate_pressure_patch(
    slab_length,
    slab_width,
    pressure_length,
    pressure_width
):
    """
    Generate a pressure rectangle guaranteed to lie
    completely inside the slab.
    """

    # Centre of slab
    center_x = slab_length / 2
    center_y = slab_width / 2

    x1 = center_x - pressure_length / 2
    x2 = center_x + pressure_length / 2

    y1 = center_y - pressure_width / 2
    y2 = center_y + pressure_width / 2

    return x1, x2, y1, y2


# ============================================================
# GENERATE CASES
# ============================================================

case_number = 1

for (
    slab_length,
    slab_width,
    nx,
    ny,
    E,
    thickness,
    alpha,
    k_value,
    temperature,
    pressure,
) in product(
    X_LENGTH_VALUES,
    Y_WIDTH_VALUES,
    NX_VALUES,
    NY_VALUES,
    E_VALUES,
    THICKNESS_VALUES,
    ALPHA_VALUES,
    K_VALUES,
    TEMP_VALUES,
    PRESSURE_VALUES,
):

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    # np.linspace guarantees evenly distributed nodes
    x_coords = np.linspace(
        0,
        slab_length,
        int(nx)
    )

    y_coords = np.linspace(
        0,
        slab_width,
        int(ny)
    )

    # Convert to ISLAB-compatible strings
    x_coords_str = " ".join(
        f"{x:.4f}" for x in x_coords
    )

    y_coords_str = " ".join(
        f"{y:.4f}" for y in y_coords
    )

    # --------------------------------------------------------
    # Pressure footprint
    # --------------------------------------------------------

    # For now use the midpoint of the allowed footprint range.
    # You can later make footprint size another DOE variable.
    pressure_length = (
        PRESSURE_LENGTH_MIN +
        PRESSURE_LENGTH_MAX
    ) / 2

    pressure_width = (
        PRESSURE_WIDTH_MIN +
        PRESSURE_WIDTH_MAX
    ) / 2

    # Make sure footprint fits
    if pressure_length > slab_length:
        continue

    if pressure_width > slab_width:
        continue

    x1, x2, y1, y2 = generate_pressure_patch(
        slab_length,
        slab_width,
        pressure_length,
        pressure_width
    )

    # --------------------------------------------------------
    # Create case directory
    # --------------------------------------------------------

    case_folder = os.path.join(
        MAIN_FOLDER,
        f"Case_{case_number:05d}"
    )

    os.makedirs(case_folder, exist_ok=True)

    # --------------------------------------------------------
    # Fill template
    # --------------------------------------------------------

    content = template.format(
        NX=int(nx),
        NY=int(ny),

        X_COORDS=x_coords_str,
        Y_COORDS=y_coords_str,

        E=f"{E:.2f}",

        THICKNESS=f"{thickness:.2f}",

        ALPHA=f"{alpha:.7E}",

        KVALUE=f"{k_value:.2f}",

        TEMP=f"{temperature:.2f}",

        PRESSURE=f"{pressure:.4f}",

        X1=f"{x1:.4f}",
        X2=f"{x2:.4f}",
        Y1=f"{y1:.4f}",
        Y2=f"{y2:.4f}",
    )

    # --------------------------------------------------------
    # Write input
    # --------------------------------------------------------

    input_file = os.path.join(
        case_folder,
        "input.in2"
    )

    with open(
        input_file,
        "w"
    ) as f:
        f.write(content)

    case_number += 1


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("ISLAB CASE GENERATION COMPLETE")
print("=" * 60)

print(f"Cases generated : {case_number - 1}")
print(f"Output folder   : {MAIN_FOLDER}")
print(f"Levels/parameter: {N_LEVELS}")
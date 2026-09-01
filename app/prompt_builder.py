"""Build the FLUX prompt for one generation request.

The prompt is assembled from what this request actually knows:

    1. the room's measured dimensions, from the solved `SceneGeometry`, so the
       model is told the room's real size rather than left to infer it;
    2. a dimensional inventory, one row per reference, with the catalogue's real
       centimetres;
    3. one reproduction block per reference, keyed to the furniture's *type* - a
       sofa gets asked about chaise direction and cushion count, a sideboard
       about door style and handle finish, a floor lamp about shade diameter;
    4. `ESSENTIALS`, the standing directives that apply to every request.

The type is what selects the checklist, so it is resolved generously: an explicit
`type` on the placement wins, then a `type`/`category` field on the catalogue
metadata, then keyword matching against the product name. Names in this catalogue
are German (Hoeffner), so the aliases carry both languages.

WHAT THIS MODULE DOES NOT DO
    It does not decide which images are sent to FLUX. It labels each reference by
    the position it will occupy in the request (`image 2`, `image 3`, ...) and by
    filename, and the caller is responsible for the two agreeing.
"""

import re
from dataclasses import dataclass, field

# The standing directives sent with every request. Kept short on purpose:
# FLUX.2's text encoder truncates at 512 tokens, so this covers only what the
# model alone decides - how the pieces are lit, how they meet the floor, and
# whether they stay the product they came from. Placement and scale are
# already settled by the composite, and saying so again invites it to move
# things.
ESSENTIALS = """The room is fixed. Keep the walls, floor, ceiling, windows, blinds, molding, baseboards, outlets, vents and doors exactly as photographed, and keep the camera position, height, angle and framing unchanged.

Light every inserted piece with the room's own light. Preserve the existing daylight direction, window illumination, blind shadows, shadow softness, colour temperature and exposure. The side of each piece facing the window is brighter and the opposite side falls into shadow; every new shadow runs in the same direction, and with the same softness, as the shadows already in the room. Add no studio lighting, no rim light and no cinematic grading, and do not change the time of day.

Every piece rests on the floor and touches it. Each needs a contact shadow beneath its base, ambient occlusion where it meets the floor and walls, and correct occlusion where pieces overlap. Nothing floats, nothing looks pasted on, and no outline, halo or cut-out edge surrounds any object.

Reproduce each product exactly as its reference shows it and keep its supplied dimensions. Do not substitute a similar piece, redesign it, or change its colour, material or finish to suit the room.

Return one single photorealistic photograph of this room - not a collage, not a moodboard, not a product render, with no reference backgrounds and no CGI appearance."""


# --------------------------------------------------------------------------- #
# furniture types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FurnitureType:
    """One furniture category and what must be looked at to reproduce it.

    `attributes` is the checklist that becomes the "Analyze:" bullet list of the
    reproduction block. `generic` names the thing the model must not fall back
    to. "Do not replace it with another generic cream sectional" does more
    work than any positive instruction, because a generic substitute is the
    actual failure mode.
    """

    slug: str
    label: str  # how the block refers to it, lowercase
    aliases: tuple  # keywords matched against the product name
    attributes: tuple
    generic: str  # the substitute to forbid


_COMMON_TAIL = (
    "Upholstery or surface colour",
    "Material texture and finish",
    "Seam, joint and panel placement",
    "Visible hardware, its finish and its placement",
)

TYPES = {
    t.slug: t
    for t in (
        FurnitureType(
            slug="sofa",
            label="sofa",
            aliases=("sofa", "couch", "sectional", "settee", "chaise", "ecksofa", "wohnlandschaft"),
            attributes=(
                "L-shaped or straight configuration",
                "Chaise direction, if any",
                "Overall width",
                "Overall height",
                "Overall depth",
                "Seat depth",
                "Seat height, if visible/provided",
                "Armrest height",
                "Armrest shape and width",
                "Back cushion arrangement",
                "Seat cushion arrangement",
                "Number of cushions",
                "Number and shape of pillows",
                "Upholstery colour",
                "Upholstery texture",
                "Cushion softness",
                "Seam placement",
                "Leg shape",
                "Leg material",
                "Leg colour",
                "Distance between legs",
                "Relationship between chaise and main sofa",
            ),
            generic="generic cream sectional",
        ),
        FurnitureType(
            slug="armchair",
            label="armchair",
            aliases=("armchair", "arm chair", "lounge chair", "sessel", "ohrensessel", "fauteuil"),
            attributes=(
                "Overall width, height and depth",
                "Seat height",
                "Seat depth",
                "Backrest height and shape",
                "Wing or wingless back",
                "Armrest height, shape and width",
                "Number and arrangement of cushions",
                "Buttoning, tufting or piping, if present",
                "Upholstery colour",
                "Upholstery texture",
                "Seam placement",
                "Swivel, rocker or fixed base",
                "Leg or base shape",
                "Leg material",
                "Leg colour",
                "Distance between legs",
            ),
            generic="generic upholstered armchair",
        ),
        FurnitureType(
            slug="dining_chair",
            label="dining chair",
            aliases=("dining chair", "stuhl", "esszimmerstuhl", "side chair", "chair", "freischwinger"),
            attributes=(
                "Overall width, height and depth",
                "Seat height",
                "Seat shape and whether it is upholstered",
                "Backrest height and shape",
                "Back slat, spindle or panel arrangement",
                "Presence and shape of armrests",
                "Frame material and finish",
                "Leg shape and taper",
                "Cantilever or four-leg base",
                "Stretchers between legs, if present",
                "Upholstery colour and texture, if upholstered",
                "Seam and stitching placement",
            ),
            generic="generic wooden dining chair",
        ),
        FurnitureType(
            slug="sideboard",
            label="sideboard",
            aliases=("sideboard", "credenza", "buffet", "anrichte", "highboard", "lowboard"),
            attributes=(
                "Overall width, height and depth",
                "Number of doors",
                "Number of drawers",
                "Door style: flat, panelled, fluted, reeded or slatted",
                "Drawer front alignment and gaps",
                "Handle type, shape and finish",
                "Handle placement",
                "Plinth, base frame or legs",
                "Leg shape, material and colour",
                "Carcass material and finish",
                "Wood grain direction, if wood",
                "Top surface material and edge profile",
                "Interior shelves, if visible",
            ),
            generic="generic oak sideboard",
        ),
        FurnitureType(
            slug="dresser",
            label="chest of drawers",
            aliases=("dresser", "chest of drawers", "kommode", "schubladenkommode"),
            attributes=(
                "Overall width, height and depth",
                "Number of drawers",
                "Number of drawer rows and columns",
                "Drawer front style and gaps",
                "Handle type, shape and finish",
                "Handle placement",
                "Plinth or legs",
                "Leg shape, material and colour",
                "Carcass material and finish",
                "Wood grain direction, if wood",
                "Top surface and edge profile",
            ),
            generic="generic white chest of drawers",
        ),
        FurnitureType(
            slug="wardrobe",
            label="wardrobe",
            aliases=("wardrobe", "armoire", "kleiderschrank", "schrank", "closet"),
            attributes=(
                "Overall width, height and depth",
                "Number of doors",
                "Hinged or sliding doors",
                "Door panel arrangement",
                "Presence and placement of mirrors",
                "Handle type, shape and finish",
                "Plinth, cornice or flush top",
                "Carcass material and finish",
                "Wood grain direction, if wood",
                "Visible interior, if any door is open",
            ),
            generic="generic white wardrobe",
        ),
        FurnitureType(
            slug="dining_table",
            label="dining table",
            aliases=("dining table", "esstisch", "esszimmertisch", "kitchen table"),
            attributes=(
                "Overall width, height and depth",
                "Top shape: rectangular, square, round or oval",
                "Top thickness",
                "Top edge profile",
                "Top material and finish",
                "Wood grain direction, if wood",
                "Base type: four legs, trestle, pedestal or cantilever",
                "Leg shape and taper",
                "Leg position relative to the top edge",
                "Presence and depth of an apron",
                "Extension leaves, if visible",
                "Leg material and colour",
            ),
            generic="generic rectangular dining table",
        ),
        FurnitureType(
            slug="coffee_table",
            label="coffee table",
            aliases=("coffee table", "couchtisch", "cocktail table"),
            attributes=(
                "Overall width, height and depth",
                "Top shape",
                "Top thickness and edge profile",
                "Top material and finish",
                "Lower shelf or storage, if present",
                "Base type: legs, pedestal, frame or block",
                "Leg shape and taper",
                "Leg material and colour",
                "Distance between legs",
                "Nesting or single piece",
            ),
            generic="generic round coffee table",
        ),
        FurnitureType(
            slug="side_table",
            label="side table",
            aliases=("side table", "end table", "beistelltisch", "nightstand", "nachttisch", "bedside"),
            attributes=(
                "Overall width, height and depth",
                "Top shape and diameter or dimensions",
                "Top material and finish",
                "Drawers or shelves, if present",
                "Handle type and finish, if present",
                "Base type and leg shape",
                "Leg material and colour",
            ),
            generic="generic round side table",
        ),
        FurnitureType(
            slug="desk",
            label="desk",
            aliases=("desk", "schreibtisch", "writing table", "bureau"),
            attributes=(
                "Overall width, height and depth",
                "Top material, finish and thickness",
                "Number and placement of drawers",
                "Handle type and finish",
                "Modesty panel, if present",
                "Cable ports or grommets, if visible",
                "Frame or leg structure",
                "Leg shape, material and colour",
            ),
            generic="generic office desk",
        ),
        FurnitureType(
            slug="bed",
            label="bed",
            aliases=("bed", "bett", "boxspring", "doppelbett", "bedstead"),
            attributes=(
                "Overall width, height and depth",
                "Headboard height",
                "Headboard shape and profile",
                "Headboard upholstery, buttoning or panelling",
                "Presence and height of a footboard",
                "Frame material and finish",
                "Mattress height above the frame",
                "Base type: legs, platform or plinth",
                "Leg shape, material and colour",
                "Upholstery colour and texture, if upholstered",
                "Visible bedding, if part of the product",
            ),
            generic="generic upholstered double bed",
        ),
        FurnitureType(
            slug="shelf",
            label="shelving unit",
            aliases=("shelf", "shelving", "bookcase", "bookshelf", "regal", "bücherregal", "etagere"),
            attributes=(
                "Overall width, height and depth",
                "Number of shelves",
                "Shelf spacing and whether it is even",
                "Shelf thickness",
                "Open back or backing panel",
                "Upright and frame structure",
                "Closed compartments or drawers, if present",
                "Material and finish",
                "Wood grain direction, if wood",
                "Wall-mounted or floor-standing",
                "Base or leg structure",
            ),
            generic="generic open bookcase",
        ),
        FurnitureType(
            slug="tv_unit",
            label="TV unit",
            aliases=("tv unit", "tv stand", "media console", "tv-board", "tv board", "lowboard tv"),
            attributes=(
                "Overall width, height and depth",
                "Number of doors, drawers and open compartments",
                "Compartment arrangement",
                "Door and drawer front style",
                "Handle type and finish, or push-to-open",
                "Cable management openings",
                "Base: plinth, legs or wall-mounted",
                "Leg shape, material and colour",
                "Carcass material and finish",
            ),
            generic="generic black TV stand",
        ),
        FurnitureType(
            slug="floor_lamp",
            label="floor lamp",
            aliases=("floor lamp", "stehlampe", "standleuchte", "arc lamp", "bogenlampe"),
            attributes=(
                "Overall height",
                "Base diameter and shape",
                "Base material, finish and colour",
                "Stem shape: straight, arced or articulated",
                "Stem material and finish",
                "Shade shape",
                "Shade diameter and height",
                "Shade material and opacity",
                "Shade colour, inside and out",
                "Shade angle relative to the stem",
                "Number of light heads",
                "Visible switch, cable and plug",
            ),
            generic="generic tripod floor lamp",
        ),
        FurnitureType(
            slug="table_lamp",
            label="table lamp",
            aliases=("table lamp", "tischlampe", "tischleuchte", "desk lamp", "nachttischlampe"),
            attributes=(
                "Overall height",
                "Base shape, diameter and proportion to the shade",
                "Base material, finish and colour",
                "Shade shape",
                "Shade diameter and height",
                "Shade material and opacity",
                "Shade colour, inside and out",
                "Visible switch and cable",
            ),
            generic="generic ceramic table lamp",
        ),
        FurnitureType(
            slug="pendant_lamp",
            label="pendant light",
            aliases=("pendant", "hängeleuchte", "hängelampe", "chandelier", "kronleuchter", "ceiling lamp"),
            attributes=(
                "Overall drop height",
                "Shade or diffuser shape",
                "Shade diameter",
                "Shade material and opacity",
                "Shade colour, inside and out",
                "Number of light heads and their arrangement",
                "Suspension: rod, cable or chain",
                "Canopy shape and finish",
                "Metal finish",
            ),
            generic="generic dome pendant light",
        ),
        FurnitureType(
            slug="rug",
            label="rug",
            aliases=("rug", "teppich", "carpet", "läufer", "runner"),
            attributes=(
                "Overall width and length",
                "Shape: rectangular, round or runner",
                "Pile height",
                "Pile texture: flat-woven, tufted, shag or looped",
                "Pattern and its scale",
                "Pattern placement relative to the edges",
                "All colours present and their proportion",
                "Border or binding",
                "Fringe, if present",
            ),
            generic="generic beige area rug",
        ),
        FurnitureType(
            slug="ottoman",
            label="ottoman",
            aliases=("ottoman", "pouf", "footstool", "hocker", "sitzhocker", "bench", "bank"),
            attributes=(
                "Overall width, height and depth",
                "Shape: round, square or rectangular",
                "Top profile: flat, domed or buttoned",
                "Buttoning or tufting pattern",
                "Upholstery colour and texture",
                "Seam placement",
                "Base: legs, plinth or fully upholstered",
                "Leg shape, material and colour",
                "Storage function, if visible",
            ),
            generic="generic round pouf",
        ),
        FurnitureType(
            slug="mirror",
            label="mirror",
            aliases=("mirror", "spiegel", "wandspiegel"),
            attributes=(
                "Overall width and height",
                "Shape: round, oval, rectangular or arched",
                "Frame width and depth",
                "Frame material and finish",
                "Frame profile",
                "Bevel on the glass, if present",
                "Wall-mounted or leaning",
            ),
            generic="generic round black-framed mirror",
        ),
        FurnitureType(
            slug="plant",
            label="plant",
            aliases=("plant", "pflanze", "tree", "baum", "palm", "monstera", "ficus"),
            attributes=(
                "Overall height including the pot",
                "Species and leaf shape",
                "Leaf size, colour and density",
                "Trunk or stem structure",
                "Pot shape and height",
                "Pot material, finish and colour",
                "Visible soil or covering",
            ),
            generic="generic potted fiddle-leaf fig",
        ),
        FurnitureType(
            slug="artwork",
            label="artwork",
            aliases=("artwork", "picture", "poster", "bild", "gemälde", "wall art", "print", "canvas"),
            attributes=(
                "Overall width and height",
                "Orientation: portrait or landscape",
                "Frame presence, width and finish",
                "Mount or passepartout, if present",
                "Subject and composition of the image",
                "Dominant colours",
                "Canvas, glass or matte surface",
            ),
            generic="generic abstract canvas print",
        ),
    )
}

GENERIC_TYPE = FurnitureType(
    slug="furniture",
    label="furniture item",
    aliases=(),
    attributes=(
        "Overall width, height and depth",
        "Overall silhouette and proportions",
        "Configuration and number of parts",
        "Every visible panel, door, drawer and shelf",
        "Material and finish of each surface",
        "Colour of each surface",
        "Visible hardware, its shape and finish",
        "Base or leg structure",
        "Leg shape, material and colour",
        "Seam, joint and edge details",
        "Any distinctive detail that identifies this specific product",
    ),
    generic="generic substitute of the same category",
)

# Longest alias first: "dining chair" must beat "chair", "coffee table" must beat
# "table", or a keyword match on the shorter one wins by accident.
_ALIAS_INDEX = sorted(
    ((alias, t) for t in TYPES.values() for alias in t.aliases),
    key=lambda pair: -len(pair[0]),
)


def resolve_type(explicit=None, metadata=None, name=None):
    """Pick the furniture type, most trustworthy source first.

    An explicit slug on the placement beats the catalogue, which beats guessing
    from the product name. An unrecognised explicit slug is an error worth
    raising rather than silently falling back, since it means the caller asked
    for a checklist that does not exist.
    """
    if explicit:
        slug = _slugify(explicit)
        if slug not in TYPES:
            raise ValueError(
                f"unknown furniture type {explicit!r} — known types: "
                + ", ".join(sorted(TYPES))
            )
        return TYPES[slug]

    for candidate in (metadata or {}).get("type"), (metadata or {}).get("category"):
        if candidate:
            slug = _slugify(candidate)
            if slug in TYPES:
                return TYPES[slug]

    haystack = " ".join(part for part in (name, (metadata or {}).get("name")) if part).lower()
    for alias, furniture_type in _ALIAS_INDEX:
        if alias in haystack:
            return furniture_type
    return GENERIC_TYPE


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# one reference
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    """One furniture reference as the prompt needs to talk about it."""

    filename: str
    furniture_type: FurnitureType
    name: str = ""
    width_cm: float = None
    height_cm: float = None
    depth_cm: float = None
    image_index: int = None  # 1-based position in the FLUX reference list
    _note: str = field(default="", repr=False)

    @property
    def dimension_text(self):
        """`180 × 90 × 40 cm`, with `?` for anything the catalogue lacks."""
        parts = [
            f"{value:g}" if value else "?"
            for value in (self.width_cm, self.height_cm, self.depth_cm)
        ]
        return " × ".join(parts) + " cm"

    @property
    def label(self):
        return self.name or self.furniture_type.label

    def inventory_row(self, position):
        """One row of the dimensional inventory."""
        return (
            f"{position:02d} — `{self.filename}` — {self.label} — "
            f"{self.furniture_type.label} — W × H × D: {self.dimension_text}"
        )

    def reproduction_block(self):
        """A section-11-style block, but for this reference and its type."""
        kind = self.furniture_type.label
        where = f"`{self.filename}`"
        if self.image_index:
            where += f" (image {self.image_index})"

        lines = [
            f"If {where} is the {kind} reference:",
            "",
            f"Identify it as a specific {kind} object.",
            "",
            "Analyze:",
            "",
        ]
        lines += [f"* {attribute}" for attribute in self.furniture_type.attributes]
        lines += [
            "",
            f"Then reproduce THAT {kind}.",
            "",
            f"Do not replace it with another {self.furniture_type.generic}.",
            "",
            f"The final {kind} should immediately resemble the exact furniture "
            f"shown in `{self.filename}`.",
            "",
        ]
        if self._known_dimensions():
            lines.append(
                f"The {kind} measures {self.dimension_text} (width × height × depth). "
                "These dimensions are authoritative and must not be altered to make "
                "it fit the room."
            )
        else:
            lines.append(
                f"No dimensions are supplied for the {kind}. Estimate them "
                "conservatively from the reference image's proportions."
            )
        if self._note:
            lines += ["", self._note]
        return "\n".join(lines)

    def _known_dimensions(self):
        return any(v for v in (self.width_cm, self.height_cm, self.depth_cm))


# --------------------------------------------------------------------------- #
# the room
# --------------------------------------------------------------------------- #
def room_metrics(geometry):
    """Measure the room from a solved `SceneGeometry`, in metres.

    These are the *visible* extents, not the room's true dimensions: the floor
    is only reconstructed where it can be seen, so a floor hidden behind a bed
    shortens the measurement. Reported as such, because telling the model a
    confident wrong number is worse than telling it an honest approximate one.

    Accuracy is bounded by `camera_height_m` exactly as everything else is —
    it is one global scale factor, so a camera height that was guessed makes
    these numbers proportionally wrong.
    """
    import numpy as np

    x_min, x_max = geometry.floor_span
    width, height = geometry.size
    scale = geometry.units_per_metre

    left = geometry.floor_point(x_min, geometry.floor_y_at(x_min))
    right = geometry.floor_point(x_max, geometry.floor_y_at(x_max))
    back_wall_width = float(np.linalg.norm(right - left)) / scale

    centre_x = (x_min + x_max) / 2.0
    far = geometry.floor_point(centre_x, geometry.floor_y_at(centre_x))
    near = geometry.floor_point(centre_x, height - 1)
    depth_to_back_wall = float(far[2]) / scale
    visible_floor_depth = float(far[2] - near[2]) / scale

    return {
        "back_wall_width_m": back_wall_width,
        "depth_to_back_wall_m": depth_to_back_wall,
        "visible_floor_depth_m": visible_floor_depth,
        "camera_height_m": geometry.camera_height_m,
        "image_size": (width, height),
    }


def room_block(geometry, scene_name="room.png"):
    """The measured-room section, or a note that the room was not measured."""
    if geometry is None:
        return (
            f"No measured dimensions are available for `{scene_name}`. Read the "
            "room's scale from the architecture visible in the image and from the "
            "supplied furniture dimensions."
        )

    try:
        metrics = room_metrics(geometry)
    except (ValueError, ZeroDivisionError) as exc:
        # A floor that could not be measured is not a reason to fail the whole
        # generation; the model still has the photograph.
        return (
            f"The floor of `{scene_name}` could not be measured ({exc}). Read the "
            "room's scale from the architecture visible in the image."
        )

    return "\n".join(
        [
            f"`{scene_name}` has been reconstructed in 3D. Its measured dimensions are:",
            "",
            f"* Width along the far wall: {metrics['back_wall_width_m']:.2f} m",
            f"* Distance from the camera to the far wall: {metrics['depth_to_back_wall_m']:.2f} m",
            f"* Depth of visible floor: {metrics['visible_floor_depth_m']:.2f} m",
            f"* Camera height above the floor: {metrics['camera_height_m']:.2f} m",
            "",
            "These are the extents of the floor that is *visible* in the photograph, "
            "so the real room may continue past them.",
            "",
            "Use them to judge whether the supplied furniture dimensions fit the space. "
            "The furniture dimensions remain authoritative: if a piece is too large for "
            "the room, place it so that it is partly out of frame rather than shrinking it.",
        ]
    )


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_prompt(
    user_prompt,
    references,
    geometry=None,
    scene_name="room.png",
    include_essentials=True,
):
    """The prompt FLUX is actually sent.

    Section order is chosen for truncation, not for reading: a text encoder with
    a fixed sequence length drops the *tail*, so the sections run most valuable
    first. The per-reference reproduction blocks come last despite being the
    point of this module - they are by far the largest section, and a 22-item
    "Analyze:" checklist instructs a reasoning step FLUX does not have, so they
    are what can be lost most cheaply.
    """
    sections = []

    sections.append("# THE ROOM\n\n" + room_block(geometry, scene_name))

    if references:
        rows = [ref.inventory_row(i) for i, ref in enumerate(references, start=1)]
        sections.append(
            "\n\n".join(
                [
                    "# FURNITURE IN THIS REQUEST",
                    "This inventory is complete; do not add pieces to it.",
                    "\n\n".join(rows),
                ]
            )
        )

    if include_essentials:
        sections.append("# STANDING DIRECTIVES\n\n" + ESSENTIALS)

    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        sections.append(
            "\n\n".join(
                [
                    "# WHAT WAS ASKED FOR",
                    user_prompt,
                    "This describes arrangement and mood only. It licenses no change "
                    "to the room, the camera, or the identity or dimensions of the "
                    "furniture.",
                ]
            )
        )

    if references:
        blocks = [
            f"## Reference {i} - {ref.furniture_type.label}\n\n{ref.reproduction_block()}"
            for i, ref in enumerate(references, start=1)
        ]
        sections.append(
            "\n\n".join(
                [
                    "# EXACT REPRODUCTION, PER REFERENCE",
                    "\n\n---\n\n".join(blocks),
                ]
            )
        )

    return "\n\n---\n\n".join(sections)


def reference_from_catalogue(metadata, spec=None, image_index=None, filename=None):
    """Build a `Reference` from one catalogue row and its placement spec.

    The spec wins over the catalogue for every field, since it is the caller
    correcting a row they can see is wrong. Dimensions given to the composite in
    metres are reused here rather than re-read, so the prompt cannot disagree
    with the geometry about how big a thing is.
    """
    spec = spec or {}
    metadata = metadata or {}

    def _cm(spec_metres_key, spec_cm_key, metadata_key):
        if spec.get(spec_metres_key) is not None:
            return float(spec[spec_metres_key]) * 100.0
        if spec.get(spec_cm_key) is not None:
            return float(spec[spec_cm_key])
        value = metadata.get(metadata_key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    name = spec.get("name") or metadata.get("name") or ""
    return Reference(
        filename=filename or _filename_for(metadata),
        furniture_type=resolve_type(spec.get("type"), metadata, name),
        name=name,
        width_cm=_cm("width_m", "width_cm", "widthCm"),
        height_cm=_cm("height_m", "height_cm", "heightCm"),
        depth_cm=_cm("depth_m", "depth_cm", "depthCm"),
        image_index=image_index,
    )


def _filename_for(metadata):
    """A stable name the prompt can point at.

    The catalogue does not necessarily store one, but the prompt set is built
    around naming references by filename, so one is derived from the id and the
    product name rather than referring to "the second image".
    """
    stored = metadata.get("filename") or metadata.get("fileName")
    if stored:
        return str(stored)
    slug = _slugify(metadata.get("name") or "reference") or "reference"
    return f"{slug}_{metadata.get('id', 'x')}.png"

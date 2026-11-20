"""Build the FLUX.2 prompt for one generation request.

Written for how FLUX.2 reads a prompt: it weighs the start most, it has no
negative prompt (a "do not X" still puts X in the prompt), and it refers to
inputs as "image 1", "image 2", ... So the prompt states what the picture
should show, in plain positive sentences, most important first:

    1. the task and one line per product (`intro_block`), naming each
       reference by its image number;
    2. the layout, as sentences generated from the placement plan
       (`layout_block`), which follows the living-room rule set in
       placement_plan.py;
    3. `DIRECTIVES`, the short standing instructions for every request;
    4. the user's own request, as mood and styling;
    5. the room's approximate size, one line.

The caller (generate_routes) puts replaced-furniture and fixed-position notes
between 1 and 2 when a request has them.

The furniture type is resolved generously: an explicit `type` on the placement
wins, then a `type`/`category` field on the catalogue metadata, then keyword
matching against the product name. Names in this catalogue are German
(Hoeffner), so the aliases carry both languages. The type drives the layout
rules and the wording here.

WHAT THIS MODULE DOES NOT DO
    It does not decide which images are sent to FLUX. It labels each reference by
    the position it will occupy in the request (`image 2`, `image 3`, ...), and the
    caller is responsible for the two agreeing.
"""

import re
from dataclasses import dataclass, field

# Standing directives sent with every request: what only the model decides -
# light, contact with the floor, and that each product stays itself. Every
# sentence describes the wanted result; FLUX.2 has no negative prompt.
DIRECTIVES = """Keep the room exactly as photographed: the same walls, floor, ceiling, windows, doors, mouldings, outlets and existing light fixtures, and the same camera position, height, angle and framing.

Light every new piece with the room's own daylight, matching the direction, softness, colour temperature and exposure of the shadows already in the room; the side of each piece facing the window is the brighter one. Floor-standing pieces rest on the floor with soft contact shadows, wall-mounted pieces sit on their wall at the planned height, and every edge blends cleanly into the room.

Each product matches its reference image exactly in shape, configuration, colour, material, finish and proportions, at its stated size.

The result is one natural, photorealistic interior photograph of this room."""

# Kept for callers that imported the old name.
ESSENTIALS = DIRECTIVES

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
            aliases=("sideboard", "credenza", "buffet", "anrichte", "highboard"),
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
            slug="lowboard",
            label="lowboard",
            # A lowboard is not a sideboard: it is the low media cabinet, and
            # the English "TV unit" names the same piece, so both land here.
            aliases=("lowboard", "tv unit", "tv stand", "media console", "tv-board", "tv board"),
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
            generic="generic black lowboard",
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


def _haystacks(*parts):
    """Lowercased text to keyword-match, in a spaced and a squashed form.

    The squashed form is what lets `coffeetable_2` match the alias "coffee
    table": a filename drops the space the alias carries, so matching only the
    spaced form misses every compounded name.
    """
    text = " ".join(str(part) for part in parts if part).lower()
    spaced = re.sub(r"[^a-z0-9äöüß]+", " ", text).strip()
    return spaced, spaced.replace(" ", "")


def _stem(filename):
    """`armchair_2.jpg` -> `armchair_2`, so the extension cannot match an alias."""
    if not filename:
        return ""
    return str(filename).rsplit("/", 1)[-1].rsplit(".", 1)[0]


def resolve_type(explicit=None, metadata=None, name=None, filename=None):
    """Pick the furniture type, most trustworthy source first.

    An explicit slug on the placement beats the catalogue, which beats guessing
    from the product name. An unrecognised explicit slug is an error worth
    raising rather than silently falling back, since it means the caller asked
    for a checklist that does not exist.

    The filename is matched too, and last. A catalogue that stores no `type`
    and an autogenerated `name` often still has a hand-named file - sofa_1.jpg,
    highboard_2.jpg - and that is the only furniture word in the row.
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

    metadata = metadata or {}
    filename = filename or metadata.get("filename") or metadata.get("fileName")
    spaced, squashed = _haystacks(name, metadata.get("name"), _stem(filename))
    for alias, furniture_type in _ALIAS_INDEX:
        if alias in spaced or alias.replace(" ", "") in squashed:
            return furniture_type
    return GENERIC_TYPE


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# one reference
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    """One furniture reference as the prompt needs to talk about it.

    `furniture_type.attributes` and `.generic` are no longer sent: a checklist
    of attribute names asks for a reasoning step FLUX does not have, and naming
    the generic substitute puts it into the prompt. The product line states
    what is known instead.
    """

    filename: str
    furniture_type: FurnitureType
    name: str = ""
    width_cm: float = None
    height_cm: float = None
    depth_cm: float = None
    image_index: int = None  # 1-based position in the FLUX reference list
    catalogue_id: object = None
    chaise_side: str = None  # "left" / "right", seen facing the sofa
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

    def product_line(self):
        """`Image 2: Ecksofa Lina, a sofa, 280 × 85 × 170 cm (W × H × D).`"""
        where = f"Image {self.image_index}" if self.image_index else self.filename
        kind = self.furniture_type.label
        article = "an" if kind[:1] in "aeiou" else "a"
        text = (f"{where}: {self.name}, {article} {kind}" if self.name
                else f"{where}: {article} {kind}")
        if self._known_dimensions():
            text += f", {self.dimension_text} (W × H × D)"
        if self.chaise_side:
            text += f", L-shaped with the chaise on the {self.chaise_side} as seen from the front"
        text += "."
        if self._note:
            text += " " + self._note
        return text

    def _known_dimensions(self):
        return any(v for v in (self.width_cm, self.height_cm, self.depth_cm))


def intro_block(references, uploads=0):
    """The task, and one line per product. Goes first: FLUX weighs it most."""
    lines = [
        "# TASK",
        "Image 1 is this room with the new furniture pasted in as flat placeholders at "
        "their planned positions. Turn it into a photograph of the same room with these "
        "exact products standing there:",
    ]
    lines += [ref.product_line() for ref in references]
    lines.append(
        "The placeholders in image 1 give position, orientation and size; the product "
        "images give shape, configuration, colour, materials and details. Each product "
        "keeps its stated dimensions."
    )
    if uploads:
        first = len(references) + 2
        last = first + uploads - 1
        span = f"Image {first}" if first == last else f"Images {first} to {last}"
        lines.append(f"{span}: extra style references from the user, for mood and materials.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the layout, from the placement plan
# --------------------------------------------------------------------------- #
def _reference_for(entry, references):
    label = entry.get("label") or ""
    for ref in references:
        if ref.catalogue_id is not None and f"(id {ref.catalogue_id})" in label:
            return ref
    return None


def _cm(metres):
    return int(round(float(metres) * 100 / 5.0) * 5)


def _layout_sentence(entry, ref, sofa_l_shaped):
    relation = entry.get("relation") or {}
    rule = relation.get("rule")
    kind = (ref.furniture_type.label if ref else None) or relation.get("type") or "piece"
    name = f"The {kind} (image {ref.image_index})" if ref and ref.image_index else f"The {kind}"
    wall = relation.get("wall_side") or "wall"
    side = relation.get("picture_side")
    seen = f" on its {side} side in the picture" if side else ""
    distance = relation.get("distance_m")
    gap = f"about {_cm(distance)} cm" if distance else "a short distance"

    if rule == "opposite_sofa":
        return f"{name} stands against the {wall}, directly opposite the sofa, as the media wall."
    if rule == "next_to_media":
        return f"{name} stands against the {wall}, right next to the lowboard."
    if rule == "in_front_of_sofa" and kind == "coffee table" and sofa_l_shaped:
        return (f"{name} stands in the inner corner of the L-shaped sofa, {gap} from the "
                "front of the seat and from the chaise.")
    if rule == "in_front_of_sofa":
        return f"{name} stands centred in front of the sofa, {gap} from its front edge."
    if rule == "beside_sofa":
        return f"{name} stands right beside the sofa's armrest{seen}."
    if rule == "beside_coffee_table":
        return f"{name} stands in front of the sofa, next to the coffee table{seen}."
    if rule == "near_sofa":
        return (f"{name} stands close to the sofa{seen}, turned toward the coffee table "
                "so the seating forms one conversation group.")
    if rule == "against_wall":
        height = entry.get("mount_height_m") or 0
        if height > .05:
            return f"{name} hangs on the {wall} at {_cm(height)} cm above the floor."
        text = f"{name} stands against the {wall}"
        wall_gap = relation.get("wall_gap_m") or 0
        if kind == "sofa" and wall_gap >= .15:
            text += f", {_cm(wall_gap)} cm in front of it"
        if relation.get("chaise_side"):
            text += f", with its chaise on the {relation['chaise_side']} as seen from the front"
        return text + "."
    return None


def layout_block(plan, references):
    """The placement plan in words FLUX can relate to the picture.

    Metres in camera space mean nothing to the model; "the coffee table stands
    centred in front of the sofa, about 45 cm from it" does. Pieces the user
    placed by hand (no relation) are stated separately as fixed positions.
    """
    entries = [e for e in plan or () if e.get("relation")]
    if not entries:
        return ""
    order = {"sofa": 0, "lowboard": 1, "coffee_table": 2, "side_table": 3, "ottoman": 3,
             "armchair": 4}
    entries.sort(key=lambda e: order.get((e.get("relation") or {}).get("type"), 9))
    sofa_l_shaped = any((e["relation"].get("type") == "sofa" and e["relation"].get("chaise_side"))
                        for e in entries)
    sentences = []
    for entry in entries:
        sentence = _layout_sentence(entry, _reference_for(entry, references), sofa_l_shaped)
        if sentence:
            sentences.append(sentence)
    types = {e["relation"].get("type") for e in entries}
    if {"sofa", "lowboard"} <= types:
        sentences.append("The view from the sofa to the media wall stays open.")
    sentences.append("Open floor stays visible between the pieces as a clear walkway, "
                     "and every door can open freely.")
    return "# LAYOUT\n" + "\n".join(sentences)


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
    """One line of approximate room size, or "" when nothing was measured."""
    if geometry is None:
        return ""

    dimensions = getattr(geometry, "room_dimensions", {})
    if dimensions:
        parts = []
        for key, word in (("width", "wide"), ("length", "long"), ("height", "high")):
            estimate = dimensions.get(key, {})
            value = estimate.get("value_m")
            if estimate.get("status") == "estimated" and value is not None:
                parts.append(f"{value:.1f} m {word}")
        if not parts:
            return ""
        return ("The room is roughly " + ", ".join(parts) + f"; the camera is "
                f"{geometry.camera_height_m:.2f} m above the floor. Use this only for scale.")

    try:
        metrics = room_metrics(geometry)
    except (ValueError, ZeroDivisionError):
        # A floor that could not be measured is not a reason to fail the whole
        # generation; the model still has the photograph.
        return ""
    return (f"The visible floor is about {metrics['back_wall_width_m']:.1f} m wide along the "
            f"far wall and {metrics['depth_to_back_wall_m']:.1f} m deep; the camera is "
            f"{metrics['camera_height_m']:.2f} m above the floor. Use this only for scale.")


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_prompt(
    user_prompt,
    references,
    geometry=None,
    scene_name="room.png",
    include_essentials=True,
    plan=None,
):
    """Layout, directives, the user's request and the room size, in that order.

    The task and product lines (`intro_block`) go before this, and the caller
    adds replaced-furniture and fixed-position notes between the two, so the
    order FLUX reads is: task and products, special notes, layout, directives,
    user request, room size. FLUX.2 weighs the start most.
    """
    sections = []

    layout = layout_block(plan, references)
    if layout:
        sections.append(layout)

    if include_essentials:
        sections.append("# LOOK\n" + DIRECTIVES)

    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        sections.append(
            "# STYLE REQUEST\n" + user_prompt + "\nThis sets mood and styling; the room, "
            "the camera, the layout above and the products stay as described."
        )

    room = room_block(geometry, scene_name)
    if room:
        sections.append("# ROOM SCALE\n" + room)

    return "\n\n".join(sections)


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
    filename = filename or _filename_for(metadata)
    return Reference(
        filename=filename,
        furniture_type=resolve_type(spec.get("type"), metadata, name, filename),
        name=name,
        width_cm=_cm("width_m", "width_cm", "widthCm"),
        height_cm=_cm("height_m", "height_cm", "heightCm"),
        depth_cm=_cm("depth_m", "depth_cm", "depthCm"),
        image_index=image_index,
        catalogue_id=metadata.get("id"),
        chaise_side=spec.get("chaise_side"),
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

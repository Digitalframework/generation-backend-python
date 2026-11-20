"""The living-room rule set, as the placement planner applies it.

A synthetic room: camera at the origin looking down +z, floor 1.4 m below,
a left and a right wall 4 m apart facing each other, and a back wall 8.5 m away.
The camera stands off-centre: a flat cutout whose front is edge-on to the
camera cannot be drawn, so a perfectly centred camera hides placements.
"""

import numpy as np
import pytest
from PIL import Image

from app.composite import Placement
from app.placement_plan import (PlacementConflict, Wall, plan_placements,
                                SIDE_PASSAGE_M)
from app.vggt_scene import SceneGeometry

UP = np.array([0., -1., 0.])


def wall(name, origin, inward, start, end):
    inward = np.asarray(inward, float)
    return Wall(name, np.asarray(origin, float), np.cross(UP, inward), inward, start, end)


def room(walls=None):
    walls = walls or [
        wall("wall_1", [-2.5, 1.4, 0.], [1., 0., 0.], 1.5, 8.5),  # left
        wall("wall_2", [1.5, 1.4, 0.], [-1., 0., 0.], -8.5, -1.5),  # right
        wall("wall_3", [0., 1.4, 8.5], [0., 0., -1.], -2.5, 1.5),  # back
    ]
    K = np.array([[300., 0., 400.], [0., 300., 300.], [0., 0., 1.]])
    return SceneGeometry(K, UP, 1.4, 1.4, (800, 600), np.full(800, 400.), walls=walls)


def piece(kind, width, depth, height=.8):
    return Placement(Image.new("RGBA", (10, 10), "red"), 0., width, height,
                     depth_size_m=depth, kind=kind, label=kind)


def rear(item):
    """Rear centre of a planned piece on the floor."""
    facing = np.cross(item.right_axis, UP)
    return item.base_m - facing * item.depth_size_m, facing


def test_living_room_follows_the_rule_set():
    sofa = piece("sofa", 2.2, .9, .85)
    lowboard = piece("lowboard", 1.8, .45, .5)
    coffee = piece("coffee_table", 1.0, .6, .4)
    side = piece("side_table", .45, .45, .55)
    ottoman = piece("ottoman", .6, .6, .42)
    chair = piece("armchair", .8, .8, .9)
    sideboard = piece("sideboard", 1.6, .45, .85)
    items = [sideboard, chair, ottoman, side, coffee, lowboard, sofa]  # any order
    plan_placements(items, [], room())

    # S1 + M1: sofa on a wall, the media wall directly opposite.
    assert sofa.relation["rule"] == "against_wall"
    assert lowboard.relation["rule"] == "opposite_sofa"
    assert {sofa.wall_id, lowboard.wall_id} == {"wall_1", "wall_2"}

    # C1: coffee table 45 cm in front of the sofa, free-standing.
    assert coffee.wall_id is None and coffee.relation["rule"] == "in_front_of_sofa"
    sofa_rear, facing = rear(sofa)
    coffee_rear, _ = rear(coffee)
    assert np.isclose((coffee_rear - sofa.base_m) @ facing, .45, atol=.01)

    # T1: side table right at an armrest, rear aligned with the sofa.
    assert side.relation["rule"] == "beside_sofa"
    side_rear, _ = rear(side)
    assert np.isclose(abs((side_rear - sofa_rear) @ sofa.right_axis),
                      2.2 / 2 + .05 + .45 / 2, atol=.01)

    # O1/O2: ottoman in front of the sofa, beside the coffee table.
    assert ottoman.relation["rule"] == "beside_coffee_table"
    ottoman_rear, _ = rear(ottoman)
    assert np.isclose((ottoman_rear - sofa.base_m) @ facing, .45, atol=.01)

    # A1/A2: armchair near the sofa, turned toward the seating group.
    assert chair.relation["rule"] == "near_sofa" and chair.wall_id is None

    # B2: sideboard on the media wall, next to the lowboard.
    assert sideboard.relation["rule"] == "next_to_media"
    assert sideboard.wall_id == lowboard.wall_id

    for item in items:
        assert item.relation["type"] == item.kind


def test_l_sofa_keeps_the_inner_corner_for_the_coffee_table():
    sofa = piece("sofa", 2.8, 1.6, .85)
    coffee = piece("coffee_table", .9, .6, .4)
    plan_placements([sofa, coffee], [{"chaise_side": "right", "seat_depth_m": .95}, {}], room())
    sofa_rear, facing = rear(sofa)
    coffee_rear, _ = rear(coffee)
    # C2: ~45 cm from the main body's front ...
    assert np.isclose((coffee_rear - sofa_rear) @ facing, .95 + .45, atol=.01)
    # ... and from the chaise's inner side (chaise on the sofa's +right end).
    chaise_inner = 2.8 / 2 - .9
    coffee_edge = (coffee_rear - sofa_rear) @ sofa.right_axis + .9 / 2
    assert np.isclose(chaise_inner - coffee_edge, .45, atol=.01)
    assert sofa.relation["chaise_side"] == "right"


def test_without_a_sofa_a_coffee_table_goes_on_a_wall_as_before():
    coffee = piece("coffee_table", 1.0, .6, .4)
    plan_placements([coffee], [], room())
    assert coffee.wall_id is not None


def test_explicit_wall_overrides_the_type_rule():
    sofa, chair = piece("sofa", 2.2, .9), piece("armchair", .8, .8, .9)
    plan_placements([sofa, chair], [{}, {"wall_id": "wall_3"}], room())
    assert chair.wall_id == "wall_3"


def test_sofa_wall_gap_is_capped_at_50_cm():
    sofa = piece("sofa", 2.2, .9)
    plan_placements([sofa], [{"wall_gap_m": 1.2}], room())
    assert sofa.relation["wall_gap_m"] == .5


def test_walkway_in_front_of_the_lowboard_stays_clear():
    # A narrow room: sofa, 45 cm, coffee table and the lowboard's 80 cm zone
    # cannot all fit, so the conflict is reported instead of a blocked path.
    narrow = room([
        wall("wall_1", [-1.5, 1.4, 0.], [1., 0., 0.], 2., 7.),
        wall("wall_2", [.9, 1.4, 0.], [-1., 0., 0.], -7., -2.),
    ])
    items = [piece("sofa", 2.0, .9), piece("lowboard", 1.6, .45, .5),
             piece("coffee_table", 1.0, .6, .4)]
    with pytest.raises(PlacementConflict) as error:
        plan_placements(items, [], narrow)
    assert error.value.issues[0]["type"] == "coffee_table"


def test_optional_pieces_are_flagged_to_leave_out_first():
    items = [piece("sofa", 2.2, .9), piece("armchair", 3.5, 3.5, .9)]
    with pytest.raises(PlacementConflict) as error:
        plan_placements(items, [], room())
    issue = error.value.issues[0]
    assert issue["optional"] is True
    assert issue["options"][0] == "exclude_from_this_generation"


def test_free_standing_pieces_keep_a_side_passage():
    assert SIDE_PASSAGE_M >= .6


# --------------------------------------------------------------------------- #
# what FLUX is told about the layout
# --------------------------------------------------------------------------- #
def test_prompt_states_the_layout_in_words_and_stays_positive():
    from app.composite import compose
    from app.prompt_builder import build_prompt, intro_block, reference_from_catalogue

    rows = [({"id": 1, "name": "Sofa A", "widthCm": 220, "heightCm": 85, "depthCm": 90}, {}),
            ({"id": 2, "name": "Lowboard B", "widthCm": 180, "heightCm": 50, "depthCm": 45}, {}),
            ({"id": 3, "name": "Couchtisch C", "widthCm": 100, "heightCm": 40, "depthCm": 60}, {}),
            ({"id": 4, "name": "Sessel D", "widthCm": 80, "heightCm": 90, "depthCm": 80}, {})]
    items = []
    for metadata, _spec in rows:
        ref = reference_from_catalogue(metadata)
        item = piece(ref.furniture_type.slug, metadata["widthCm"] / 100,
                     metadata["depthCm"] / 100, metadata["heightCm"] / 100)
        item.label = f"{metadata['name']} (id {metadata['id']})"
        items.append(item)
    geometry = room()
    plan_placements(items, [], geometry)
    plan = compose(Image.new("RGB", geometry.size, "white"), items, geometry).placement_plan
    refs = [reference_from_catalogue(m, s, image_index=i + 2) for i, (m, s) in enumerate(rows)]
    text = intro_block(refs) + "\n\n" + build_prompt("cosy", refs, geometry=geometry, plan=plan)

    assert text.startswith("# TASK")
    assert "Image 2: Sofa A, a sofa, 220 × 85 × 90 cm" in text
    assert "Image 5: Sessel D, an armchair" in text
    assert "The lowboard (image 3) stands against the" in text
    assert "directly opposite the sofa" in text
    assert "The coffee table (image 4) stands centred in front of the sofa, about 45 cm" in text
    assert text.index("# LAYOUT") < text.index("# LOOK") < text.index("# STYLE REQUEST")
    # FLUX.2 has no negative prompt: nothing phrased as a prohibition.
    lowered = text.lower()
    for phrase in ("do not", "don't", " no ", "never", "not a "):
        assert phrase not in lowered

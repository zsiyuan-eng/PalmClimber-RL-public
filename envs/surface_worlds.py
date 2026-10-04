"""Generate camera-facing obstacle bands with room for the six-wheel footprint."""
from __future__ import annotations

import math


V8_TIRE_ANGULAR_INFLATION = .24
V8_TIRE_VERTICAL_INFLATION = .037
V8_TIRE_CENTER_RADIUS = .182
V8_PASSIVE_TILT_RESERVE = .205
V8_ADDITIONAL_YAW_RESERVE = .025
V8_ALIAS_PERIOD = math.pi / 3.
V8_WIDTH_TIERS = (.045, .060, .080, .105, .125, .140)


def _wrap(value):
    return (float(value) + math.pi) % (2. * math.pi) - math.pi


def certify_visible_navigation_world(world, *, front_azimuth=-math.pi / 4.):
    """External construction-only proof of separated bands and clear yaw.

    The certificate is conservative within the declared .205 rad base tilt
    reserve and .24/.037 tire extents.  It is a geometric route-existence
    check, not a promise that the learned controller will discover that route.
    """
    patches = world['terrain_map']
    angular_margin = V8_TIRE_ANGULAR_INFLATION + V8_ADDITIONAL_YAW_RESERVE
    vertical_margin = (V8_TIRE_VERTICAL_INFLATION
                       + V8_TIRE_CENTER_RADIUS * math.sin(V8_PASSIVE_TILT_RESERVE))
    ordered = sorted(range(len(patches)), key=lambda index: patches[index]['height'])
    # Distinct patches within a band are separated by .10 m nominally;
    # bands are at least .35 m apart.  A .16 m clustering threshold keeps
    # nominal pairs together without borrowing a clear gap from another band.
    groups = []
    for index in ordered:
        if not groups or patches[index]['height'] - patches[groups[-1][-1]]['height'] > .16:
            groups.append([])
        groups[-1].append(index)
    rows = []
    for indices in groups:
        reference = patches[indices[0]]['theta']
        offsets = [_wrap(patches[i]['theta'] - reference) for i in indices]
        lower = min(offset - patches[i]['theta_half_width'] - angular_margin
                    for i, offset in zip(indices, offsets))
        upper = max(offset + patches[i]['theta_half_width'] + angular_margin
                    for i, offset in zip(indices, offsets))
        free_width = V8_ALIAS_PERIOD - (upper - lower)
        if free_width <= .04:
            raise ValueError('Native full-tire band does not leave a usable angular corridor')
        free_center = reference + (upper + lower + V8_ALIAS_PERIOD) / 2.
        bottom = min(patches[i]['height'] - patches[i]['height_half_width'] - vertical_margin
                     for i in indices)
        top = max(patches[i]['height'] + patches[i]['height_half_width'] + vertical_margin
                  for i in indices)
        rows.append(dict(patch_indices=indices, nominal_height_min=min(patches[i]['height'] for i in indices),
                         nominal_height_max=max(patches[i]['height'] for i in indices),
                         conservative_full_tire_bottom=bottom, conservative_full_tire_top=top,
                         safe_configuration_azimuth=_wrap(free_center),
                         safe_channel_half_width=free_width / 2.))
    gaps = []
    for previous, following in zip(rows, rows[1:]):
        gap = following['conservative_full_tire_bottom'] - previous['conservative_full_tire_top']
        if gap <= .045:
            raise ValueError('Native full-tire bands leave insufficient vertical space to change yaw')
        gaps.append(gap)
    if rows:
        if world['start_height'] >= rows[0]['conservative_full_tire_bottom'] - .045:
            raise ValueError('Initial pose starts inside the conservative full-tire band')
        if world['goal_height'] <= rows[-1]['conservative_full_tire_top'] + .045:
            raise ValueError('Task height does not clear the last conservative full-tire band')
    visibility = []
    for index, patch in enumerate(patches):
        # The full physical patch, not merely its centre, must lie on the
        # camera-facing hemisphere.  These values describe static geometry.
        maximum_front_offset = abs(_wrap(patch['theta'] - front_azimuth)) + patch['theta_half_width']
        if maximum_front_offset >= math.pi / 2.:
            raise ValueError('Declared visible obstacle extends onto the far hemisphere')
        visibility.append(dict(patch_index=index, theta=patch['theta'],
                               true_theta_width=2. * patch['theta_half_width'],
                               true_height_width=2. * patch['height_half_width'],
                               minimum_camera_facing_cosine=math.cos(maximum_front_offset)))
    return dict(schema='visible_front_native_tire_world_v8', privileged_construction_only=True,
                actor_or_mpc_map_access=False, contact_geometry_semantics='native_tire_against_static_lip_facets',
                alias_period=V8_ALIAS_PERIOD, front_azimuth=float(front_azimuth),
                full_tire_angular_inflation=V8_TIRE_ANGULAR_INFLATION,
                full_tire_vertical_inflation=V8_TIRE_VERTICAL_INFLATION,
                additional_yaw_reserve=V8_ADDITIONAL_YAW_RESERVE,
                tire_center_radius=V8_TIRE_CENTER_RADIUS, passive_tilt_reserve=V8_PASSIVE_TILT_RESERVE,
                rows=rows, clear_inter_band_vertical_gaps=gaps, visible_patches=visibility,
                minimum_safe_channel_half_width=min((r['safe_channel_half_width'] for r in rows), default=None),
                geometric_feasibility_only=True, learned_success_not_assumed=True)


def construct_visible_navigation_world(rng, *, difficulty=3, empty_probability=.10,
                                       encounter_probability=.85, minimum_obstacles=3,
                                       maximum_obstacles=6, front_azimuth=-math.pi / 4.):
    """Sample 3--6 varied-width front obstacles in three traversable bands.

    Curriculum zero is empty, one has a single narrow obstacle, two has
    two height bands, and three uses all three bands.  The actor decides
    every movement from causal sensors.  Randomness here constructs the
    native plant before the episode and is never fed into that decision.
    """
    if isinstance(difficulty, bool) or int(difficulty) != difficulty or not 0 <= difficulty <= 3:
        raise ValueError('difficulty must be an integer from zero to three')
    if not 0. <= empty_probability <= 1. or not 0. <= encounter_probability <= 1.:
        raise ValueError('world probabilities must be between zero and one')
    if not 1 <= minimum_obstacles <= maximum_obstacles <= 6:
        raise ValueError('visible-front obstacle count must be between one and six')
    if not math.isfinite(front_azimuth):
        raise ValueError('front azimuth must be finite')
    start = float(rng.uniform(.47, .51))
    goal = float(rng.uniform(1.72, 1.78))
    start_theta = float(rng.uniform(-.08, .08))
    goal_theta = start_theta + float(rng.uniform(-.025, .025))
    terrain = []
    if difficulty and float(rng.random()) >= empty_probability:
        if difficulty == 1:
            count, band_count = 1, 1
        elif difficulty == 2:
            count, band_count = int(rng.integers(2, 5)), 2
        else:
            count = int(rng.integers(minimum_obstacles, maximum_obstacles + 1))
            band_count = min(3, count)
        allocation = [count // band_count + int(band < count % band_count) for band in range(band_count)]
        if band_count == 1:
            centers = [.86 + float(rng.uniform(-.02, .02))]
        elif band_count == 2:
            centers = [.81, 1.30]
        else:
            centers = [.73, 1.115, 1.50]
        # Wheel six contacts the front at zero chassis azimuth.  This
        # chosen physical sector stays visible across the three row phases.
        encounter = float(rng.random()) < encounter_probability
        phase_shift = float(rng.uniform(-.025, .025))
        if encounter:
            first_phase = start_theta + float(rng.uniform(-.08, .08))
        else:
            first_phase = .36 + float(rng.uniform(-.03, .03))
        physical_first = front_azimuth + (first_phase - (front_azimuth + V8_ALIAS_PERIOD))
        physical_first = max(front_azimuth - .30, min(front_azimuth + .24, physical_first))
        physical_phases = [physical_first,
                           front_azimuth - .12 + phase_shift,
                           front_azimuth + .19 - phase_shift]
        if difficulty == 1:
            widths = [float(rng.choice(V8_WIDTH_TIERS[:3]))]
        else:
            widths = list(map(float, rng.choice(V8_WIDTH_TIERS, size=count, replace=False)))
        # Dense worlds always contain a wide real lip, not just a wide
        # presentation outline.  Its size remains inside the route proof.
        if difficulty == 3 and count >= 3 and max(widths) < .125:
            widths[-1] = .140
        cursor = 0
        for band, (center, number) in enumerate(zip(centers, allocation)):
            center += float(rng.uniform(-.006, .006))
            offsets = [-.05, .05] if number == 2 else [0.]
            for height_offset in offsets:
                terrain.append(dict(kind='blocked', theta=_wrap(physical_phases[band] + float(rng.uniform(-.008, .008))),
                                    height=center + height_offset, theta_half_width=widths[cursor],
                                    height_half_width=float(rng.uniform(.020, .026)), support_fraction=0.))
                cursor += 1
    world = dict(start_height=start, start_azimuth=start_theta, goal_height=goal, goal_azimuth=goal_theta,
                 terrain_map=terrain, payload_mass=float(rng.uniform(.025, .040)), target_radius=.60,
                 friction_range=(.60, .70), sway_amplitude=0.)
    return world, certify_visible_navigation_world(world, front_azimuth=front_azimuth)

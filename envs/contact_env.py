"""Native MuJoCo tire and roller collisions with raised bark lips.

The MPC nominal model excludes the obstacle facets. Tree traction uses a
friction-limited elastic brush model; lip collisions use native contact forces."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from envs.common import MODEL_PATH, asset_contract, asset_directory, load_model
from envs.tree_work_env import TreeWorkEnv


@dataclass(frozen=True)
class NativeLipContactConfig:
    lip_depth: float = .010
    angular_facet_max_width: float = .04
    max_patches: int = 32
    solver_time_constant: float = .006
    solver_damping_ratio: float = 1.

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = cls(**asdict(value))
        elif isinstance(value, dict):
            result = cls(**value)
        else:
            raise TypeError("contact_config must be a dict or NativeLipContactConfig")
        for key in ("lip_depth", "angular_facet_max_width", "solver_time_constant", "solver_damping_ratio"):
            x = getattr(result, key)
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number)) or not np.isfinite(x) or x <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not .002 <= result.lip_depth <= .03 or not .005 <= result.angular_facet_max_width <= .1:
            raise ValueError("Lip depth/facet width is outside the supported geometry range")
        if isinstance(result.max_patches, bool) or not isinstance(result.max_patches, int) or not 1 <= result.max_patches <= 32:
            raise ValueError("max_patches must be an integer in [1,32]")
        if result.solver_time_constant < .004:
            raise ValueError("Contact time constant must be at least two native timesteps")
        return result


def lip_facets(patches, tree_radius, contact_config):
    """Return the same exact native geometry used for contact and display.

    Facets form a tangent polygon around the tree. Outer-face angular end
    points coincide with the declared patch bounds; the inner 2 mm is buried
    in the bark. This polygon, rather than an ideal painted rectangle, is the
    physical obstacle. All four edges remain collidable.
    """
    output = []
    c = NativeLipContactConfig.from_value(contact_config)
    for patch_index, patch in enumerate(patches):
        if patch["kind"] != "blocked":
            continue
        count = max(1, int(np.ceil(2*patch["theta_half_width"]/c.angular_facet_max_width)))
        width = 2*patch["theta_half_width"]/count
        for facet_index in range(count):
            theta = patch["theta"]-patch["theta_half_width"]+(facet_index+.5)*width
            radius = tree_radius+(c.lip_depth-.002)/2
            output.append(dict(
                name=f"v8_lip_patch_{patch_index:02d}_facet_{facet_index:03d}",
                patch_index=patch_index, facet_index=facet_index, theta=float(theta),
                pos=[radius*np.cos(theta), radius*np.sin(theta), patch["height"]],
                quat=[np.cos(theta/2), 0., 0., np.sin(theta/2)],
                half_size=[(c.lip_depth+.002)/2,
                           (tree_radius+c.lip_depth)*np.tan(width/2), patch["height_half_width"]],
                angular_width=float(width), lip_depth=c.lip_depth))
    return output


class TreeWorkContactEnv(TreeWorkEnv):
    """Original single-ring robot with opt-in finite-volume lip collisions."""

    def __init__(self, task="combined", controller_mode="mpc", config=None,
                 render_mode=None, contact_config=None):
        self.contact_config = NativeLipContactConfig.from_value(contact_config)
        self._v8_lip_geom_ids = np.zeros(0, dtype=int)
        self._v8_lip_geom_patch_indices = np.zeros(0, dtype=int)
        self._v8_tire_geom_ids = np.zeros(0, dtype=int)
        self._v8_contact_geometry = []
        self._v8_geom_to_wheel = {}
        self._v8_geom_to_patch = {}
        super().__init__(task=task, controller_mode=controller_mode, config=config, render_mode=render_mode)
        if not self.config.sensor_only_navigation:
            raise ValueError("native contacts require sensor_only_navigation=True")

    def _compile_lips(self, patches):
        geometry = lip_facets(patches, self.config.tree_radius, self.contact_config)
        if not geometry:
            # Preserve the unmodified original compiler path in empty worlds.
            model = load_model()
        else:
            asset_contract()
            root = ET.parse(MODEL_PATH).getroot()
            root.find("compiler").set("meshdir", str(asset_directory()))
            for wheel_index in range(1, 7):
                body = next(b for b in root.iter("body") if b.get("name") == f"wheel{wheel_index}")
                geoms = body.findall("geom")
                # Main visible tire plus its ten visible capsule rollers.
                # The hub and yellow rotation marks are internal/decorative.
                for geom_index in [0, *range(2, 12)]:
                    geoms[geom_index].set("contype", "4")
                    geoms[geom_index].set("conaffinity", "0")
            world = root.find("worldbody")
            for facet in geometry:
                attrs = {key: " ".join(f"{x:.17g}" for x in facet[key])
                         for key in ("pos", "quat")}
                attrs["size"] = " ".join(f"{x:.17g}" for x in facet["half_size"])
                ET.SubElement(world, "geom", name=facet["name"], type="box", **attrs,
                    contype="0", conaffinity="4", group="0", rgba=".47 .24 .09 1",
                    friction="0.7 0.005 0.0001", margin="0", gap="0",
                    solref=f"{self.contact_config.solver_time_constant:.17g} {self.contact_config.solver_damping_ratio:.17g}",
                    solimp=".98 .995 .001")
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        # Static world geoms must not change any physical robot address,
        # inertia or actuator. The separate nominal model is never rebuilt.
        for kind, count_key in ((mujoco.mjtObj.mjOBJ_BODY, "nbody"), (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
                                (mujoco.mjtObj.mjOBJ_SITE, "nsite"), (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu")):
            if getattr(model, count_key) != getattr(self.nominal_model, count_key):
                raise RuntimeError("contact geometry changed the original robot topology")
            for index in range(getattr(model, count_key)):
                if mujoco.mj_id2name(model, kind, index) != mujoco.mj_id2name(self.nominal_model, kind, index):
                    raise RuntimeError("contact geometry changed the original robot indexing")
        for key in ("body_mass", "body_inertia", "jnt_qposadr", "jnt_dofadr", "jnt_range",
                    "dof_armature", "dof_damping", "actuator_ctrlrange", "actuator_forcerange"):
            if not np.array_equal(getattr(model, key), getattr(self.nominal_model, key)):
                raise RuntimeError(f"contact geometry changed original robot {key}")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.model, self.data = model, mujoco.MjData(model)
        self._v8_contact_geometry = geometry
        self._v8_lip_geom_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_GEOM, x["name"]) for x in geometry], dtype=int)
        self._v8_lip_geom_patch_indices = np.array([x["patch_index"] for x in geometry], dtype=int)
        self._v8_geom_to_patch = dict(zip(self._v8_lip_geom_ids.tolist(), self._v8_lip_geom_patch_indices.tolist()))
        self._v8_geom_to_wheel = {}
        if geometry:
            for i in range(6):
                body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, f"wheel{i+1}")
                for geom_id in np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_contype == 4)):
                    self._v8_geom_to_wheel[int(geom_id)] = i
        self._v8_tire_geom_ids = np.array(list(self._v8_geom_to_wheel), dtype=int)

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else deepcopy(options)
        if "terrain_map" not in options:
            # A deterministic preliminary reset samples the original default
            # scene. The final reset uses the same seed and random draw order.
            super().reset(seed=seed, options=options)
            options["terrain_map"] = deepcopy(self.terrain_map)
        patches = self._validate_map(options["terrain_map"], max_patches=self.contact_config.max_patches)
        self._compile_lips(patches)
        return super().reset(seed=seed, options=options)

    def _native_lip_contacts(self):
        contacts = []
        wheel_force = np.zeros(6)
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self._v8_geom_to_wheel and geom2 in self._v8_geom_to_patch:
                tire, lip, sign = geom1, geom2, -1.
            elif geom2 in self._v8_geom_to_wheel and geom1 in self._v8_geom_to_patch:
                tire, lip, sign = geom2, geom1, 1.
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            wheel = self._v8_geom_to_wheel[tire]
            normal_force = max(0., float(force[0]))
            wheel_force[wheel] += normal_force
            contacts.append(dict(contact_index=contact_index, wheel_index=wheel,
                patch_index=self._v8_geom_to_patch[lip], tire_geom_id=tire, lip_geom_id=lip,
                tire_geom_name=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, tire),
                lip_geom_name=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, lip),
                geom1=geom1, geom2=geom2, position=contact.pos.tolist(), distance=float(contact.dist),
                normal_force=normal_force, force_local=force.copy().tolist(),
                force_world_on_wheel=(sign*np.asarray(contact.frame).reshape(3, 3).T @ force[:3]).tolist()))
        return contacts, wheel_force

    def _native_tire_bounds(self):
        bounds = []
        for geom_id, wheel in self._v8_geom_to_wheel.items():
            axis = self.data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
            radius, half_length = self.model.geom_size[geom_id, :2]
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = np.abs(axis)*half_length+radius*np.sqrt(np.maximum(0., 1-axis**2))
            elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = np.abs(axis)*half_length+radius
            else:
                raise RuntimeError("Unexpected native tire geometry")
            position = self.data.geom_xpos[geom_id]
            bounds.append(dict(geom_id=geom_id, wheel_index=wheel,
                lower=(position-extent).tolist(), upper=(position+extent).tolist()))
        return bounds

    def _external_forces(self):
        # Original brush/depression forces remain identical. Suppressing only
        # the old blocked-patch spring here avoids double-counting native lip
        # constraints; the true map stays available solely for audit/rendering.
        original_map = self.terrain_map
        self.terrain_map = [p for p in original_map if p["kind"] == "depression"]
        try:
            super()._external_forces()
        finally:
            self.terrain_map = original_map
        _, loads = self._native_lip_contacts()
        self._obstacle_contact_force[:] = loads
        self._blocked_contact_truth[:] = loads > 1e-6
        self._blocked_encounter_truth |= bool(np.any(loads > 1.))

    def get_audit_state(self):
        contacts, loads = self._native_lip_contacts()
        info = super().get_audit_state()
        info.update(native_obstacle_model="native_mujoco_finite_tire_static_box_lips_v8",
            native_obstacle_contact_config=asdict(self.contact_config),
            native_obstacle_contacts=contacts, native_obstacle_geometry=deepcopy(self._v8_contact_geometry),
            native_tire_geom_bounds=self._native_tire_bounds(),
            native_obstacle_lip_geom_ids=self._v8_lip_geom_ids.tolist(),
            native_obstacle_tire_geom_ids=self._v8_tire_geom_ids.tolist(),
            native_obstacle_max_penetration=max([0., *[-x["distance"] for x in contacts]]),
            obstacle_contact_force_truth=loads.tolist(), blocked_contact_truth=(loads > 1e-6).tolist(),
            blocked_encounter_truth=bool(self._blocked_encounter_truth or np.any(loads > 1.)))
        return info

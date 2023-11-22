# FIXME duplicated => SHARE!!
from dataclasses import dataclass
from enum import Enum
from typing import List

from scipy.spatial.transform import Rotation as scipy_r

from common.common_transforms import *


class ErrorType(Enum):
    ADDITIVE = 1
    MULTIPLICATIVE = 2


class LazyRandom:

    # NOTE: PYTHON - this is a trick to force the dataclasses containing
    # instances of this class to call __str__ (through __repr__)
    def __repr__(self):
        return self.__str__()

    def __init__(self, type: ErrorType):
        self.type = type

    def get_error_by_type(self, orig_value, random_value):
        if self.type == ErrorType.ADDITIVE:
            return orig_value + random_value
        elif self.type == ErrorType.MULTIPLICATIVE:
            return orig_value * random_value
        else:
            raise ValueError(f"Unknown error type: {self.type}")

    def get_err_vector(self, orig_value):
        random_value = self.get_vector(orig_value.shape)
        return self.get_error_by_type(orig_value, random_value)

    def get_vector(self, shape):
        pass


class LazyGauss(LazyRandom):

    def __str__(self) -> str:
        return f"Normal(sigma={self.sigma}, mu={self.mu}), {self.type} error"

    def __init__(self, sigma, mu=0, type=ErrorType.ADDITIVE):
        self.mu = mu
        self.sigma = sigma
        super().__init__(type)

    def get_vector(self, shape):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=shape)


class LazyConstant(LazyRandom):

    def __str__(self) -> str:
        return f"constant = {self.value}, {self.type} error, change signs = {self.change_signs}, elementwise = {self.elementwise}"

    def __init__(self, value, type=ErrorType.ADDITIVE, change_signs=False, elementwise=False):
        assert change_signs or not elementwise
        self.value = value
        self.change_signs = change_signs
        self.elementwise = elementwise
        super().__init__(type=type)

    def get_vector(self, shape):
        if self.change_signs:
            if not self.elementwise:
                v = self.value
                if random.uniform(0.0, 1.0) > 0.5:
                    v = -v
            else:
                v = np.random.uniform(low=0.0, high=1.0, size=shape)
                v = ((v > 0.5).astype(float) * 2 - 1) * self.value
        else:
            v = self.value
        # TODO note that this is a special case -> value of 0.05 for multiplicative
        # would actually mean either 1.05 or 0.95, this is not consistent to other uses
        # of multiplicative (where it would be e.g. mu=1, sigma > 0)
        if self.type == ErrorType.MULTIPLICATIVE:
            v += 1.0
        return np.ones(shape=shape) * v


class LazyConstantNorm(LazyRandom):

    def __str__(self) -> str:
        return f"constant norm = {self.value}, {self.type} error"

    def __init__(self, value, type=ErrorType.ADDITIVE):
        self.value = value
        super().__init__(type=type)

    def get_vector(self, shape):
        assert len(shape) == 1
        un = np.random.uniform(low=-1.0, high=1.0, size=shape)
        un = un / np.linalg.norm(un) * self.value
        assert np.isclose(np.linalg.norm(un), self.value)
        return un


class LazyUniform(LazyRandom):

    def __str__(self) -> str:
        return f"Uniform({self.min, self.max}), {self.type} error"

    def bounds(self) -> str:
        return f"{self.min}_{self.max}"

    def __init__(self, min, max, type=ErrorType.ADDITIVE):
        self.min = min
        self.max = max
        super().__init__(type)

    def get_vector(self, shape):
        return np.random.uniform(low=self.min, high=self.max, size=shape)


class Smp(Enum):
    EXACT = 0
    UNIFORM = 1


class Rot(Enum):

    def __str__(self):
        mm = {
            Rot.X: "X",
            Rot.Y: "Y",
            Rot.Z: "Z",
        }
        return mm[self]

    X = 0
    Y = 1
    Z = 2


DEV_MAX = 180


@dataclass
class RSampling:
    axis: Rot
    deviation: float
    deviationType: Smp


class OldRotationConstraint(Enum):
    NONE = 0
    YZ_ROTATION = 1
    Y_ROTATION_Z_DEV = 2
    Y_ROTATION_X_DEV = 3
    VERTICAL = 4
    VERTICAL_FIXED = 5
    IDENTITY = 6
    BB_Y_Z = 7
    BB_Y_X_Z = 8


class RotSampler:

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"CompositionRotationSampler([{self.rotations}]"

    def __init__(self, rotations: List[RSampling]):
        # validate
        assert len(rotations) <= 3
        assert len(set([r.axis for r in rotations])) == len(rotations), f"duplicate axes in {rotations}"
        self.rotations = rotations
        used_axes = set([rotation.axis for rotation in self.rotations])
        self.unused_rotations = list(set([Rot.X, Rot.Y, Rot.Z]) - used_axes)

    def sample_one_rotation(self, rotation: RSampling):

        function_map = {
            Smp.EXACT: {
                Rot.X: rot_x_exact_deg,
                Rot.Y: rot_y_exact_deg,
                Rot.Z: rot_z_exact_deg,
            },
            Smp.UNIFORM: {
                Rot.X: rot_x_uniform_deg,
                Rot.Y: rot_y_uniform_deg,
                Rot.Z: rot_z_uniform_deg,
            }
        }
        return function_map[rotation.deviationType][rotation.axis](rotation.deviation)

    def decompose_to_angles(self, m):

        string_map = {
            Rot.X: "x",
            Rot.Y: "y",
            Rot.Z: "z",
        }
        unused_axes = [string_map[r] for r in self.unused_rotations]
        rotation_str = "".join([string_map[rotation.axis] for rotation in self.rotations] + unused_axes)

        sc_rot = scipy_r.from_matrix(m)
        sc_angles = sc_rot.as_euler(rotation_str, degrees=True)

        return np.array(sc_angles)

    def decompose_and_info(self, m):

        pc = np.get_printoptions()["precision"]
        np.set_printoptions(precision=2)

        angles = self.decompose_to_angles(m).tolist()
        angles_zero = angles[len(self.rotations):]
        np.allclose(angles_zero, np.zeros_like(angles_zero))

        angles_from_left = angles[:len(self.rotations)]
        angles_from_left.reverse()
        fn_map = {
            Rot.X: (ROT_X_deg, lambda x: f"R_x({x:.01f})"),
            Rot.Y: (ROT_Y_deg, lambda y: f"R_y({y:.01f})"),
            Rot.Z: (ROT_Z_deg, lambda z: f"R_z({z:.01f})"),
        }
        m_assert = np.eye(3)
        pretty = []
        pretty_thorough = []
        for deviation, rotation in zip(angles_from_left, self.rotations[::-1]):
            fnc, r_pretty_f = fn_map[rotation.axis]
            rotation = fnc(deviation)
            m_assert = m_assert @ rotation
            pretty_rotation = r_pretty_f(deviation)
            pretty.append(pretty_rotation)
            pretty_thorough.append(f"{pretty_rotation}=\n{rotation}")
        assert np.allclose(m, m_assert)
        pretty_thorough = "\n".join(pretty_thorough)
        pretty = " @ ".join(pretty)

        np.set_printoptions(precision=pc)

        return pretty, pretty_thorough, angles_from_left

    def compose_rotation_from_left(self, angles):
        fn_map = {
            Rot.X: ROT_X_deg,
            Rot.Y: ROT_Y_deg,
            Rot.Z: ROT_Z_deg,
        }
        composed = np.eye(3)
        for deviation, rotation in zip(angles, self.rotations[::-1]):
            composed = composed @ fn_map[rotation.axis](deviation)
        return composed

    def assert_rotation(self, m, angles):

        # a) recreate the rotation matrix, assert the same matrix is created
        fn_map = {
            Rot.X: ROT_X_deg,
            Rot.Y: ROT_Y_deg,
            Rot.Z: ROT_Z_deg,
        }
        m_assert = np.eye(3)
        for deviation, rotation in zip(angles, self.rotations):
            m_assert = fn_map[rotation.axis](deviation) @ m_assert
        assert np.allclose(m_assert, m)

        # b) assert the rotation angles agains scipy decomposition
        angles = angles + [0.0] * len(self.unused_rotations)
        sc_angles = self.decompose_to_angles(m)
        assert np.allclose(np.array(angles), sc_angles)

    def sample(self):
        angles = []
        m = np.eye(3)
        for rotation in self.rotations:
            angle, rotation_m = self.sample_one_rotation(rotation)
            angles.append(angle)
            m = rotation_m @ m

        perform_assert_rotation = True
        if perform_assert_rotation:
            self.assert_rotation(m, angles)
        return m

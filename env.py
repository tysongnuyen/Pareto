import math
import random
import numpy as np
from typing import Any, List, Mapping, Tuple
from dataclasses import dataclass

length_m = 1000
width_m = 1000
num_uavs = 5
num_ues = 50
total_system_prbs = 135
if total_system_prbs % num_uavs:
    raise ValueError("total_system_prbs must divide evenly across UAVs")
uav_max_prb = total_system_prbs // num_uavs
uav_max_power_w = 2.0
power_step_w = 0.1
allocation_max_prbs = 10
allocation_max_power_w = 1.0
required_mbs = [5,10,15]
grid_size = 10
seed = 2026

eta_ant: float = 1.0
d_0 = 1.0
f_c = 3.5e9
C = 3.0e8
k_B = 1.380649e-23
T = 290.0
Bandwidth_BRP = 360000
a = 2.0
F = 9.0

@dataclass(frozen=True)
class PhysConstant:
    eta_ant: float = eta_ant
    d_0: float = d_0
    f_c: float = f_c
    C: float = C
    k_B: float = k_B
    T: float = T
    Bandwidth_BRP: float = Bandwidth_BRP
    a: float = a
    F: float = F
    Theta_U: Tuple[float, ...] = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)
    H_U: Tuple[float, ...] = (250.0, 260.0, 270.0, 280.0, 290.0, 300.0)
    P_max_w: float = uav_max_power_w

    @property
    def P_max_dbm(self) -> float:
        """Maximum transmit power in dBm, derived for reporting only."""

        return 10.0 * math.log10(self.P_max_w) + 30.0

    @property
    def noise_figure_linear(self) -> float:
        return 10 ** (self.F / 10.0)

@dataclass(frozen=True)
class UE:
    id: int
    x_m:float
    y_m:float
    require_mbps: float


@dataclass(frozen=True)
class UAV:
    id: int
    max_prbs: int
    max_power_w: float

# Chuyen vao trong thuat toan
@dataclass(frozen=True)
class solution:
    index: int
    x_u: float
    y_u: float
    h_u: float
    theta_u: float
    r_u: float


# =========== Mot vai ham tinh toan doi don vi
#Cai nay thi tu khai bao trong thuat toan cua minh
def dbm_to_wat(dbm: float)->float:
    return 10**((dbm-30)/10)

def wat_to_dbm(watt: float)->float:
    if watt <= 0:
        raise ValueError("watt must be positive")
    return 10.0 * math.log10(watt) + 30.0

def make_power_levels_w(
    step_w: float = power_step_w,
    max_power_w: float | None = None,
    *,
    include_zero: bool = False,
) -> np.ndarray:
    """Return a power grid ending at the supplied maximum, optionally with zero."""

    if not math.isfinite(step_w) or step_w <= 0:
        raise ValueError("step_w must be positive")
    maximum = phys.P_max_w if max_power_w is None else max_power_w
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_power_w must be positive")
    num_steps = int(round(maximum / step_w))
    if not math.isclose(num_steps * step_w, maximum, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("max_power_w must be an integer multiple of step_w")
    return np.arange(0 if include_zero else 1, num_steps + 1, dtype=np.float64) * step_w

def deg_to_rad(deg: float)->float:
    return math.pi/180 * deg

def antenna_gain(theta_deg: float,eta: float = eta_ant)->float:
    return 2*eta/(1 - math.cos(deg_to_rad(theta_deg)))

def dis_2d(x_u,y_u,x_k,y_k)->float:
    return math.hypot(x_u - x_k, y_u - y_k)

def dis_3d(x_u,y_u,h_u,x_k,y_k)->float:
    return math.sqrt((x_u - x_k)**2 + (y_u - y_k)**2 + h_u**2)

def coverage_radius(h_u: float, theta_deg: float)->float:
    return h_u * math.tan(deg_to_rad(theta_deg))



# =========== make UE + make UAV + make Solution
def make_ue(seed: int = seed)-> List[UE]:
    coordinate_rng = np.random.default_rng(seed)
    requirement_rng = random.Random(seed)
    ue = []
    for i in range(num_ues):
        x_m = coordinate_rng.uniform(0,width_m)
        y_m = coordinate_rng.uniform(0,length_m)
        r_k = requirement_rng.choice(required_mbs)
        temp_ue = UE(i,x_m,y_m,r_k)
        ue.append(temp_ue)
    return ue

phys = PhysConstant()

def make_uav()->List[UAV]:
    uav = []
    for i in range(num_uavs):
        temp_uav = UAV(i, uav_max_prb, phys.P_max_w)
        uav.append(temp_uav)
    return uav



def make_grid(start: float,end: float, num_point: int)->List[float]:
    offset = (end - start) / num_point
    return [start + i*offset for i in range(num_point+1)]

def make_solution()->List[solution]:
    y_m = make_grid(0,length_m, grid_size)
    x_m = make_grid(0,width_m, grid_size)
    sol: List[solution] = []
    for x in (x_m):
        for y in (y_m):
            for h in (phys.H_U):
                for theta in (phys.Theta_U):
                    sol.append(solution(index = len(sol), x_u = x,y_u = y,h_u = h,theta_u = theta, r_u = coverage_radius(h,theta)))

    if not sol:
        raise ValueError("Non avaliable solution")
    return sol



def gain(
    uav_pos: solution,
    ue: Any,
    physcon: PhysConstant = phys,
) -> float:
    ue_x = ue["x_m"] if isinstance(ue, Mapping) else ue.x_m
    ue_y = ue["y_m"] if isinstance(ue, Mapping) else ue.y_m
    ant_gain = antenna_gain(uav_pos.theta_u, physcon.eta_ant)
    distance_3d = dis_3d(
        uav_pos.x_u, uav_pos.y_u, uav_pos.h_u, ue_x, ue_y
    )
    free_space_gain = (
        physcon.C / (4 * math.pi * physcon.f_c * physcon.d_0)
    ) ** 2
    return ant_gain * free_space_gain * (physcon.d_0 / distance_3d) ** physcon.a

def sinr_w(
    total_power_w: float,
    num_prbs: int,
    channel_gain: float,
    physcon: PhysConstant = phys,
) -> float:
    if total_power_w <= 0 or num_prbs <= 0:
        return 0.0
    bandwidth = num_prbs * physcon.Bandwidth_BRP
    noise = physcon.k_B * physcon.T * bandwidth * physcon.noise_figure_linear
    return total_power_w * channel_gain / noise

def data_rate_mbps(
    total_power_w: float,
    num_prbs: int,
    channel_gain: float,
    physcon: PhysConstant = phys,
) -> Tuple[float, float]:
    sinr = sinr_w(total_power_w, num_prbs, channel_gain, physcon)
    rate_bps = num_prbs * physcon.Bandwidth_BRP * math.log2(1.0 + sinr)
    return rate_bps / 1e6, sinr




    

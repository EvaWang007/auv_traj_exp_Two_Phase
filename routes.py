import numpy as np

STYLE2_BASE = np.array([[120, 110], [190, 150], [250, 180], [210, 300], [150, 385], [235, 420], [325, 455], [455, 620], [380, 780], [560, 915]], dtype=np.float32)
STYLE4_BASE = np.array([[125, 105], [100, 145], [205, 185], [275, 195], [235, 305], [315, 335], [355, 410], [490, 560], [640, 735], [520, 900]], dtype=np.float32)

def build_waypoints(style: str, traj_id: int) -> np.ndarray:
    base = STYLE2_BASE if style == 'traj2_like' else STYLE4_BASE
    rng = np.random.default_rng(5000 + traj_id)
    jitter = np.column_stack([rng.uniform(-45, 45, size=len(base)), rng.uniform(-35, 35, size=len(base))]).astype(np.float32)
    if style == 'traj2_like':
        drift = np.column_stack([np.linspace(0, rng.uniform(80, 160), len(base)), np.linspace(0, rng.uniform(-10, 25), len(base))]).astype(np.float32)
    else:
        drift = np.column_stack([np.linspace(0, rng.uniform(110, 220), len(base)), np.linspace(0, rng.uniform(-25, 15), len(base))]).astype(np.float32)
    waypoints = np.clip(base + jitter + drift, [90, 90], [920, 950]).astype(np.float32)
    return waypoints

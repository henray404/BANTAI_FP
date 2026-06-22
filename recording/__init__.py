# recording/ — Person 5 full trajectory recorder + faithful replay.
"""Record EVERYTHING that happens in a scenario run (all joints, poses, events, reward, metadata)
to CSV + JSON sidecar, and replay the best run faithfully (kinematics preserved, not re-simulated).

Core (recorder.py) is pure-stdlib and testable off-Isaac. The Isaac glue (state_extractor.py,
replay.py) is import-safe and only touches torch/sim at call time.
"""

from .recorder import TrajectoryRecorder, TrajectoryReader

__all__ = ["TrajectoryRecorder", "TrajectoryReader"]

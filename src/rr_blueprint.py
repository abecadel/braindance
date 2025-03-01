import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    contents = [
        rrb.Spatial3DView(origin=f"{parent_log_path}"),
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=f"{pinhole_path}/image",
            ),
            rrb.Spatial2DView(
                origin=f"{pinhole_path}/segmentation",
            ),
            rrb.Spatial2DView(
                origin=f"{cam_log_path}/disparity",
            ),
        ),
    ]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(contents=contents, column_shares=[3, 1]),
        collapse_panels=True,
    )
    return blueprint

"""Render one colored PLY point-cloud frame to a publication-ready PNG.

Uses the same Open3D visualizer as PlayPointClouds.py, but configures a fixed
front camera, a high-resolution hidden window, and tight white-canvas cropping.
Running one frame per process avoids a Windows/Open3D issue where recreating a
hidden rendering window can yield blank images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def render(src: Path, dst: Path, width: int, height: int,
           point_size: float, zoom: float) -> tuple[int, int, int]:
    cloud = o3d.io.read_point_cloud(str(src))
    if cloud.is_empty():
        raise RuntimeError(f"Could not load point cloud: {src}")

    # 8i uses Y as the body-height axis. Rotate axes so Open3D's camera uses Z
    # vertically while retaining the original coordinates and RGB attributes.
    points = np.asarray(cloud.points).copy()[:, [0, 2, 1]]
    cloud.points = o3d.utility.Vector3dVector(points)

    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = dst.with_name(f".{dst.stem}_raw.png")

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(
        window_name=f"Render {src.stem}", width=width, height=height, visible=False
    )
    if not created:
        raise RuntimeError("Open3D could not create its hidden rendering window")

    try:
        vis.add_geometry(cloud)
        options = vis.get_render_option()
        options.background_color = np.array([1.0, 1.0, 1.0])
        options.point_size = point_size

        camera = vis.get_view_control()
        camera.set_lookat(cloud.get_axis_aligned_bounding_box().get_center())
        camera.set_front([0.0, 1.0, 0.0])
        camera.set_up([0.0, 0.0, 1.0])
        camera.set_zoom(zoom)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(raw), do_render=True)
    finally:
        vis.destroy_window()

    image = Image.open(raw).convert("RGB")
    pixels = np.asarray(image)
    foreground = np.any(pixels < 248, axis=2)
    ys, xs = np.where(foreground)
    if not len(xs):
        raw.unlink(missing_ok=True)
        raise RuntimeError(f"Open3D produced a blank render: {src}")

    pad = max(30, round(min(width, height) * 0.018))
    crop = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(image.width, int(xs.max()) + pad + 1),
        min(image.height, int(ys.max()) + pad + 1),
    )
    image.crop(crop).save(dst, format="PNG", optimize=False)
    raw.unlink(missing_ok=True)
    size = Image.open(dst).size
    return len(cloud.points), size[0], size[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=3200)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--zoom", type=float, default=0.60)
    args = parser.parse_args()

    count, width, height = render(
        args.src, args.dst, args.width, args.height, args.point_size, args.zoom
    )
    print(f"{args.src.name}: {count:,} points -> {width}x{height} px -> {args.dst}")


if __name__ == "__main__":
    main()

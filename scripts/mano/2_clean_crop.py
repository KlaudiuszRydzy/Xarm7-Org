import os
import sys
from pathlib import Path
xgym_path = Path(__file__).resolve().parents[2]  # Adjust path to ~/repos/xgym
sys.path.append(str(xgym_path))
import numpy as np
from tqdm import tqdm
import jax
from xgym.rlds.util import apply_xyz, apply_persp, solve_uv2xyz, apply_uv, perspective_projection
from xgym.rlds.util.transform import center_crop
from xgym.rlds.util.render import render_openpose


# Preclean function
def preclean(step):
    dropkeys = [
        "box_center",
        "box_size",
        "img_size",
        "personid",
        "cam",
        "cam_t",
        "vertices",
    ]
    step = step.data if hasattr(step, "data") else step
    step = {k.replace("pred_", ""): v for k, v in step.items()}
    step["keypoints_3d"] += step.pop("cam_t_full")[None, :]
    print(f"After preclean -> keypoints_3d: {step['keypoints_3d'].shape}")

    for k in dropkeys:
        step.pop(k, None)  # Avoid errors if the key doesn't exist
    return step


# Pipe function
def pipe(frame, out):
    out = out.data if hasattr(out, "data") else out
    f = out["scaled_focal_length"]
    print("persp start")
    P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
    print("persp end")
    points = out["keypoints_3d"]
    size = 224  # Final crop size

    # Apply center cropping
    transform = center_crop(size=size, seed=None, img=frame)
    print("uv start")
    frame = apply_uv(frame, mat=transform, dsize=(size, size))
    print("uv end")
    print("solve start")
    T = solve_uv2xyz(points, P=P, U=transform)
    print("solve end")

    print("apply xyz start")
    print(f"points shape: {points.shape}, T shape: {T.shape}")
    points3d = apply_xyz(points, mat=T)
    print("apply xyz end")
    print("apply persp start")
    points2d = apply_persp(points3d, P)
    print("apply persp end")

    out["keypoints_3d"] = points3d
    out["keypoints_2d"] = points2d
    out["img"] = frame
    return out


# Clean function
def clean(out):
    out = out.data if hasattr(out, "data") else out
    obskeys = [
        "focal_length",
        "scaled_focal_length",
        "img",
        "keypoints_2d",
        "keypoints_3d",
        "mano.betas",
        "mano.global_orient",
        "mano.hand_pose",
        "right",
    ]
    out = {k.replace("pred_", "").replace("_params", ""): v for k, v in out.items()}
    out = jax.tree_map(
        lambda x: x[0].astype(np.float32) if len(x.shape) and x.shape[0] in [1, 2] else x,
        out,
    )
    out = {k: v for k, v in out.items() if k in obskeys}
    out["keypoints_2d"] = out["keypoints_2d"].reshape(21, 3)[:, :-1]
    if not all(k in out for k in obskeys):
        return None  # Skip invalid data
    return out


# Main function
def main():
    # Define directories
    MANO_2 = Path.home() / "data_xgym" / "mano" / "2_hamer"
    MANO_3 = Path.home() / "data_xgym" / "mano" / "3_center"
    MANO_2DONE = MANO_2.parent / f"{MANO_2.stem}_done"

    MANO_3.mkdir(exist_ok=True)
    MANO_2DONE.mkdir(exist_ok=True)

    # Process each stacked .npz file
    for npz_file in tqdm(list(MANO_2.glob("*.npz")), desc="Processing .npz files"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            sequence = {k: data[k] for k in data.files}

            # Apply preclean, pipe, and clean
            cleaned_sequence = []
            for i, frame in enumerate(sequence["img"]):
                step = {k: v[i] for k, v in sequence.items()}
                step = preclean(step)
                step = pipe(frame, step)
                step = clean(step)
                if step is not None:
                    cleaned_sequence.append(step)

            if cleaned_sequence:
                # Save cleaned sequence to MANO_3
                output_file = MANO_3 / npz_file.name
                np.savez(output_file, **{k: np.stack([s[k] for s in cleaned_sequence]) for k in cleaned_sequence[0]})

            # Move processed file to MANO_2DONE
            npz_file.rename(MANO_2DONE / npz_file.name)

        except Exception as e:
            print(f"Error processing {npz_file}: {e}")


if __name__ == "__main__":
    main()

"""Sample synthetic small-lesion patches from a trained DDPM.

Loads `ddpm.pth` produced by `train_lesion_ddpm.py`, samples N patches
conditioned on a log-volume distribution drawn uniformly over small
lesions (default: 0.05 mL to 2 mL), and writes them to
<nnUNet_preprocessed>/synthetic_lesion_bank/<idx>.npz with keys:
  image:    (1, 64, 64, 64) float32  (denormalised back to image-space if requested)
  vol_ml:   target volume scalar
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parents[1] / "src"))


def main() -> int:
    import numpy as np
    import torch
    from nnunet_isles.diffusion import LesionDDPM, LesionDDPMSampler

    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vol-min-ml", type=float, default=0.05)
    parser.add_argument("--vol-max-ml", type=float, default=2.0)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-diffusion-steps-override",
        type=int,
        default=None,
        help="optional: reduce sampling steps for smoke runs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(args.ddpm_checkpoint), map_location="cpu")
    net = LesionDDPM(in_channels=1, base_channels=ckpt["base_channels"]).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    sampler = LesionDDPMSampler(num_steps=ckpt["num_diffusion_steps"])

    rng = np.random.default_rng(args.seed)
    log_vols_all = np.log(rng.uniform(args.vol_min_ml, args.vol_max_ml, size=args.n_samples) + 1.0).astype(
        np.float32
    )

    n_done = 0
    for batch_start in range(0, args.n_samples, args.batch_size):
        batch_end = min(args.n_samples, batch_start + args.batch_size)
        log_vol = torch.from_numpy(log_vols_all[batch_start:batch_end])
        shape = (batch_end - batch_start, 1, args.patch_size, args.patch_size, args.patch_size)
        x = sampler.sample(
            net,
            shape=shape,
            log_vol=log_vol,
            device=device,
            n_steps_override=args.n_diffusion_steps_override,
        )
        for k in range(x.shape[0]):
            vol_ml = float(math.exp(log_vols_all[batch_start + k]) - 1.0)
            out_path = args.output_dir / f"synth_{batch_start + k:06d}.npz"
            np.savez_compressed(
                str(out_path),
                image=x[k].detach().cpu().numpy().astype(np.float32),
                vol_ml=np.float32(vol_ml),
            )
        n_done += batch_end - batch_start
        print(f"  sampled {n_done}/{args.n_samples}")

    print(f"[sample_synthetic_lesions] wrote {n_done} → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

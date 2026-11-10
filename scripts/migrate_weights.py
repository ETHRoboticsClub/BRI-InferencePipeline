"""Script to migrate and verify weights from BRI-TrainingPipeline to BRI-InferencePipeline."""

import argparse
import sys
from pathlib import Path

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import TRAINING_KEY_MAP, Model  # noqa: E402

DEFAULT_SRC = Path(r"C:\Users\pette\Documents\ETHRC_HRI\BRI-TrainingPipeline\runs\cross_subject\best_model.pt")
DEFAULT_DST = REPO_ROOT / "models" / "best_model.pt"


def migrate_weights(src_path: Path, dst_path: Path, verify: bool = True) -> None:
    """Migrate weights from training checkpoint to modular inference checkpoint.

    Args:
        src_path: Source checkpoint file path.
        dst_path: Destination checkpoint file path.
        verify: Whether to perform end-to-end verification of output identity.
    """
    if not src_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found at: {src_path}")

    print(f"[1/4] Loading source checkpoint from: {src_path}")
    raw_ckpt = torch.load(src_path, map_location="cpu")
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
    elif isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        state_dict = raw_ckpt["model"]
    else:
        state_dict = raw_ckpt

    print(f"[2/4] Mapping {len(state_dict)} checkpoint keys to modular architecture:")
    mapped_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = TRAINING_KEY_MAP.get(k, k)
        mapped_dict[new_k] = v
        print(f"  {k} -> {new_k} (shape {list(v.shape)})")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped_dict, dst_path)
    print(f"[3/4] Saved migrated state dict to: {dst_path}")

    if verify:
        print("[4/4] Verifying weight loading and numerical identity...")
        # 1. Load model with original weights via key mapper
        model_from_src = Model(channels=8, targets=15)
        model_from_src.load_pretrained_weights(src_path, map_location="cpu")
        model_from_src.eval()

        # 2. Load model with newly saved migrated weights
        model_from_dst = Model(channels=8, targets=15)
        model_from_dst.load_pretrained_weights(dst_path, map_location="cpu")
        model_from_dst.eval()

        # 3. Test on deterministic pseudo-random input
        torch.manual_seed(42)
        test_input = torch.randn(2, 8, 240)
        with torch.no_grad():
            out_src = model_from_src(test_input)
            out_dst = model_from_dst(test_input)

        max_diff = torch.max(torch.abs(out_src - out_dst)).item()
        print(f"  Maximum absolute difference: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Verification failed: max diff {max_diff} exceeds tolerance 1e-6!"
        assert out_dst.shape == (2, 15), f"Unexpected output shape: {out_dst.shape}"
        print(f"  Sample decoded joint angles (deg):\n  {out_dst[0, :5].tolist()}")
        print("[SUCCESS] Weight migration and verification completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate and verify TRR model weights.")
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help="Path to source checkpoint in BRI-TrainingPipeline",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help="Path to destination checkpoint in BRI-InferencePipeline",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip output verification test",
    )
    args = parser.parse_args()

    migrate_weights(src_path=args.src, dst_path=args.dst, verify=args.verify)


if __name__ == "__main__":
    main()

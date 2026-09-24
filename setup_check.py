"""Run this after `pip install -r requirements.txt` to verify your environment."""

import importlib
import sys

PACKAGES = ["torch", "torchvision", "numpy", "PIL", "cv2", "skimage", "matplotlib", "tqdm"]


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]      {name} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [MISSING] {name}")
            ok = False

    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: none (CPU only - use Colab/Kaggle for experiments)")

        # quick sanity check of the attack + metrics code
        from attack.pgd import pgd_disrupt
        from eval.metrics import psnr

        model = torch.nn.Conv2d(3, 3, 3, padding=1)
        x = torch.rand(1, 3, 64, 64)
        x_adv = pgd_disrupt(model, x, eps=8 / 255, steps=5)
        print(f"PGD smoke test: PSNR(clean, protected) = {psnr(x, x_adv):.1f} dB")
    except Exception as exc:  # noqa: BLE001
        print(f"Smoke test failed: {exc}")
        ok = False

    print("\nAll good!" if ok else "\nFix the issues above, then run again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

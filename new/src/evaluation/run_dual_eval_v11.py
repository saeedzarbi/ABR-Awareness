import os

from evaluate_all_models_v11 import run_eval


def main():
    configs = [
        ("policy", "_v11_policy"),
    ]

    # V11 focuses on method-level safety; system-wide guard is not applied here.
    os.environ.setdefault("ABR_SAFETY_GUARD", "0")

    for label, suffix in configs:
        print("=" * 72)
        print(f"Running V11 evaluation ({label})")
        print("=" * 72)
        run_eval(episodes_per_video=20, suffix=suffix)


if __name__ == "__main__":
    main()


import os

from evaluate_all_models_v13 import run_eval


def main():
    # V13 focuses on method-level guards; no global system guard.
    os.environ.setdefault("ABR_SAFETY_GUARD", "0")
    print("=" * 72)
    print("Running V13 evaluation")
    print("=" * 72)
    run_eval(episodes_per_video=20, suffix="_v13_policy")


if __name__ == "__main__":
    main()

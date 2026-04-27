import os

from evaluate_all_models_v7 import run_eval


def main():
    configs = [
        ("raw", "0", "_v7_raw"),
        ("light-guard", "light", "_v7_light"),
        ("strong-guard", "2", "_v7_safe"),
    ]

    for label, guard, suffix in configs:
        print("=" * 72)
        print(f"Running V7 evaluation ({label}, ABR_SAFETY_GUARD={guard})")
        print("=" * 72)
        os.environ["ABR_SAFETY_GUARD"] = guard
        # Default: apply guard only to RL methods (fair comparisons).
        # To treat the guard as a shared system-level component, set:
        #   set ABR_GUARD_SCOPE=all
        os.environ.setdefault("ABR_GUARD_SCOPE", "rl")
        run_eval(episodes_per_video=20, suffix=suffix, guard_scope=os.environ["ABR_GUARD_SCOPE"])


if __name__ == "__main__":
    main()


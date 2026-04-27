import os

from evaluate_all_models_v8 import run_eval


def main():
    configs = [
        ("raw", "0", "_v8_raw"),
        ("light-guard", "light", "_v8_light"),
        ("strong-guard", "2", "_v8_safe"),
    ]

    # Default: apply guard only to RL methods (fair comparisons).
    os.environ.setdefault("ABR_GUARD_SCOPE", "rl")

    for label, guard, suffix in configs:
        print("=" * 72)
        print(f"Running V8 evaluation ({label}, ABR_SAFETY_GUARD={guard}, ABR_GUARD_SCOPE={os.environ['ABR_GUARD_SCOPE']})")
        print("=" * 72)
        os.environ["ABR_SAFETY_GUARD"] = guard
        run_eval(episodes_per_video=20, suffix=suffix, guard_scope=os.environ["ABR_GUARD_SCOPE"])


if __name__ == "__main__":
    main()


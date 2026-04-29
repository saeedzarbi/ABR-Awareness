import os

from evaluate_all_models_v10 import run_eval


def main():
    """
    V10 evaluation launcher.

    Policy-only protocol:
      - ABR_SAFETY_GUARD=0
      - ABR_GUARD_SCOPE=none
    """

    configs = [
        ("policy-only", "0", "none", "_v10_policy"),
        ("system-light", "light", "rl", "_v10_light"),
        ("system-strong", "2", "rl", "_v10_safe"),
    ]

    for label, guard, scope, suffix in configs:
        print("=" * 72)
        print(
            f"Running V10 evaluation ({label}, ABR_SAFETY_GUARD={guard}, ABR_GUARD_SCOPE={scope})"
        )
        print("=" * 72)
        os.environ["ABR_SAFETY_GUARD"] = guard
        os.environ["ABR_GUARD_SCOPE"] = scope
        run_eval(episodes_per_video=20, suffix=suffix, guard_scope=scope)


if __name__ == "__main__":
    main()


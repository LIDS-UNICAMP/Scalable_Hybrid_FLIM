"""Run progressive NAS checks with an integer SDK teardown exit code.

W&B 0.27's protobuf teardown rejects the boolean SystemExit used by unittest's
CLI. This runner keeps standard unittest results and explicitly closes the SDK
with an integer; production experiment execution does not use unittest's exit.
"""
import sys
import unittest


CHECKS = ["spifil", "model", "metrics", "genetics", "pipeline", "config", "cpu",
          "dbcv", "logging", "wandb_pipeline", "monitor"]


def main(names=None):
    names = names or [f"analysis.checks.test_progressive_{name}" for name in CHECKS]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    code = 0 if result.wasSuccessful() else 1
    sdk = sys.modules.get("wandb")
    if sdk is not None:
        sdk.teardown(exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

import pytest
import os
import subprocess
import sys

def test_pipeline_runs():
    """
    Smoke test to ensure the training script runs end-to-end with the default config.
    We override max_epochs to 1 and limit batches to ensure speed.
    """
    # Construct the command to run the training script
    # We use overrides to make it fast
    command = [
        sys.executable, "src/train.py",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=2",
        "trainer.limit_test_batches=2",
        "trainer.accelerator=cpu", # Force CPU for CI/tests usually
        "trainer.devices=1",
        "data.batch_size=4",
        "data.num_workers=0", # Avoid multiprocessing issues in simple tests
        "experiment_name=test_run"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if it succeeded
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0, "Training script failed to run."
    assert "Starting training..." in result.stdout
    assert "Starting testing..." in result.stdout

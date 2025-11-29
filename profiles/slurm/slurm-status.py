#!/usr/bin/env python3
"""
SLURM status script for Snakemake.
Checks the status of a SLURM job.
"""

import subprocess
import sys
import time

jobid = sys.argv[1]

# Try multiple times in case of transient errors
for i in range(3):
    try:
        output = subprocess.check_output(
            ["sacct", "-j", jobid, "--format", "State", "--noheader", "--parsable2"],
            text=True
        )
        state = output.strip().split("\n")[0]

        # Map SLURM states to Snakemake states
        if state == "COMPLETED":
            print("success")
        elif state in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]:
            print("running")
        elif state in ["FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"]:
            print("failed")
        elif state == "CANCELLED":
            print("failed")
        else:
            print("running")

        break
    except subprocess.CalledProcessError as e:
        # Job might not be in sacct yet
        if i >= 2:
            print("failed")
        else:
            time.sleep(1)
            continue
    except Exception as e:
        print("failed")
        break

sys.exit(0)

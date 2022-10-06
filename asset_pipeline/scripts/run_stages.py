import multiprocessing
import os
import signal
import subprocess
import sys
import traceback
import yaml

import click

PARAMS_PATH = "../params.yaml"


def process_target(args):
    try:
        stage, target, timeout = args
        cmd = ["dvc", "repro", f"{stage}@{target}"]
        print("Running", " ".join(cmd))
        return subprocess.run(cmd, timeout=timeout, stdin=subprocess.PIPE).returncode == 0
    except:
        traceback.print_exc()
        return False


@click.command()
@click.argument('stage')
@click.option('--subset', default="combined", help='Which subset of targets to run.', type=str)
@click.option('--processes', default=3, help='Number of concurrent processes to run.', type=int)
@click.option('--timeout', default=15*60, help='Seconds to wait until the process is terminated.', type=int)
def run_stages(stage, subset, processes, timeout):
    params_path = os.path.join(os.path.dirname(__file__), PARAMS_PATH)
    with open(params_path, "r") as f:
        files = yaml.load(f, yaml.SafeLoader)

    assert subset in files.keys(), f"Could not find key {subset}"

    # Get the appropriate targets.
    targets = files[subset]

    # Run parallel processing.
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(processes)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        res = pool.map_async(process_target, [(stage, target, timeout) for target in targets])
        results = res.get(1e6) # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()
    pool.join()

    # Split the targets into successes and failures.
    successes = []
    failures = []
    for target, success in zip(targets, results):
        if success:
            successes.append(target)
        else:
            failures.append(target)

    # Print the results.
    print(f"Successes ({len(successes)} / {len(targets)}):")
    for x in sorted(successes):
        print("    " + x)

    print(f"\nFailures ({len(failures)} / {len(targets)}):")
    for x in sorted(failures):
        print("    " + x)

    print(f"\nSuccessfully ran ({len(successes)} / {len(targets)}) targets.")

if __name__ == "__main__":
    run_stages()
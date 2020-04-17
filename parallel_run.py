#!/usr/bin/env python
"""
Setup and Launch Multiple Model Runs

This is a wrapper of src/train.py that launches several models in parallel. It
reads an "experiments" yaml file, which includes some metadata and an entry for
each run-to-be-sbatched. For each of those runs,

* a new configuration yaml file is written, in a directory specified by the
  master experiments yaml file. Those yaml files are used as the argument for
  train.py
* an sbatch script is written, which tells what shell commands to run and how
  many resources to command, in the associated cluster job

After having written both the config and the launch script, we then call
'sbatch' on each of the launch scripts.

example use:
# note that singularity should not be loaded (since we need to call sbatch)
python3 parallel_run.py -e shared/experiment.yaml
"""
from pathlib import Path
from textwrap import dedent
import argparse
import os
import subprocess
import yaml


def env_to_path(path):
    """Transorms an environment variable mention in a conf file
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds
    Args:
        path (str): path potentially containing the env variable
    """
    if not isinstance(path, str):
        return path

    path_elements = path.split("/")
    for i, d in enumerate(path_elements):
        if "$" in d:
            path_elements[i] = os.environ.get(d.replace("$", ""))
    if any(d is None for d in path_elements):
        return ""
    return "/".join(path_elements)


def write_conf(run_dir, cname, param):
    """Write config file from params

    If conf_name exists, increments a counter in the name:
    explore.yaml -> explore (1).yaml -> explore (2).yaml ...
    """
    if not cname.endswith(".yaml"):
        cname += ".yaml"

    with open(run_dir / cname, "w") as f:
        yaml.dump(param, f, default_flow_style=False)
    return run_dir / cname


def zip_for_tmpdir(conf_path):
    """
    Copy files to Local Tmpdir

    Reading and writing is much faster when you copy them to the cluster's
    local tmpdir.
    """
    cmd = ""
    original_path = Path(conf_path).resolve()
    zip_name = original_path.name + ".tar"
    zip_path = str(original_path.parent / zip_name)

    no_zip = not Path(zip_path).exists()
    if no_zip:
        cmd = dedent(
            f"""\
            if [ -d "$SLURM_TMPDIR" ]; then
                cd {str(original_path.parent)}
                tar -cf {zip_name} {original_path.name} > /dev/null
            fi
            """
        )

    cmd += dedent(f"""
        echo "copying data"
        cp {zip_path} $SLURM_TMPDIR
        cd $SLURM_TMPDIR
        echo "untarring data"
        tar -xf {zip_name} > /dev/null
    """
    )
    return cmd


def template(param, conf_path, run_dir):
    zip_command = zip_for_tmpdir(env_to_path(param["data"]["path"]))
    sbp = param["sbatch"]
    indented = "\n            "
    base = "\n"
    zip_command = indented.join(zip_command.split(base))
    return dedent(
        f"""\
        #!/bin/bash
        #SBATCH --account=def-bengioy               # Yoshua pays for your job
        #SBATCH --cpus-per-task={sbp["cpus"]}
        #SBATCH --gres=gpu:1
        #SBATCH --mem={sbp["mem"]}
        #SBATCH --time={sbp["runtime"]}
        #SBATCH -o {env_to_path(sbp["slurm_out"])}  # Write the log in $SCRATCH

        {zip_command}
        cd {sbp["repo_path"]}

        module load singularity
        echo "Starting job"
        singularity exec --nv --bind {env_to_path(param["data"]["path"])},{str(run_dir)}\\
                {sbp["singularity_path"]}\\
                python3 -m src.train \\
                -c "{str(conf_path)}"\\
                -s 10
        """
    )

if __name__ == '__main__':
    # Parse arguments to this file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exploration_file",
        type=str,
        default="shared/experiment.yaml",
        help="Where to find the exploration file",
    )
    opts = parser.parse_args()

    # get configuration parameters for this collection of experiments
    default_train = yaml.safe_load(open("shared/train.yaml"))
    default_sbatch = yaml.safe_load(open("shared/sbatch.yaml"))

    with open(opts.exploration_file, "r") as f:
        exploration_params = yaml.safe_load(f)

    # setup the experiment directory
    exp_name = exploration_params["metadata"]["name"]
    exp_dir = env_to_path(exploration_params["metadata"]["directory"])
    exp_dir = Path(exp_dir).resolve()
    exp_dir = exp_dir / exp_name
    os.makedirs(exp_dir, exist_ok=True)

    # -----------------------------------------
    # Launch a collection of jobs, indexed by params
    for run_name, run_value in exploration_params["runs"].items():
        run_dir = exp_dir / f"{run_name}"
        os.makedirs(run_dir, exist_ok=True)

        default_sbatch.update(run_value["sbatch"])
        run_value["sbatch"] = default_sbatch

        default_train.update(run_value["train"])
        run_value["train"] = default_train

        conf_path = write_conf(run_dir, run_name, run_value["train"])
        template_str = template(run_value, conf_path, run_dir)

        file = run_dir / f"{run_name}.sh"
        with file.open("w") as f:
            f.write(template_str)

        print(subprocess.check_output(f"sbatch {str(file)}", shell=True))
        print("In", str(run_dir), "\n")

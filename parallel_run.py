#!/usr/bin/env python
from pathlib import Path
from textwrap import dedent
from src.cluster_utils import env_to_path, increasable_name
import yaml

def write_conf(run_dir, param):
    """Write config file from params

    If conf_name exists, increments a counter in the name:
    explore.yaml -> explore (1).yaml -> explore (2).yaml ...
    """
    cname = param["sbatch"].get("conf_name", "overwritable_conf")
    if not cname.endswith(".yaml"):
        cname += ".yaml"

    with open(run_dir / cname, "w") as f:
        yaml.dump(param["config"], f, default_flow_style=False)
    return run_dir / cname


def zip_for_tmpdir(conf_path):
    """
    Copy files to Local Tmpdir

    Reading and writing is much faster when you copy them to the cluster's
    local tmpdir.
    """
    cmd = ""
    original_path = Path(conf_path).resolve()
    zip_name = original_path.name + ".zip"
    zip_path = str(original_path / zip_name)

    no_zip = not Path(zip_path).exists()
    if no_zip:
        cmd = dedent(
            f"""\
            if [ -d "$SLURM_TMPDIR" ]; then
                cd {str(original_path)}
                zip -r {zip_name} patches masks > /dev/null
            fi
            """
        )

    cmd += dedent("""
        cp {zip_path} $SLURM_TMPDIR
        cd $SLURM_TMPDIR
        unzip {zip_name} > /dev/null
    """
    )
    return cmd



def template(param, conf_path, run_dir):
    zip_command = zip_for_tmpdir(param["config"]["data"]["original_path"])
    sbp = param["sbatch"]
    indented = "\n            "
    base = "\n"
    zip_command = indented.join(zip_command.split(base))
    return dedent(
        f"""\
        #!/bin/bash
        #SBATCH --account=rpp-bengioy               # Yoshua pays for your job
        #SBATCH --cpus-per-task={sbp["cpus"]}       # Ask for 6 CPUs
        #SBATCH --gres=gpu:1                        # Ask for 1 GPU
        #SBATCH --mem={sbp["mem"]}G                 # Ask for 32 GB of RAM
        #SBATCH --time={sbp.get("runtime", "24:00:00")}
        #SBATCH -o {env_to_path(sbp["slurm_out"])}  # Write the log in $SCRATCH

        {zip_command}

        module load singularity/3.4
        echo "Starting job"
        singularity exec --nv --bind {param["config"]["data"]["path"]},{str(run_dir)}\\
                {sbp["singularity_path"]}\\
                python3 -m src.train \\
                -m "{sbp["message"]}" \\
                -c "{str(conf_path)}"\\
                -o "{str(run_dir)}"
        """
    )

if __name__ == '__main__':
    param = yaml.load(open("/Users/krissankaran/Desktop/glacier_mapping/conf/defaults.yaml"))
    param["data"]["original_path"] = param["data"]["path"]


    # Parse arguments to this file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exploration_file",
        type=str,
        default="explore.yaml",
        help="Where to find the exploration file",
    )
    parser.add_argument(
        "-d",
        "--exp_dir",
        type=str,
        help="Where to store the experiment, overrides what's in the exp file",
    )
    opts = parser.parse_args()

    # get configuration parameters for this collection of experiments
    default_yaml = yaml.safe_load(open("shared/defaults.yaml"))
    sbatch_yaml = yaml.safe_load(open("shared/sbatch.yaml"))

    with open(opts.exploration_file, "r") as f:
        exploration_params = yaml.safe_load(f)
        assert isinstance(exploration_params, dict)

    # setup the experiment directory
    exp_dir = Path(
        env_to_path(exploration_params["experiment"]["exp_dir"])
    ).resolve()

    exp_name = exploration_params["experiment"].get("name", "explore-experiment")
    exp_dir = EXP_ROOT_DIR / exp_name
    exp_dir = increasable_name(exp_dir)
    exp_dir.mkdir()

    # -----------------------------------------
    # Get parameters corresponding to each experiment
    #
    # params: List[Dict[tr, Any]] = []
    params = []
    exp_runs = exploration_params["runs"]
    if "repeat" in exploration_params["experiment"]:
        exp_runs *= int(exploration_params["experiment"]["repeat"]) or 1
    for p in exp_runs:
        params.append(
            {
                "sbatch": {**sbatch_yaml, **p["sbatch"]},
                "config": {
                    "model": {
                        **default_yaml["model"],
                        **(p["config"]["model"] if "model" in p["config"] else {}),
                    },
                    "train": {
                        **default_yaml["train"],
                        **(p["config"]["train"] if "train" in p["config"] else {}),
                    },
                    "data": {
                        **default_yaml["data"],
                        **(p["config"]["data"] if "data" in p["config"] else {}),
                    },
                },
            }
        )

    # -----------------------------------------
    # Launch a collection of jobs, indexed by params
    for i, param in enumerate(params):
        run_dir = exp_dir / f"run_{i}"
        run_dir.mkdir()
        sbp = param["sbatch"]

        original_data_path = param["config"]["data"]["path"]
        param["config"]["data"]["path"] = "$SLURM_TMPDIR"
        param["config"]["data"]["original_path"] = original_data_path
        conf_path = write_conf(run_dir, param)  # returns Path() from pathlib
        template = get_template(param, conf_path, run_dir, opts.template_name)

        file = run_dir / f"run-{sbp['conf_name']}.sh"
        with file.open("w") as f:
            f.write(template)

        print(subprocess.check_output(f"sbatch {str(file)}", shell=True))
        print("In", str(run_dir), "\n")

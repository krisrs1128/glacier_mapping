#!/usr/bin/env python
from pathlib import Path
from textwrap import dedent
import yaml

def write_conf(run_dir, param):
    """Write config file from params to config/conf_name
    If conf_name exisits, increments a counter in the name:
    explore.yaml -> explore (1).yaml -> explore (2).yaml ...
    """
    cname = param["sbatch"].get("conf_name", "overwritable_conf")
    if not cname.endswith(".yaml"):
        cname += ".yaml"

    with open(run_dir / cname, "w") as f:
        yaml.dump(param["config"], f, default_flow_style=False)
    return run_dir / cname


def zip_for_tmpdir(conf_path):
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
                -o "{str(run_dir)}" \\
                {"-n" if sbp["no_comet"] else "-f" if sbp["offline"] else ""}
        """
    )


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




import re
import time
import argparse
import pandas as pd

from subprocess import Popen, PIPE
from typing import Optional, Tuple


def nsml_account_check(nsml_path, nsml_auth_dir=""):
    # check the NSML account
    if nsml_auth_dir == "":
        res = run_cmd(command="{} whoami".format(nsml_path))
    else:
        res = run_cmd(command="{} whoami".format(nsml_path), env={"NSML_AUTH_DIR": nsml_auth_dir})
    _nsml_id = str(res[0].decode("utf-8")).strip("\n")
    return _nsml_id


def get_inference_session_name(stdout, pattern, base_kw, user_id=""):
    _res = list(set(re.findall(pattern=pattern, string=stdout.decode())))
    return "{}/{}/{}".format(user_id, base_kw, _res[0])


def run_cmd(command: str, verbose: bool = True, env: Optional = None) -> Tuple[bytes, bytes, int]:
    if verbose:
        print("Command: {}".format(command))
    popen = Popen(command, stdout=PIPE, stderr=PIPE, env=env, shell=True)
    # NOTE: both stdout and stderr are important (stdout: from nsml_v0, stderr: from nsml server)
    stdout, stderr = popen.communicate()
    return stdout, stderr, popen.returncode  # 0 if success otherwise 1


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--nsml_path', type=str, default="~/nsml")
    argparse.add_argument('--nsml_id', type=str, default="KR20343", help="NSML ID")
    argparse.add_argument('--nsml_auth_dir', type=str, default="")
    argparse.add_argument('--nsml_data_name', type=str, default="")
    argparse.add_argument('--gpu_driver_version', type=int, default=0)
    argparse.add_argument('--num_cpus', type=int, default=8)
    argparse.add_argument('--size_memory', type=int, default=15)
    argparse.add_argument('--device', type=str, default="cuda")
    argparse.add_argument('--params_path', type=str, default="./data/params/params.csv")
    args = argparse.parse_args()
    print(args)

    nsml_id = nsml_account_check(nsml_path=args.nsml_path, nsml_auth_dir=args.nsml_auth_dir)
    assert nsml_id == args.nsml_id, "Current account: {} Your specified account: {}".format(nsml_id, args.nsml_id)

    # Make the escape text file
    with open("sess_names_log.txt", "w") as file:
        file.write("TEMP\n")

    # Make the escape text file
    with open("error_log.txt", "w") as file:
        file.write("TEMP\n")

    if args.nsml_data_name != "":
        additional_cmd = f"-d {args.nsml_data_name}"
        base_kw = args.nsml_data_name
        pattern = r'{}\W\w*\W\w*\W(\d*)'.format(args.nsml_id)
    else:
        additional_cmd = ""
        base_kw = "None"
        pattern = r'{}\W\w*\W(\d*)'.format(args.nsml_id)

    _sess_names_list, _cmd_list = list(), list()
    num_cpus = "-c {}".format(args.num_cpus)
    device = "-g 0" if args.device == "cpu" else ""
    if args.gpu_driver_version == 0:
        driver = ""
    else:
        driver = f"--gpu-driver-version={str(args.gpu_driver_version)}" if args.device == "cuda" else ""
    returncode = 1
    while returncode == 1:
        # Set a command
        _tmp = ""
        # https://line-enterprise.slack.com/archives/CLH4038HK/p1673591027358469?thread_ts=1673581311.611289&cid=CLH4038HK
        # _tmp = "--gpu-model=P40" if args.nsml_id == "KR81092" else ""
        _tmp = "--gpu-model=V100" if args.nsml_id == "KR81092" else ""
        _cmd = '{nsml_path} run --memory="{size_memory}G" {gpu_driver_version} {device} {num_cpus} --esm=A019480 ' \
               '{additional_cmd} -e launch_run.py {tmp} --args "--params_path={params_path}"'.format(
            nsml_path=args.nsml_path, gpu_driver_version=driver, device=device, num_cpus=num_cpus,
            size_memory=args.size_memory, additional_cmd=additional_cmd, params_path=args.params_path, tmp=_tmp
        )

        # Execute the command
        if args.nsml_auth_dir == "":
            # === on Local
            stdout, stderr, returncode = run_cmd(command=_cmd)
        else:
            # === on NFS
            stdout, stderr, returncode = run_cmd(command=_cmd, env={"NSML_AUTH_DIR": args.nsml_auth_dir})
        if returncode == 1:
            print("=== Error Log ===")
            # print(stderr)
            print(stdout)
            with open("error_log.txt", "a") as file:
                file.write("{}\n{}\n".format(_cmd, stdout))
            time.sleep(30)  # this is from the preprocess pipeline
    print(stdout, returncode)
    sess_name = get_inference_session_name(stdout=stdout,
                                           pattern=pattern,
                                           base_kw=base_kw,
                                           user_id=args.nsml_id)

    df_params = pd.read_csv(args.params_path)
    df_params["nsml_sess"] = sess_name
    # df_params.to_csv(args.params_path, index=False)

# local -> specify params-csv -> main_nsml -> launch many runs with main_temp
# nsml run main.py --args=csv_path -> python main.py --args=....

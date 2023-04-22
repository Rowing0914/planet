import time
import argparse
import pandas as pd

from launch_nsml_session import nsml_account_check, get_inference_session_name, run_cmd

NSML_SESS_NAME_COL = "memo"

argparse = argparse.ArgumentParser()
argparse.add_argument('--nsml_path', type=str, default="~/nsml")
argparse.add_argument('--nsml_id', type=str, default="KR20343", help="NSML ID")
argparse.add_argument('--nsml_auth_dir', type=str, default="")
argparse.add_argument('--mount_nfs', type=bool, default=False)
argparse.add_argument('--gpu_driver_version', type=int, default=0)
argparse.add_argument('--num_cpus', type=int, default=8)
argparse.add_argument('--size_memory', type=int, default=15)
argparse.add_argument('--device', type=str, default="cpu")
argparse.add_argument('--nsml_mainPy', type=str, default="main.py")
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

# Read the csv to organise all the hyper-params
df_params = pd.read_csv(args.params_path)
# print(df_params.columns)
if "nsml_sess" in df_params.columns:
    df_params = df_params.drop("nsml_sess", 1)
# print(df_params.columns)

list_params = list()
for row in df_params.iterrows():
    dict_param = row[1].to_dict()
    list_params.append(" ".join(["--{}={}".format(k, v) for k, v in dict_param.items() if k != NSML_SESS_NAME_COL]))

if args.mount_nfs:
    # additional_cmd = "-d wallet-nfs --nfs-output"  # this is for fire server
    additional_cmd = "-d ami-nfs-slink --nfs-output"  # this is for cdb server
    # base_kw = "wallet-nfs"
    base_kw = "ami-nfs-slink"
    pattern = r'{}\W\w*\W\w*\W(\d*)'.format(args.nsml_id)
else:
    additional_cmd = ""
    base_kw = "None"
    pattern = r'{}\W\w*\W(\d*)'.format(args.nsml_id)

_sess_names_list, _cmd_list = list(), list()
for params in list_params:
    try:
        num_cpus = [i for i in params.split("--") if i.startswith("num_cpus")][0].split("=")[-1].strip(" ")
    except:
        num_cpus = args.num_cpus
    params = params.replace("--num_cpus={}".format(int(num_cpus)), "")
    num_cpus = "-c {}".format(int(num_cpus))
    # _device = [i for i in params.split("--") if i.startswith("device")][0].split("=")[-1].strip(" ")
    _device = args.device
    device = "-g 0" if _device == "cpu" else ""
    if args.gpu_driver_version == 0:
        driver = ""
    else:
        driver = f"--gpu-driver-version={str(args.gpu_driver_version)}" if _device == "cuda" else ""
    returncode = 1
    while returncode == 1:
        # Set a command
        # https://line-enterprise.slack.com/archives/CLH4038HK/p1673591027358469?thread_ts=1673581311.611289&cid=CLH4038HK
        _tmp = "--gpu-model=V100" if args.nsml_id == "KR81092" else ""
        _cmd = '{nsml_path} run --memory="{size_memory}G" {gpu_driver_version} {device} {num_cpus} --esm=A019480' \
               ' {additional_cmd} {tmp} -e {mainPy} --args "{params}"'.format(
            nsml_path=args.nsml_path, gpu_driver_version=driver, device=device, size_memory=args.size_memory,
            num_cpus=num_cpus, additional_cmd=additional_cmd, mainPy=args.nsml_mainPy, params=params, tmp=_tmp
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
    _sess_names_list.append(sess_name)
    _cmd_list.append(_cmd)
    print("{} / {}".format(len(_sess_names_list), len(list_params)))
    time.sleep(5)  # this is from the preprocess pipeline

    print("=== Updating the temp file... ===")
    # Save the sess name and the cmd temporarily
    with open("sess_names_log.txt", "a") as file:
        file.write("{}\n{}\n".format(_cmd, sess_name))
    print("=== Finish updating the temp file... ===")

df_params["nsml_sess"] = _sess_names_list
# df_params.to_csv(args.params_path, index=False)

with open("cmd.txt", "w") as file:
    file.write("\n".join(_cmd_list))

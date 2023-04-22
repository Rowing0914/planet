import argparse
import subprocess
import pandas as pd

argparse = argparse.ArgumentParser()
argparse.add_argument('--params_path', type=str, default="./data/params/control/VR_Longer/vr_ddpg_params.csv")
args = argparse.parse_args()
print(args)

df_params = pd.read_csv(args.params_path)
list_params = []
if "nsml_sess" in df_params: del df_params["nsml_sess"]
for row in df_params.iterrows():
    dict_param = row[1].to_dict()
    list_params.append(' '.join([f'--{k}={v}' for k, v in dict_param.items() if k != "name"]))

# if df_params["env"][0].startswith("mujoco"):
print("================ START: prep mujoco ====================")
subprocess.Popen("sh ./prep_mujoco.sh", shell=True).wait()
import os

os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH')}:/home/nsml/.mujoco/mujoco210/bin"
# import mujoco_py  # this is to fully compile the mujoco for the following main exp
# this is to fully compile the mujoco for the following main exp
subprocess.Popen("python ./_mujoco_demo.py", shell=True).wait()
print("================ END: prep mujoco ====================")

processes = list()
for i, params in enumerate(list_params):
    print("python main.py {params}".format(params=params))
    process = subprocess.Popen("python main.py {params}".format(params=params), shell=True)
    processes.append(process)

output = [p.wait() for p in processes]

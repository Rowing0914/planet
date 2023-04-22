# this doesn't take long
mkdir -p ./.mujoco &&
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz &&
  tar -xf mujoco.tar.gz -C ./.mujoco &&
  rm mujoco.tar.gz

ls ./.mujoco/mujoco210/
cp -R ./.mujoco/ /home/nsml/.mujoco/
ls /home/nsml/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nsml/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

echo "start pip install"
pip install mujoco-py==2.1.2.14
echo "done pip install"

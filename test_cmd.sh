nsml run -e=main.py -c=8 --gpu-model=V100 --esm=A019480 --args=""
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/pendulum.csv

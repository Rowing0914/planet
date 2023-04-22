nsml run -e=main.py -c=8 --gpu-model=V100 --esm=A019480 --args=""

python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/ant.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/halfcheetah.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/hopoper.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/humanoid.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/humanoid-standup.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/inv-double-pendulum.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/inv-pendulum.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/reacher.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/swimmer.csv
python launch_nsml_session.py --nsml_id=KR81092 --device=cuda --params_path=./params/walker2d.csv

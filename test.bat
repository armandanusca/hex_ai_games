cd ./agents/
python setup.py build_ext --inplace
cd ..
python .\Hex.py "a=rave_pypy;python agents/mcts_agent.py" "a=rave_norma;python agents/naive_mcts_agent.py" -v
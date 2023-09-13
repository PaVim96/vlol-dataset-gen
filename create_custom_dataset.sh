# Custom dataset with custom rule
docker run -it --gpus all -v $(pwd):/home/workdir vlol  python3 main.py --replace_symbolics True --dataset_size 16 --min_train_length 1 --max_train_length 1 --visualization SimpleObjects --classification custom --background base_scene --distribution RandomTrains --distribution_config thesis_data.json
# Classic dataset with easy rule
docker run --gpus device=0 -v $(pwd):/home/workdir vlol  main.py --visualization SimpleObjects --classification easy --background base_scene --distribution MichalskiTrains

from michalski_trains.michalski_intervention_dataset import MichalskiIntervenedDataset
import argparse 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument("--dataset-settings", required=False, default ="SimpleObjects_custom_RandomTrains_base_scene_len_1-1", type=str, help="Filename in output/image_generator containing all data and settings")
    args = parser.parse_args()


    data = MichalskiIntervenedDataset("SimpleObjects_custom_RandomTrains_base_scene_len_1-1", ds_size=16, resize=True)
    data[0]
    import ipdb; ipdb.set_trace()
    print("XD")


"docker run -it --gpus all -v $(pwd):/home/workdir vlol  python3 create_dataset.py"
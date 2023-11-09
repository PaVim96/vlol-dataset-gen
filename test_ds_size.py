import os
def print_missing_idxs(full_path, ds_size, num_interventions): 
    files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
    #sample numbers
    file_ints = [int(name.split("_")[0]) for name in files]

    #intervention numbers
    file_int_ints = [int(name.split("_")[3]) for name in files]
    print("Number of samples: {} of {}".format(len(file_ints), ds_size * (num_interventions + 1)))
    unique_ints = set(file_ints)

    
    print("Testing whether any particular sample is missing:")
    sample_missing = False
    for i in range(ds_size): 
        if i not in unique_ints: 
            sample_missing = True
            print(i)
    print("Sample missing: {}".format(sample_missing))

    print("Testing whether any intervention is missing on a particular sample")
    int_dims = {k: [] for k in unique_ints if k in range(ds_size)}
    for curr_key in int_dims.keys(): 
        needed_idxs = [i for i, val in enumerate(file_ints) if val == curr_key]
        intervention_num_list = [file_int_ints[i] for i in needed_idxs]
        int_dims[curr_key] = intervention_num_list

    needed_length = num_interventions + 1
    intervention_missing = False
    for k,v in int_dims.items(): 
        if len(v) != needed_length: 
            intervention_missing = True
            print("Sample number {} is missing interventions {}".format(k, v))
    print("Any interventional sample missing: {}".format(intervention_missing))

path = "/home/patrick/Desktop/CCKD/datasets/MichalskiTrain/SimpleObjects_custom_RandomTrains_base_scene_len_1-3/images"
num_interventions = 4 
samples = 8000
print_missing_idxs(path, samples, num_interventions)

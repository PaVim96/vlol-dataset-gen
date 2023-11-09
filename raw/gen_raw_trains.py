import math
import os
import random
from pyswip import Prolog
from michalski_trains.m_train import BlenderCar, MichalskiTrain, SimpleCar
from copy import deepcopy

def gen_raw_michalski_trains(class_rule, out_path, num_entries=10000, with_occlusion=False, min_cars=2, max_cars=4):
    """ Generate Michalski trains descriptions using the Prolog train generator
        labels are derived by the classification rule
    Args:
        out_path: string path to save the generated train descriptions
        num_entries: int, number of michalski trains which are generated
        with_occlusion: boolean, whether to include occlusion of the train payloads
        class_rule: str, classification rule used to derive the labels
    """
    rule_path = f'example_rules/{class_rule}_rule.pl'
    os.makedirs('output/tmp/raw/', exist_ok=True)
    generator_tmp = 'output/tmp/raw/generator_tmp.pl'
    try:
        os.remove(generator_tmp)
    except OSError:
        pass
    with open("raw/train_generator_long_cars.pl", 'r') as gen, open(rule_path, 'r') as rule:
        with open(generator_tmp, 'w+') as generator:
            generator.write(gen.read())
            generator.write(rule.read())

    prolog = Prolog()
    prolog.consult(generator_tmp)
    try:
        os.remove(out_path)
    except OSError:
        os.makedirs(out_path.rsplit('/', 1)[0], exist_ok=True)
        pass
    with open(out_path, 'w+') as all_trains:
        west_counter, east_counter = 0, 0
        while west_counter < int(math.ceil(num_entries / 2)) or east_counter < int(num_entries / 2):
            try:
                os.remove(f'output/tmp/raw/MichalskiTrains.txt')
            except OSError:
                pass
            n_cars = random.randint(min_cars, max_cars)
            for _ in prolog.query(f"trains({n_cars})."):
                continue
            train = open('output/tmp/raw/MichalskiTrains.txt', 'r').read()
            t_angle = get_random_angle(with_occlusion)
            tmp = train.split(" ", 1)
            train = tmp[0] + f' {t_angle} ' + tmp[1]
            if 'east' in train and east_counter < int(num_entries / 2):
                all_trains.write(train)
                east_counter += 1
            elif 'west' in train and west_counter < int(math.ceil(num_entries / 2)):
                all_trains.write(train)
                west_counter += 1
            os.remove('output/tmp/raw/MichalskiTrains.txt')
        print(f'generated {west_counter} westbound trains and {east_counter} eastbound trains')
    os.remove(generator_tmp)


def parse_config(distribution_config): 
    attribute_choices = dict()      
    attribute_choices["length"] = ['short', 'long']
    attribute_choices["shape"] = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
    attribute_choices["double"] = ['not_double', 'double']
    attribute_choices["roof"] = ['none', 'arc', 'flat', 'jagged', 'peaked']
    attribute_choices["l_shape"] = ['rectangle', 'triangle', 'circle', 'diamond', 'hexagon', 'utriangle']
    if distribution_config is not None: 
        import json
        with open("distribution_configs/{}".format(distribution_config), "r") as json_file:
            json_file = json.load(json_file)
            attributes = ["angle", "length", "shape", "double", "roof", "l_shape"]
            for possible_attribute in attributes:
                try: 
                    new_attribute_list = json_file[possible_attribute]
                    assert isinstance(new_attribute_list, list)
                    for choice in new_attribute_list:
                        assert choice in attribute_choices[possible_attribute]
                    attribute_choices[possible_attribute] = new_attribute_list
                except: 
                    pass 
    return attribute_choices




def gen_raw_random_trains(class_rule, out_path, num_entries=10000, with_occlusion=False, min_cars=2, max_cars=4, distribution_config = None, rel_cars_path = None):
    """ Generate random trains descriptions
    Args:
        out_path: string path to save the generated train descriptions
        class_rule: str, classification rule used to derive the labels
        num_entries: int number of michalski trains which are generated
        with_occlusion: boolean whether to include occlusion of the train payloads
        max_cars: int maximum number of cars in a train
        min_cars: int minimum number of cars in a train
    """
    attribute_settings = parse_config(distribution_config) 
    print("Distribution settings are: {}".format(attribute_settings))
    classifier = 'output/tmp/raw/concept_tester_tmp.pl'
    os.makedirs('output/tmp/raw/', exist_ok=True)
    rule_path = f'example_rules/{class_rule}_rule.pl'

    try:
        os.remove(classifier)
    except OSError:
        pass
    with open("raw/train_generator.pl", 'r') as gen, open(rule_path, 'r') as rule:
        with open(classifier, 'w+') as generator:
            generator.write(gen.read())
            generator.write(rule.read())
    prolog = Prolog()
    prolog.consult(classifier)
    west_counter = 0
    east_counter = 0
    none_counter = 0
    try:
        os.remove(out_path)
    except OSError:
        pass
    os.makedirs(out_path.rsplit('/', 1)[0], exist_ok=True)


    #west and east attributes
    east_attrs = {
        "roof" : ["none", "peaked", "jagged"],
        "length": "long",
        "l_num": 2, 
        "wheels": "3"

    }

    west_attrs = {
        "roof" : ["flat", "arc"],
        "length": "short",
        "l_num": 1, 
        "wheels": "2"
    }

    dir_attrs = [east_attrs, west_attrs]

    def is_ok_func(dir_attrs, length:str, roof:str, l_num: int): 
        assert isinstance(l_num, int)
        length_ok = length != dir_attrs["length"]
        roof_ok = True
        for roof_val in dir_attrs["roof"]: 
            if roof == roof_val: 
                roof_ok = False
                break
        lnum_ok = l_num != dir_attrs["l_num"]  
        return length_ok or roof_ok or lnum_ok

    #to know which ones i need to care about inverening
    relevant_cars = []
    with open(out_path, 'w+') as text_file:
        while west_counter < int(math.floor(num_entries / 3)) or east_counter < int(math.floor(num_entries / 3)) or none_counter < int(math.floor(num_entries / 3)):
            t_angle = get_random_angle(with_occlusion)
            train = ''
            m_cars = f''

            num_cars = random.randint(min_cars, max_cars)
            assert num_cars == 3, "bruh something wrong"

            #NOTE: added for patricks data
            #dir in [east,west,none]
            random_dir_idx = random.randint(0, 2)
            car_num_label_idx = 1
            for j in range(num_cars):
                train += ', ' if len(train) > 0 else ''

                n = j + 1
                #only force attributes if we have west or east train
                if j == car_num_label_idx and random_dir_idx != 2:
                    needed_attrs = dir_attrs[random_dir_idx]
                    length = needed_attrs["length"]
                    roof = needed_attrs["roof"][random.randint(0, len(needed_attrs["roof"]))-1]
                    l_num = needed_attrs["l_num"]
                    wheels = needed_attrs["wheels"]
                #"none" direction
                else:
                    is_ok = False
                    while not is_ok:
                        length = random.choice(attribute_settings["length"])
                        roof = random.choice(attribute_settings["roof"])
                        if length == 'long':
                            #NOTE: P.V change
                            wheels = "3"
                            #l_num = 2
                            #wheels = random.choice(['2', '3'])
                            l_num = random.randint(2, 3)
                        else:
                            wheels = '2'
                            l_num = random.randint(0, 1)
                            #l_num = 1
                        is_ok = is_ok_func(east_attrs, length, roof, l_num) and is_ok_func(west_attrs, length, roof, l_num)

                shape = random.choice(attribute_settings["shape"])
                double = random.choice(attribute_settings["double"])
                l_shape = random.choice(attribute_settings["l_shape"])
                car = str(
                    n) + ' ' + shape + ' ' + length + ' ' + double + ' ' + roof + ' ' + wheels + ' ' + l_shape + ' ' + str(
                    l_num)
                train += f'c({str(n)}, {shape}, {length}, {double}, {roof}, {wheels}, l({l_shape}, {str(l_num)}))'

                if j != 0:
                    car = ' ' + car
                m_cars = m_cars + car
                # m_cars.append(michalski.MichalskiCar(n, shape, length, double, roof, wheels, l_num, l_shape))
            # m_trains.append(michalski.MichalskiTrain(m_cars, None))

            #NOTE: i change this such that we have 3 classes, this is a very hacky way tho
            if random_dir_idx == 0 and east_counter < int(num_entries / 3): 
                m_cars = f'{"east"} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                east_counter += 1
            elif random_dir_idx == 1 and west_counter < int(num_entries / 3): 
                m_cars = f'{"west"} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                west_counter += 1
            elif random_dir_idx == 2 and none_counter < int(num_entries / 3):                 
                m_cars = f'{"none"} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                none_counter += 1

            relevant_cars.append(1)
            """ q = list(prolog.query(f"eastbound([{train}])."))
            p = 'west' if len(q) == 0 else 'east'
            if p == 'east' and east_counter < int(num_entries / 2):
                relevant_cars.append(car_num_label_idx)
                m_cars = f'{p} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                east_counter += 1
            if p == 'west' and west_counter < int(math.ceil(num_entries / 2)):
                relevant_cars.append(car_num_label_idx)
                m_cars = f'{p} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                west_counter += 1 """
    print(f'generated {west_counter} westbound trains and {east_counter} eastbound trains and {none_counter} none trains')
    os.remove(classifier)

    #only write if we really have more than single cars
    if len(relevant_cars) > 0:
        try: 
            with open(rel_cars_path, "w") as write_file: 
                for idx in relevant_cars: 
                    write_file.write(str(idx) + '\n')
                print("Succesfully written relevant cars")
        except: 
            if len(relevant_cars) > 0: 
                raise ValueError("something wrong foo")



def gen_raw_trains(raw_trains, classification_rule, out_path, num_entries=10000, replace_existing=True,
                   with_occlusion=False, min_cars=2, max_cars=4, distribution_config = None, rel_cars_path = None):
    """ Generate random or Michalski train descriptions
    Args:
        raw_trains: string type of train which is generated available options: 'RandomTrains' and 'MichalskiTrains'
        out_path: string path to save the generated train descriptions
        classification_rule: str, path to classification rule used to derive the labels
        num_entries: int number of michalski trains which are generated
        replace_existing: bool whether the existing copy shall be replaced by a new copy
        with_occlusion: boolean whether to include occlusion of the train payloads
        max_cars: int maximum number of cars in a train
        min_cars: int minimum number of cars in a train
    """
    if min_cars > max_cars:
        raise ValueError(f'min_train_length {min_cars} is larger than max_train_length {max_cars}')
    if replace_existing:
        if raw_trains == 'RandomTrains':
            gen_raw_random_trains(classification_rule, out_path, num_entries, with_occlusion, min_cars, max_cars, distribution_config = distribution_config, rel_cars_path=rel_cars_path)
        elif raw_trains == 'MichalskiTrains':
            gen_raw_michalski_trains(classification_rule, out_path, num_entries, with_occlusion, min_cars, max_cars)


def read_trains(file, toSimpleObjs=False, scale=(.5, .5, .5)):
    """ read the trains generated by the prolog train generator
    Args:
        file: str directory from which the trains are loaded or list of trains which need to be loaded
        # num: int number of michalski trains which are read
        toSimpleObjs: if train is transformed to simple objects we need to update the pass indices accordingly
        scale: tuple scale of the train (x, y, z) if None each train is scaled individually to fit the scene
        (required space = number of cars + engine + free space)
    """
    lines = file
    m_trains = []
    if isinstance(file, str):
        with open(file, "r") as a:
            lines = a.readlines()
    for line in lines:
        m_cars = []
        l = line.split(' ')
        dir = l[0]
        t_angle = l[1]
        train_length = len(l) // 8
        if scale is not None:
            train_scale = scale
        else:
            #  scale the train to fit the scene (required space = number of cars + engine + free space)
            train_scale = 3 / (train_length + 2) if train_length > 4 else 0.5
        for c in range(train_length):
            
            ind = c * 8
            # a = (l[ind+i] for i in range(8))
            print("ATTRIBUTE START PER CAR")
            print("----------------------------------------")
            print("{}, {}, {}, {}, {}, {}, {}".format(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8]))
            print("----------------------------------------")
            car = BlenderCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                             l[ind + 9].strip('\n'), train_scale)
            if toSimpleObjs:
                car = SimpleCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                                l[ind + 9].strip('\n'), train_scale)
            
            m_cars.append(car)
        train = MichalskiTrain(m_cars, dir, t_angle, train_scale)
        if toSimpleObjs is True:
            train.update_pass_indices()
        # t_angle = get_random_angle(with_occlusion, angle)
        m_trains.append(train)
    return m_trains


def interevene_function(to_intervene_variable:str, variable_dict:dict, dir:str):

        intervene_dict = {
            "length": ["short", "long"], 
            "l_num": [0,1,2,3], 
            "shape": ["rectangle", "bucket", "ellipse"], 
            "double": ["double", "not_double"], 
            "l_shape": ["rectangle", "triangle", "circle", "diamond"], 
            "roof": ["none", "peaked", "jagged", "flat", "arc"], 
            "wheels": [2, 3]
        }
        
        new_values = deepcopy(variable_dict)


        
        #the current variable value of the intervened variable name
        variable_value = variable_dict[to_intervene_variable]
        #the list of possible values which can be intervened on
        corresponding_list = intervene_dict[to_intervene_variable]


        try:
            assert variable_value in corresponding_list
        except: 
            print(variable_dict)
            import ipdb; ipdb.set_trace()
        #interventions only on the non current value
        corresponding_list.remove(variable_value)

        #get a random new value for the intervened variable in the list of possible values
        rand_idx = random.randint(0, len(corresponding_list)-1)


        #the variable value after it was intervened on
        intervened_val = corresponding_list[rand_idx]
        try: 
            intervened_val = int(intervened_val)
        except: 
            pass 
        new_values[to_intervene_variable] = intervened_val

        new_dir = dir

        #if we intervene on those we need to change dir and maybe coupled variables
        if to_intervene_variable == "length" or to_intervene_variable == "roof" or to_intervene_variable == "l_num":
            if to_intervene_variable == "length": 
                if intervened_val == "long": 
                    intervened_numload_val = random.randint(2,3)
                    intervened_num_wheels = "2"
                elif intervened_val == "short": 
                    intervened_numload_val = random.randint(0,1)
                    intervened_num_wheels = "2"
                new_values["l_num"] = intervened_numload_val
                new_values["wheels"] = intervened_num_wheels
            #update direction
            is_east =  new_values["length"] == "long" and new_values["roof"] in ["none", "peaked", "jagged"] and new_values["l_num"] == 2
            is_west =  new_values["length"] == "short" and new_values["roof"] in ["arc", "flat"] and new_values["l_num"] == 1
            new_dir = "none"
            if is_east: 
                new_dir = "east"
            elif is_west: 
                new_dir = "west"

        ordered_list_names = ["shape", "length", "double", "roof", "wheels", "l_shape", "l_num"]
        old_vals = []
        order_list_vals = []
        for var_name in ordered_list_names:
            old_vals.append(variable_dict[var_name])
            order_list_vals.append(new_values[var_name])

        """ print("---------------------------------")
        print("Old vals: {}, dir: {}".format(old_vals, dir))
        print("New vals: {}, dir: {}".format(order_list_vals, new_dir))
        print("---------------------------------") """
        return order_list_vals, new_dir


def intervene_train(train, intervene_names, rel_car_idx):
    """Intervenes a single train object using the variables stated in intervened names, only accessing the car at rel_car_idx
    Returns a list of 1+len(intervene_names) train objects, where each train object is either the normal train (at 0) or an intervened train
    """
    all_trains = [train]
    car_list = train.get_cars()
    #we know this is 1 because we said so 
    #relevant_car = car_list[rel_car_idx]
    assert len(car_list) == 3, "some train without 3 cars"
    relevant_car = car_list[1]
    inst_cls = type(relevant_car)


    variable_names = ["shape", "length", "double", "roof", "wheels", "l_shape", "l_num"]


    dir = train.get_label()
    angle = train.get_angle()
    scale = train.get_blender_scale()

    car_num = relevant_car.get_car_number()
    car_shape = relevant_car.get_car_shape()
    car_length = relevant_car.get_car_length()
    car_roof = relevant_car.get_car_roof()
    car_wall = relevant_car.get_car_wall()
    car_wheels = relevant_car.get_wheel_count()
    car_load_num = relevant_car.get_load_number()
    car_l_shape = relevant_car.get_load_shape()

    variable_values = [car_shape, car_length, car_wall, car_roof, car_wheels, car_l_shape, car_load_num]
    variable_dict = {}
    #create dict of relevant car and its relevant values
    for i in range(len(variable_values)): 
        variable_dict[variable_names[i]] = variable_values[i]

    #actually perform train intervention
    for attr_name in intervene_names:
        #assumes that deepcopy means that the lists are also deepcopies of the train
        curr_train = deepcopy(train)
        curr_car_list = curr_train.get_cars()

        curr_attribute_list, curr_dir = interevene_function(attr_name, variable_dict, dir)
        curr_attribute_list = [car_num] + curr_attribute_list
        curr_car_list[rel_car_idx] = inst_cls(*curr_attribute_list)
        all_trains.append(MichalskiTrain(curr_car_list, curr_dir, angle, scale))

    

    """ intervened_train_shape = MichalskiTrain([inst_cls(car_num, intervener[car_shape], car_length, car_wall, car_roof, car_wheels, car_l_shape, car_load_num)], dir, angle, scale)    
    intervened_train_roof = MichalskiTrain([inst_cls(car_num, car_shape, car_length, car_wall, intervener[car_roof], car_wheels, car_l_shape, car_load_num)], intervened_shape_res_dir, angle, scale)    
    intervened_train_length = MichalskiTrain([inst_cls(car_num, car_shape, intervened_length_val, car_wall, car_roof, wheels, car_l_shape, new_car_num_load)], new_dir, angle, scale)     """
    return all_trains 
    


def read_trains_and_intervene(file, intervene_names, toSimpleObjs=False, scale=(.5, .5, .5), relevant_cars = None):
    lines = file
    m_trains = []

    if isinstance(file, str):
        with open(file, "r") as a:
            lines = a.readlines()

    
    for i, line in enumerate(lines):
        m_cars = []
        l = line.split(' ')
        dir = l[0]
        t_angle = l[1]
        train_length = len(l) // 8
        if scale is not None:
            train_scale = scale
        else:
            #  scale the train to fit the scene (required space = number of cars + engine + free space)
            train_scale = 3 / (train_length + 2) if train_length > 4 else 0.5
        for c in range(train_length):
            
            ind = c * 8
            # a = (l[ind+i] for i in range(8))
            """ print("ATTRIBUTE START PER CAR")
            print("----------------------------------------")
            print("{}, {}, {}, {}, {}, {}, {}".format(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8]))
            print("----------------------------------------") """
            car = BlenderCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                             l[ind + 9].strip('\n'), train_scale)
            if toSimpleObjs:
                car = SimpleCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                                l[ind + 9].strip('\n'), train_scale)
            m_cars.append(car)
        train = MichalskiTrain(m_cars, dir, t_angle, train_scale)
        if toSimpleObjs is True:
            train.update_pass_indices()
        #t_angle = get_random_angle(with_occlusion, angle)
        train = intervene_train(train, intervene_names, rel_car_idx=relevant_cars[i])
        m_trains.append(train)
    return m_trains



def get_random_angle(with_occlusion, angle=None):
    """ randomly sample an angle of the train
    Args:
        with_occlusion: boolean whether to include occlusion of the train payloads
        angle: int fixed angle, None to sample a new angle value
    """
    if angle is not None:
        train_dir = angle
    else:
        allowed_deg = [-70, 70] if with_occlusion else [-60, 60]
        train_dir = random.randint(allowed_deg[0], allowed_deg[1]) + 180 * random.randint(0, 1)
    return train_dir


def generate_m_train_attr_overview(file):
    """ Generates an overview of the train descriptions used by the train generator
    Args:
        file: string path to trains which shall be analysed
    """
    double = []
    l_num = []
    l_shape = []
    length = []
    car_position = []
    roof = []
    shape = []
    wheels = []
    trains = read_trains(file)
    for train in trains:
        for car in train.m_cars:
            if car.double not in double:
                double.append(car.double)
            if car.l_num not in l_num:
                l_num.append(car.l_num)
            if car.l_shape not in l_shape:
                l_shape.append(car.l_shape)
            if car.length not in length:
                length.append(car.length)
            if car.n not in car_position:
                car_position.append(car.n)
            if car.roof not in roof:
                roof.append(car.roof)
            if car.shape not in shape:
                shape.append(car.shape)
            if car.wheels not in wheels:
                wheels.append(car.wheels)

    with open("old/class_att", "w+") as text_file:
        text_file.write("double values:\n %s\n" % double)
        text_file.write("load numbers:\n %s\n" % l_num)
        text_file.write("load shapes:\n %s\n" % l_shape)
        text_file.write("length:\n %s\n" % length)
        text_file.write("car positions:\n %s\n" % car_position)
        text_file.write("roofs:\n %s\n" % roof)
        text_file.write("shapes:\n %s\n" % shape)
        text_file.write("wheels:\n %s\n" % wheels)
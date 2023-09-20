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




def gen_raw_random_trains(class_rule, out_path, num_entries=10000, with_occlusion=False, min_cars=2, max_cars=4, distribution_config = None):
    """ Generate random trains descriptions
    Args:
        out_path: string path to save the generated train descriptions
        class_rule: str, classification rule used to derive the labels
        num_entries: int number of michalski trains which are generated
        with_occlusion: boolean whether to include occlusion of the train payloads
        max_cars: int maximum number of cars in a train
        min_cars: int minimum number of cars in a train
    """
    distribution_settings = parse_config(distribution_config) 
    print("Distribution settings are: {}".format(distribution_settings))
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
    try:
        os.remove(out_path)
    except OSError:
        pass
    os.makedirs(out_path.rsplit('/', 1)[0], exist_ok=True)
    with open(out_path, 'w+') as text_file:
        while west_counter < int(math.ceil(num_entries / 2)) or east_counter < int(num_entries / 2):
            t_angle = get_random_angle(with_occlusion)
            train = ''
            m_cars = f''

            num_cars = random.randint(min_cars, max_cars)
            for j in range(num_cars):
                train += ', ' if len(train) > 0 else ''

                n = j + 1
                length = random.choice(distribution_settings["length"])

                if length == 'long':
                    #NOTE: P.V change
                    wheels = "3"
                    l_num = 2
                    #wheels = random.choice(['2', '3'])
                    #l_num = random.randint(0, 3)
                else:
                    wheels = '2'
                    #l_num = random.randint(0, 2)
                    l_num = 1

                shape = random.choice(distribution_settings["shape"])
                double = random.choice(distribution_settings["double"])
                roof = random.choice(distribution_settings["roof"])
                l_shape = random.choice(distribution_settings["l_shape"])
                car = str(
                    n) + ' ' + shape + ' ' + length + ' ' + double + ' ' + roof + ' ' + wheels + ' ' + l_shape + ' ' + str(
                    l_num)
                train += f'c({str(n)}, {shape}, {length}, {double}, {roof}, {wheels}, l({l_shape}, {str(l_num)}))'

                if j != 0:
                    car = ' ' + car
                m_cars = m_cars + car
                # m_cars.append(michalski.MichalskiCar(n, shape, length, double, roof, wheels, l_num, l_shape))
            # m_trains.append(michalski.MichalskiTrain(m_cars, None))
            q = list(prolog.query(f"eastbound([{train}])."))
            p = 'west' if len(q) == 0 else 'east'
            if p == 'east' and east_counter < int(num_entries / 2):
                m_cars = f'{p} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                east_counter += 1
            if p == 'west' and west_counter < int(math.ceil(num_entries / 2)):
                m_cars = f'{p} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                west_counter += 1
    print(f'generated {west_counter} westbound trains and {east_counter} eastbound trains')
    os.remove(classifier)


def gen_raw_trains(raw_trains, classification_rule, out_path, num_entries=10000, replace_existing=True,
                   with_occlusion=False, min_cars=2, max_cars=4, distribution_config = None):
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
            gen_raw_random_trains(classification_rule, out_path, num_entries, with_occlusion, min_cars, max_cars, distribution_config = distribution_config)
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
            "l_num": [1, 2], 
            "shape": ["rectangle", "bucket"], 
            "double": ["double", "not_double"], 
            "l_shape": ["rectangle", "triangle"], 
            "roof": ["arc", "none"], 
            "wheels": [2, 3]
        }
        
        new_values = deepcopy(variable_dict)


        
        variable_value = variable_dict[to_intervene_variable]
        corresponding_list = intervene_dict[to_intervene_variable]

        assert variable_value in corresponding_list
        corresponding_list.remove(variable_value)

        intervened_val = corresponding_list[0]
        try: 
            intervened_val = int(intervened_val)
        except: 
            pass 
        new_values[to_intervene_variable] = intervened_val

        new_dir = dir
        if to_intervene_variable == "length" or to_intervene_variable == "roof":
            new_dir = "east" if (new_values["length"] == "long" and new_values["roof"] == "arc") else "west"
            if to_intervene_variable == "length": 

                num_load_val = variable_dict["l_num"]
                num_wheels_val = variable_dict["wheels"]


                #get intervened num_load
                intervened_numload_val = intervene_dict["l_num"]
                intervened_numload_val.remove(num_load_val)
                intervened_numload_val = intervened_numload_val[0]
                new_values["l_num"] = intervened_numload_val


                #same for wheel val
                intervened_numwheels_val = intervene_dict["wheels"]
                intervened_numwheels_val.remove(num_wheels_val)
                intervened_numwheels_val = intervened_numwheels_val[0]
                new_values["wheels"] = intervened_numwheels_val
            


        ordered_list_names = ["shape", "length", "double", "roof", "wheels", "l_shape", "l_num"]
        old_vals = []
        order_list_vals = []
        for var_name in ordered_list_names:
            old_vals.append(variable_dict[var_name])
            order_list_vals.append(new_values[var_name])

        print("---------------------------------")
        print("Old vals: {}, dir: {}".format(old_vals, dir))
        print("New vals: {}, dir: {}".format(order_list_vals, new_dir))
        print("---------------------------------")
        return order_list_vals, new_dir


def intervene_train(train, intervene_names):
    assert len(train.get_cars()) == 1, "more than one car in data"
    car = train.get_cars()[0]

    inst_cls = type(car)
    #interventions are: 

    #car_shape: rectangle => bucket
    #roof: arc => none
    #direction = west = (length == long) && (shape == rectangle)
    #length: short => long (sample car_load_num and update direction); long => short (same)

    #return train = [normal_train, intervened_shape, intervened_roof, intervened_length]
    
    dir = train.get_label()
    angle = train.get_angle()
    scale = train.get_blender_scale()

    car_num = car.get_car_number()
    car_shape = car.get_car_shape()
    car_length = car.get_car_length()
    car_roof = car.get_car_roof()
    car_wall = car.get_car_wall()
    car_wheels = car.get_wheel_count()
    car_load_num = car.get_load_number()
    car_l_shape = car.get_load_shape()

    variable_names = ["shape", "length", "double", "roof", "wheels", "l_shape", "l_num"]
    variable_values = [car_shape, car_length, car_wall, car_roof, car_wheels, car_l_shape, car_load_num]
    variable_dict = {}
    for i in range(len(variable_values)): 
        variable_dict[variable_names[i]] = variable_values[i]



    all_trains = [train]
    for attr_name in intervene_names:
        curr_attribute_list, curr_dir = interevene_function(attr_name, variable_dict, dir)
        curr_attribute_list = [car_num] + curr_attribute_list
        all_trains.append(MichalskiTrain([inst_cls(*curr_attribute_list)], curr_dir, angle, scale))

    

    """ intervened_train_shape = MichalskiTrain([inst_cls(car_num, intervener[car_shape], car_length, car_wall, car_roof, car_wheels, car_l_shape, car_load_num)], dir, angle, scale)    
    intervened_train_roof = MichalskiTrain([inst_cls(car_num, car_shape, car_length, car_wall, intervener[car_roof], car_wheels, car_l_shape, car_load_num)], intervened_shape_res_dir, angle, scale)    
    intervened_train_length = MichalskiTrain([inst_cls(car_num, car_shape, intervened_length_val, car_wall, car_roof, wheels, car_l_shape, new_car_num_load)], new_dir, angle, scale)     """
    return all_trains 
    


def read_trains_and_intervene(file, intervene_names, toSimpleObjs=False, scale=(.5, .5, .5)):
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
        # t_angle = get_random_angle(with_occlusion, angle)
        train = intervene_train(train, intervene_names)
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
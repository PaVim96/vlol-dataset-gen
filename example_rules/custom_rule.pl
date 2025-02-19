% A costum rule:
% The costum classification rule can be adjusted according the requirements. Feel free to make any adjustments needed.
% The classification rule must be specified in the Prolog description language using the defined predicates (see README).
% New predicates can also be defined in below (structure of the trains can be found in train_generator.pl file)

#eastbound(Train):- has_car(Train,Car), load_num(Car,N), car_num(Car,N), has_wheel0(Car,N).
#eastbound(Train):- has_car(Train, Car), rectangle(Car), long(Car).
eastbound(Train):- has_car(Train, Car), long(Car), has_roof2(Car, none), load_num(Car, 2).
eastbound(Train):- has_car(Train, Car), long(Car), has_roof2(Car, peaked), load_num(Car, 2).
eastbound(Train):- has_car(Train, Car), long(Car), has_roof2(Car, jagged), load_num(Car, 2).
#eastbound(Train):- has_car(Train, Car), long(Car), has_roof2(Car, flat), load_num(Car, 2).
import json

from blender_image_generator.blender_util import enable_gpus, clean_up
from blender_image_generator.compositor import create_tree
from blender_image_generator.load_assets import *
from blender_image_generator.json_util import restore_img, restore_depth_map
from blender_image_generator.load_simple_assets import create_simple_scene, load_simple_engine
from util import *
import time

def generate_image(class_rule, base_scene, raw_trains, train_vis, t_num, train, ds_name=None,
                   save_blender=False, replace_existing_img=True,
                   high_res=False, gen_depth=False, min_cars=2, max_cars=4, train_name = ""):
    """ assemble a michalski train, render its corresponding image and generate ground truth information
    Args:
    :param:  base_scene (string)            : background scene of the train ('base_scene', 'desert_scene', 'sky_scene',
     'fisheye_scene')
    :param:  raw_trains (string)            : typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
    :param:  train_vis (string)             : visualization of the train description either 'MichalskiTrains' or
    'SimpleObjects'
    :param:  t_num (int)                    : id of the train
    :param:  train (train obj)              : train object which is assembled and rendered
    :param:  save_blender (bool)            : whether the blender scene shall be shaved
    :param:  replace_existing_img (bool)    : if there exists already an image for the id shall it be replaced?
    :param:  gen_depth (bool)               : whether to generate the depth information of the individual scenes
    :param:  high_res (bool)                : whether to render the images in high resolution (1920x1080) or standard
     resolution (480x270)
    """

    start = time.time()
    path_settings = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_{min_cars}-{max_cars}' if ds_name is None else ds_name
    output_image = f'output/tmp/image_generator/{path_settings}/images/{t_num}_m_train{train_name}.png'
    output_blendfile = f'output/tmp/image_generator/{path_settings}/blendfiles/{t_num}_m_train{train_name}.blend'
    output_scene = f'output/tmp/image_generator/{path_settings}/scenes/{t_num}_m_train{train_name}.json'
    output_depth_map = f'output/tmp/image_generator/{path_settings}/depths/{t_num}_m_train{train_name}.png'
    if os.path.isfile(output_image) and os.path.isfile(output_scene) and (os.path.isfile(
            output_depth_map) or not gen_depth) and not replace_existing_img:
        return
    os.makedirs(f'output/tmp/image_generator/{path_settings}/images', exist_ok=True)
    os.makedirs(f'output/tmp/image_generator/{path_settings}/blendfiles', exist_ok=True)
    os.makedirs(f'output/tmp/image_generator/{path_settings}/scenes', exist_ok=True)
    os.makedirs(f'output/tmp/image_generator/{path_settings}/depths', exist_ok=True)

    # collection = 'base_scene'
    # load_base_scene(filepath, collection)
    # reset scene
    # add all base scene assets
    filepath = f'data/scenes/{base_scene}.blend'
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=filepath)

    enable_gpus("CUDA")

    # render settings
    rn_scene = bpy.context.scene
    rn_scene.render.image_settings.file_format = 'PNG'
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"

    render_args.resolution_x, render_args.resolution_y = 1920, 1080
    # render_args.tile_x, render_args.tile_y = 256, 256
    if high_res:
        render_args.resolution_percentage = 100
    else:
        render_args.resolution_percentage = 15
    # bpy.data.worlds['World'].cycles.sample_as_light = True
    rn_scene.cycles.blur_glossy = 2.0
    rn_scene.cycles.max_bounces = 20
    rn_scene.cycles.samples = 512
    rn_scene.cycles.transparent_min_bounces = 8
    rn_scene.cycles.transparent_max_bounces = 8

    # load materials
    load_materials()

    # determine train direction and initial coordinates
    # rotate train in random direction
    # degrees between 240-280 and 60 - 120 are excluded for no occlusion
    # with occlusion extends the allowed degrees by 10 degrees for each direction

    train_dir = train.get_angle()
    alpha = math.radians(train_dir)

    # set scale of the train
    # train.set_scale(0.5)

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'base_scene': base_scene,
        'train_description': raw_trains,
        'classification_rule': class_rule,
        'visualization': train_vis,
        'min_cars': min_cars,
        'max_cars': max_cars,
        'image_index': t_num,
        'image_filename': os.path.basename(output_image),
        'blender_filename': os.path.basename(output_blendfile),
        'depth_map_filename': os.path.basename(output_depth_map),
        'angle': train_dir,
        # 'm_train': train.toJSON(),
        'train': train.to_txt(),
        'car_masks': {},
    }

    # create blender collection for the train which is to be created
    collection = 'train'
    train_collection = bpy.data.collections.new(collection)
    bpy.context.scene.collection.children.link(train_collection)
    layer_collection = bpy.context.view_layer.layer_collection.children[train_collection.name]
    layer_collection.exclude = False

    # determine train length and the starting point as a radius distance r for the engine
    loc_length = 0
    for car in train.m_cars:
        loc_length += car.get_car_length_scalar()
    # move rotation point away from camera for space efficiency
    init_cord = [0, -0.1, 0]
    if train_vis == 'SimpleObjects':
        displacement = .4 * train.get_blender_scale()[0]
        # add engine length and car displacements to the radius
        r = (loc_length + train.get_car_length('simple_engine') + displacement * len(train.m_cars)) / 2
        # determine engine spawn position (which is located in the middle of the engine)
        offset = train.get_car_length('simple_engine') / 2  # - 0.675 * train.get_blender_scale()[0]
        engine_pos = - r + offset

        # initialize train position
        engine_pos = get_new_pos(init_cord, engine_pos, alpha)

        load_simple_engine(train_collection, engine_pos, alpha, train.get_blender_scale())
        # create and load trains into blender
        create_simple_scene(train, train_collection, engine_pos, alpha)
    else:
        # add engine length to radius
        r = (loc_length + train.get_car_length('engine')) / 2
        # determine offset to engine spawn position (which is located at the end of the engine)
        offset = train.get_car_length('engine') - 0.675 * train.get_blender_scale()[0]
        engine_pos = - r + offset
        # load train at scale 1, z = -0.307
        init_cord[2] = -0.307 * train.get_blender_scale()[0]
        # move rotation point away from camera
        train_init_cord = get_new_pos(init_cord, engine_pos, alpha)
        # xd = engine_pos * math.cos(alpha) + offset[0]
        # yd = engine_pos * math.sin(alpha) + offset[1]
        # train_init_cord = [xd, yd, off_z]

        # load engine
        # use mat='black_metal' for black engine metal
        mat = None
        load_engine(train_collection, train_init_cord, alpha, mat, scale=train.get_blender_scale())
        # load rails at z = 0
        rail_cord = [0, -0.1, 0]
        load_rails(train_collection, rail_cord, alpha, base_scene, scale=train.get_blender_scale())
        # create and load trains into blender
        create_train(train, train_collection, train_init_cord, alpha)

    load_obj_time = time.time()
    # print('time needed pre set up: ' + str(load_obj_time - start))
    rail_time = time.time()
    # print('time needed rails: ' + str(rail_time - load_obj_time))

    asset_time = time.time()
    # print('time needed asset: ' + str(asset_time - rail_time))
    # delete duplicate materials
    clean_up()
    assets_time = time.time()
    # print('time needed load assets: ' + str(assets_time - load_obj_time))

    create_tree(train, t_num, gen_depth, path_settings)
    tree_time = time.time()

    # print('time needed tree: ' + str(tree_time - asset_time))

    rn_scene.render.filepath = output_image

    # print('time needed for compositor: ' + str(setup_end - load_obj_time))
    bpy.ops.render.render(write_still=1)

    render_time = time.time()

    # print('time needed for render: ' + str(render_time - tree_time))

    obj_mask = restore_img(train, t_num, path_settings)

    scene_struct['car_masks'] = obj_mask

    if gen_depth:
        restore_depth_map(t_num, output_depth_map, path_settings)

    if save_blender:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(output_blendfile))

    with open(output_scene, 'w+') as f:
        json.dump(scene_struct, f, indent=2)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    fin_time = time.time()

    # print('finish it time: ' + str(fin_time - render_time))

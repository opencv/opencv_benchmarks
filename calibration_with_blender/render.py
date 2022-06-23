import bpy
from bpy_extras.object_utils import world_to_camera_view
import os
import random
from math import pi
import pathlib
import json


def check_projection(cam, obj):
    scene = bpy.context.scene
    render = scene.render

    # print(obj.data.vertices[0].co, "- Vert 0 (original)")

    # Convert vertices to mesh
    me = obj.to_mesh()
    # print(me.vertices[0].co, " - Vert 0 (deformed/modified)")

    # Transform mesh according to translation and rotation
    me.transform(obj.matrix_world)
    # print(me.vertices[0].co, " - Vert 0 (deformed/modified, world space)")

    # Collect mesh coordinates
    verts = [vert.co for vert in me.vertices]
    print(list(verts))

    # Convert to normalized device coordinates
    coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]

    # x, y must be in [0, 1], z > 0
    for x, y, z in coords_2d:
        print(x, y, z)
        if x < 0 or x > 1:
            return False
        if y < 0 or y > 1:
            return False
        if z <= 0:
            return False

    return True


def set_position_origin(obj):
    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    obj.rotation_euler[0] = 0
    obj.rotation_euler[1] = 0
    obj.rotation_euler[2] = 0


def hide_objects(names):
    for name in names:
        bpy.data.objects[name].location.z = 10


if __name__ == '__main__':
    # Get camera
    c = bpy.data.objects['Camera']

    # Set camera intrincs
    # c.lens_unit = 'MILLIMETERS'
    # c.lens = 20 # focus length

    # Set camera position
    set_position_origin(c)
    c.location.z = 0.5

    # Settings
    work_dir = '/home/xperience/development/datasets'
    patterns = ['checkerboard', 'circles', 'radon']
    N = 600  # Number of genrated images

    for pattern in patterns:
        hide_objects(patterns)

        p = bpy.data.objects[pattern]
        set_position_origin(p)

        # Fix random seed
        random.seed(1)

        # Create pattern directory
        pattern_dir = os.path.join(work_dir, pattern)
        pathlib.Path(pattern_dir).mkdir(parents=True, exist_ok=True)

        dataset_info = {}
        camera_info = {'focus_length': 50}
        renders_info = {}

        n = 0
        i = 0

        while n < N and i < 10000000:
            # Set position
            p.location.x = random.uniform(-0.3, 0.3)
            p.location.y = random.uniform(-0.2, 0.2)
            p.location.z = random.uniform(0, 0.15)

            # Set rotation
            p.rotation_euler[0] = random.uniform(-0.5, 0.5)
            p.rotation_euler[1] = random.uniform(-0.5, 0.5)
            # p.rotation_euler[2] = random.uniform(0.7 * pi/2, 1.3 * pi/2)
            p.rotation_euler[2] = random.uniform(-0.5, 0.5)

            # Update matrices
            bpy.context.view_layer.update()

            # Debug
            print('>', i, ' ', n, '<')
            print(p.location)
            print(p.rotation_euler)

            # Render and save image if it fully visible
            if check_projection(c, p):
                print('True')

                render_filename = 'render-{:04d}.jpg'.format(n)
                render_path = os.path.join(pattern_dir, render_filename)

                bpy.context.scene.render.filepath = render_path
                bpy.ops.render.render(write_still=True)

                renders_info[render_filename] = {'pattern':
                                                     {'location': [p.location.x, p.location.y, p.location.z],
                                                      'rotation': [p.rotation_euler[0], p.rotation_euler[1],
                                                                   p.rotation_euler[2]]
                                                      }
                                                 }
                n += 1

            else:
                print('False')

            i += 1

        dataset_info['camera'] = camera_info
        dataset_info['renders'] = renders_info

        with open(os.path.join(pattern_dir, 'info.json'), 'w') as info_file:
            json.dump(dataset_info, info_file, indent=4)

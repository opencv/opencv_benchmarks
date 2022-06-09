import bpy
from bpy_extras.object_utils import world_to_camera_view
import random
import time


def check_projection(cam, obj):
    scene = bpy.context.scene
    render = scene.render
    
    #print(obj.data.vertices[0].co, "- Vert 0 (original)")  
    
    # Convert vertices to mesh
    me = obj.to_mesh()
    #print(me.vertices[0].co, " - Vert 0 (deformed/modified)")

    # Transform mesh according to translation and rotation
    me.transform(obj.matrix_world)
    #print(me.vertices[0].co, " - Vert 0 (deformed/modified, world space)")


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
    obj.location.z = 2

    obj.rotation_euler[0] = 0
    obj.rotation_euler[1] = 0
    obj.rotation_euler[2] = 0



# Get camera
c = bpy.data.objects['Camera']

# Set camera intrincs
c.lens_unit = 'MILLIMETERS'
c.lens = 20 # focus length

# Set camera position
set_position_origin(c)
c.location.z = 1


# Set pattern init position
p = bpy.data.objects['checkerboard']

set_position_origin(p)

N = 30 # Number of genrated images
n = 0
i = 0

random.seed(1)

while n < N and i < 1000:
    # Set position
    p.location.x = random.uniform(-0.4, 0.4)
    p.location.y = random.uniform(-0.4, 0.4)
    p.location.z = random.uniform(-0.2, 0.2)
    
    # Set rotation
    p.rotation_euler[0] = random.uniform(0, 0.5)
    p.rotation_euler[1] = random.uniform(0, 0.5)
    p.rotation_euler[2] = random.uniform(0, 0.5)
    
    # Update matrices
    bpy.context.view_layer.update()
    
    # Debug
    print('>',i,'<')
    print(p.location)
    print(p.rotation_euler)
    
    # Render and save image if it fully visible
    if check_projection(c, p):
        print('True')
        bpy.context.scene.render.filepath = '/tmp/render-{:03d}.jpg'.format(i)
        bpy.ops.render.render(write_still=True)
        n += 1

    else:
        print('False')
        
    i +=1 

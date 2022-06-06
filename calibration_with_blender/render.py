import bpy

c = bpy.data.objects[0]

c.location.x = 0
c.location.y = 0
c.location.z = 8

c.rotation_euler[0] = 0
c.rotation_euler[1] = 0
c.rotation_euler[2] = 0

p = bpy.data.objects[2]

p.location.x = 0
p.location.y = 0
p.location.z = 0

p.rotation_euler[0] = 0
p.rotation_euler[1] = 0
p.rotation_euler[2] = 0

i = 0
for a in [0, 0.25, 0.5]:
    p.rotation_euler[0] = a
    for x in [-1, 0, 1]:
        c.location.x = x

        for y in [-2, -1, 0, 1, 2]:
            c.location.y = y

            bpy.context.scene.render.filepath = '/tmp/render-{:03d}.jpg'.format(i)
            bpy.ops.render.render(write_still=True)

            i += 1

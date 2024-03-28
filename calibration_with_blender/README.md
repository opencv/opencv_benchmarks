# Calibration with blender
### Create synth images with calibration pattern by blender:
* Open calibration.blend in blender
* Copy render.py content to blender text editor
* Run script in blender
* Distort images by image_distort.cpp

### Pattern generation command
gen_pattern.py -c 14 -r 19 -T checkerboard -u px -s 220 -w 3508 -h 4961 && convert out.svg checkerboard.png

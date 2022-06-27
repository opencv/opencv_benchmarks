
To run the benchmark:
1. Download boofcv dataset (https://boofcv.org/notwiki/regression/fiducial/qrcodes_v3.zip) to path_folder.
2. Install opencv-python.
3. Use "python qr.py -alg opencv -o out.yaml -p path_folder/qrcodes/detection" to run benchmark.


To run opencv wechat algorithm:
1. Follow the steps from the previous part.
2. Install opencv-contrib-python.
3. Build OpenCV with OpenCV contrib (https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)
4. Find files detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel and set path_to_model.
5. Use "python qr.py -alg opencv_wechat -o out.yaml -p path_folder/qrcodes/detection" to run benchmark.

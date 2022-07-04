The benchmark aims to check QR codes detection and decoding with classic OpenCV implementation from
objdetect module and dnn based implementation in opencv_contrib repo contributed by the WeChat team.

To run the benchmark:
1. Download [BoofCV dataset](https://boofcv.org/notwiki/regression/fiducial/qrcodes_v3.zip) to path_folder. The dataset
has been prepared by the community BoofCV: [BoofCV main page](http://boofcv.org/index.php?title=Main_Page).
2. Install [opencv-python](https://pypi.org/project/opencv-python/) or build OpenCV.
3. Use "python qr.py -alg opencv -o out.yaml -p path_folder/qrcodes/detection" to run benchmark.


To run OpenCV and WeChat algorithm:
1. Follow the steps from the previous part.
2. Install [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/) or
[build OpenCV with OpenCV contrib]((https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)).
3. [Download files](https://github.com/WeChatCV/opencv_3rdparty) detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel and set path_to_model.
4. Use "python qr.py -alg opencv_wechat -o out.yaml -p path_folder/qrcodes/detection" to run benchmark.

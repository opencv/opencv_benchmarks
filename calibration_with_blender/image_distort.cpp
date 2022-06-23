#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

int main(int argc, char* argv[])
{
    if (argc < 8)
    {
        std::cout << "usage; " << argv[0] << "image camera_model fx fy cx cy [d0 .. dn] output"
                  << std::endl;
        return EXIT_FAILURE;
    }

    /* Load original image */
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty())
    {
        std::cout << "Could not open " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    cv::imshow("original", image);
    cv::waitKey(10);

    /* Construct K */
    cv::Mat_<float> camera_matrix = cv::Mat::eye(3, 3, CV_32F);
    camera_matrix(0, 0) = std::stof(argv[3]);
    camera_matrix(1, 1) = std::stof(argv[4]);
    camera_matrix(0, 2) = std::stof(argv[5]) == 0 ? image.size().width / 2.f: std::stof(argv[5]);
    camera_matrix(1, 2) = std::stof(argv[6]) == 0 ? image.size().height / 2.f: std::stof(argv[6]);

    std::cout << camera_matrix << std::endl;

    /* Load distortion coefficient */
    cv::Mat distortion = cv::Mat::zeros(1, 5, CV_32F);
    for (size_t i = 7; i < argc - 1; ++i)
    {
        distortion.at<float>(0, i - 7) = std::stof(argv[i]);
    }

    std::cout << distortion << std::endl;

    /* Collect original point location */
    std::vector<cv::Point2f> image_points;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            image_points.emplace_back(j, i);
        }
    }

    /* Since remap make inverse operation, for distort image we need undistort original point position */
    cv::Mat_<cv::Point2f> undistorted_points(image.size());
    cv::undistortPoints(image_points, undistorted_points, camera_matrix, distortion,
                        cv::noArray(),
                        camera_matrix);

    /* 2-channel (x & y), shape equal to result image */
    undistorted_points = undistorted_points.reshape(2, image.rows);

    std::cout << undistorted_points.size() << std::endl;

    /* Fill result image */
    cv::Mat distorted;
    cv::remap(image, distorted, undistorted_points, cv::noArray(), cv::INTER_LANCZOS4);

    /* Save result */
    cv::imshow("distorted", distorted);
    cv::waitKey(0);
    cv::imwrite(argv[argc - 1], distorted);
}
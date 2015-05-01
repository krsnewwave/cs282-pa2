/**
 * @author Dylan Valerio
 * @date May 1, 2015
 * @description This file contains solutions to the Programming Assignment
 *          2, Image Stitching for Computer Vision Class (CS282).
 * 
 * Please use like so:
 * ./pa2 [image 1] [image 2] [optional: patch size]
 * 
 * @summary The main class begins with reading the images, converting to
 *          gray scale and then equalizing the histograms. Afterwards, corners
 *          are detected using the Harris corner detection algorithm. Patches
 *          of fixed size are extracted from each detected corner. Features
 *          are created from these patches by flattening the patches to 1-row
 *          OpenCV Mat objects. For each feature, the best match on the other
 *          group of patches are computed using the Euclidean Algorithm.
 *          RANSAC was then used to determine the inlier points, and to 
 *          create a Homography matrix. This homography matrix was used to
 *          warp and stitch the two images together. Statistics like the number
 *          of inliers and average residuals are computed.
 * 
 * @build To build this, please use CMAKE. Simply run "cmake ." then "make
 *          all". The solution is divided into two classes, main.cpp and
 *          utils.cpp. There is also the main3.cpp, which correspond to
 *          Programming Assignment 3. Lastly, a bonus.cpp file is included as
 *          reference to another solution using SURF features.
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "utils.cpp"

using namespace cv;
using namespace std;

// Global variables
Mat src_1, src_2,
img1, img2, filtered_coords;

//variables for the callbacks
int thresh = 150;
int max_thresh = 255;
int ransac_thresh = 40;
int topX = 50;
int width = 20;

//named window variables
char* source_window = "Source image";
char* corners_window = "Corners detected";
char* matches_window = "Matches found";
char* warped_window = "Ransac: Warped Window";
char* composite_window = "Ransac: Composite Window";

//typedef variables for the keypoints and key features
vector<Point2f> leftImagePoints, rightImagePoints;
Coordinates corners_1, corners_2;
ScoreMap scoreMap;
BFScoreMap sortedMap;

/// Function headers
void demoKeypoints(int, void*);
void demoRansac(int, void*);
void plotMatches(int, void*);
void showCorners(Mat src, Coordinates coordinates, char* window_name);
PairKeyPoints myAlgorithm();
PairKeyPoints bestFirstMatchAlgorithm();
void plotBFMatches(int, void*);
bool isZero(Vec3b point);
Mat compositeResult(Mat left, Mat right);

/** @function main */
int main(int argc, char** argv) {
    /// Load source image and convert it to gray
    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    width = argc > 3 ? atoi(argv[3]) : 20;

    Mat gray_im1, gray_im2;
    cvtColor(img1, gray_im1, CV_BGR2GRAY);
    cvtColor(img2, gray_im2, CV_BGR2GRAY);

    equalizeHist(gray_im1, src_1);
    equalizeHist(gray_im2, src_2);

    /// Create a window and a trackbar
    namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, demoKeypoints);

    demoKeypoints(0, 0);
    waitKey(0);
    destroyWindow(source_window);

    cout << "Extracting corners" << endl;

    corners_1 = extractFeaturePoints(src_1, thresh);
    corners_2 = extractFeaturePoints(src_2, thresh);

    PairKeyPoints keyPoints = bestFirstMatchAlgorithm();
    cout << "Size of keypoints: " << keyPoints.first.size() << endl;
    cout << "Size of image 1: " << img1.size().width << "x"
            << img1.size().height << endl;
    cout << "Size of image 2: " << img2.size().width << "x"
            << img2.size().height << endl;

    leftImagePoints = keyPoints.first;
    rightImagePoints = keyPoints.second;

    //plot stitched image
    namedWindow(warped_window, CV_WINDOW_NORMAL);
    createTrackbar("Ransac threshold: ",
            warped_window, &ransac_thresh, 300, demoRansac);
    waitKey(0);
    return 0;
}

/**
 * This algorithm uses opencv's BFMatcher, as opposed to MyAlgorithm, which uses
 * my own hand-crafted matcher function.
 * @return PairKePoints - vectors of the matching keypoints, indexed equally
 */
PairKeyPoints bestFirstMatchAlgorithm() {
    cout << "Executing opencv's BFMatcher algorithm" << endl;
    Patches patches_1 = extractPatches(corners_1, src_1, width);
    Patches patches_2 = extractPatches(corners_2, src_2, width);

    //flatten the patches vector to just one Mat
    Mat left, right;
    for (int i = 0; i < patches_1.size(); i++) {
        left.push_back(patches_1.at(i));
    }
    for (int i = 0; i < patches_2.size(); i++) {
        right.push_back(patches_2.at(i));
    }

    vector<DMatch> bfMatches;
    BFMatcher matcher = BFMatcher(NORM_L2);
    matcher.match(left, right, bfMatches);

    sortedMap = createScoreMap(bfMatches);

    //good matches according to topX
    namedWindow(matches_window, CV_WINDOW_NORMAL);
    createTrackbar("Best %: ", matches_window, &topX, 100, plotBFMatches);
    waitKey(0);
    destroyWindow(matches_window);

    //now filter with the same topX
    BFScoreMap filteredMap = filterScores(sortedMap, topX);

    vector<Point2f> leftPoints, rightPoints;
    //get the points from the score map
    for (map<double, tuple<int, int>>::iterator it = filteredMap.begin();
            it != filteredMap.end(); ++it) {
        tuple<int, int> coord = it->second;
        int x = get<0>(coord);
        int y = get<1>(coord);
        tuple<int, int> coordsOfLeftImg = corners_1.at(x);
        tuple<int, int> coordsOfRightImg = corners_2.at(y);
        Point pointOfLeftImg = Point(
                get<0>(coordsOfLeftImg), get<1>(coordsOfLeftImg));
        //this may be the error
        Point pointOfRightImg = Point(
                get<0>(coordsOfRightImg),
                get<1>(coordsOfRightImg));
        leftPoints.push_back(pointOfLeftImg);
        rightPoints.push_back(pointOfRightImg);
    }

    return pair<vector<Point2f>, vector < Point2f >> (
            leftPoints, rightPoints);
}

/**
 * Call back for choosing topX matches
 * @param 
 * @param 
 */
void plotBFMatches(int, void*) {
    //the first values in the map are the lowest, remove until topx is satisfied
    BFScoreMap copyMap = filterScores(sortedMap, topX);

    //concatenate two images together
    Size img1Sz = src_1.size();
    Size img2Sz = src_2.size();
    int height = max(img1Sz.height, img2Sz.height);
    Mat plottedImg(height, img1Sz.width + img2Sz.width, img1.type());
    Mat left(plottedImg, Rect(0, 0, img1Sz.width, img1Sz.height));
    img1.copyTo(left);
    Mat right(plottedImg, Rect(img1Sz.width, 0, img2Sz.width, img2Sz.height));
    img2.copyTo(right);

    for (map<double, tuple<int, int>>::iterator it = copyMap.begin();
            it != copyMap.end(); ++it) {
        tuple<int, int> coord = it->second;
        int x = get<0>(coord);
        int y = get<1>(coord);

        //        cout << "Included score: " << it->first << " " << x << "," << y << endl;

        //max index contains the max score for coordinate 1 - 2
        //first point is the image on the right (coordinate 1)
        tuple<int, int> coordsOfLeftImg = corners_1.at(x);
        tuple<int, int> coordsOfRightImg = corners_2.at(y);
        Point pointOfLeftImg = Point(
                get<0>(coordsOfLeftImg), get<1>(coordsOfLeftImg));
        Point pointOfRightImg = Point(
                get<0>(coordsOfRightImg) + img1Sz.width,
                get<1>(coordsOfRightImg));
        line(plottedImg, pointOfLeftImg, pointOfRightImg, Scalar(0, 0, 255), 1);
        circle(plottedImg, pointOfLeftImg, 5, Scalar(0), 2, 8, 0);
        circle(plottedImg, pointOfRightImg, 5, Scalar(0), 2, 8, 0);
    }

    namedWindow(matches_window, CV_WINDOW_AUTOSIZE);
    imshow(matches_window, plottedImg);
}

/**
 * My algorithm makes use of my own matcher to search for the best matches
 * @return PairKeyPoints - vectors of the keypoints, indexed equally
 */
PairKeyPoints myAlgorithm() {
    cout << "Executing my algorithm" << endl;

    Patches patches_1 = extractPatches(corners_1, src_1, 15);
    Patches patches_2 = extractPatches(corners_2, src_2, 15);

    Mat match_scores = euclideanMatches(patches_1, patches_2);

    //sort the scores in a score map [x - first coord, y- second coord]
    scoreMap = createScoreMap(match_scores);

    //plot best matches
    namedWindow(matches_window, CV_WINDOW_NORMAL);
    createTrackbar("Best %: ", matches_window, &topX, 100, plotMatches);
    waitKey(0);

    //get key points
    return filterPoints(scoreMap, corners_1, corners_2);
}

/**
 * Callback for the RANSAC warped result and composite result
 * This function also reports statistics in the sysout. 
 * @param 
 * @param 
 */
void demoRansac(int, void*) {
    Mat mask;
    Mat H = findHomography(leftImagePoints, rightImagePoints, CV_RANSAC,
            ransac_thresh, mask);
    //i'm sure this mask is a keypoints x 1 matrix
    int numberOfInliers = 0;
    double residualAggregator = 0;
    Point2f srcTri[3], dstTri[3];
    vector<Point> leftInliers, rightInliers;
    for (int i = 0; i < mask.rows; i++) {
        bool isInlier = (unsigned int) mask.at<uchar>(i) ? true : false;
        if (isInlier) {
            numberOfInliers++;
            Point left = leftImagePoints[i];
            Point right = rightImagePoints[i];
            right.x = right.x;
            double dist = norm(Mat(left), Mat(right), NORM_L2);
            residualAggregator += dist;
            leftInliers.push_back(left);
            rightInliers.push_back(right);
        }
    }
    cout << "Number of Inliers: " << numberOfInliers << endl;
    cout << "Average residual (NORM_L2): " <<
            residualAggregator / numberOfInliers << endl;

    //concatenate two images together
    Size img1Sz = src_1.size();
    Size img2Sz = src_2.size();
    int height = max(img1Sz.height, img2Sz.height);
    Mat plottedImg(height, img1Sz.width + img2Sz.width, img1.type());
    //    Mat plottedImg(height * 2, img1Sz.width + img2Sz.width, img1.type());
    Mat left(plottedImg, Rect(0, 0, img1Sz.width, img1Sz.height));
    img1.copyTo(left);
    Mat right(plottedImg, Rect(img1Sz.width, 0, img2Sz.width, img2Sz.height));
    img2.copyTo(right);

    //display inlier matches in both images
    for (int i = 0; i < leftInliers.size(); i++) {
        Point pointOfLeftImg = leftInliers[i];
        Point pointOfRightImg = Point(rightInliers[i].x + img1Sz.width,
                rightInliers[i].y);
        line(plottedImg, pointOfLeftImg, pointOfRightImg, Scalar(0, 0, 255), 1);
        circle(plottedImg, pointOfLeftImg, 5, Scalar(0), 2, 8, 0);
        circle(plottedImg, pointOfRightImg, 5, Scalar(0), 2, 8, 0);
    }
    namedWindow("Inlier matches");
    imshow("Inlier matches", plottedImg);

    Mat warpPerspectiveResult;
    warpPerspective(img1, warpPerspectiveResult, H,
            Size(src_1.cols + src_2.cols, src_2.rows));

    //***** I tried getting the affine transform, but the results does not make
    //sense.//
    //get the indices of the best three matches from the left
    //    srcTri[0] = rightInliers[rightInliers.size()-1];
    //    srcTri[1] = rightInliers[rightInliers.size()-2];
    //    srcTri[2] = rightInliers[rightInliers.size()-3];
    //
    //    //get the indices of the best three matches from the right
    //    dstTri[0] = leftInliers[leftInliers.size()-1];
    //    dstTri[1] = leftInliers[leftInliers.size()-2];
    //    dstTri[2] = leftInliers[leftInliers.size()-3];
    //
    //    Mat trans_mat = getAffineTransform(srcTri, dstTri);
    //    Mat affineResult;
    //    warpAffine(result, affineResult, trans_mat, result.size());

    //****Just move the image to the right instead, using affine translation
    //    Mat moveRightMat = (Mat_<double>(2, 3) << 1, 0, img1.cols, 0, 1, 0);
    //    Mat movedToRightImg;
    //    warpAffine(result, movedToRightImg, moveRightMat, result.size());

    //warped image result
    Mat warpedIm = warpPerspectiveResult.clone();
    Mat half(warpedIm, Rect(0, 0, src_1.size().width,
            src_1.size().height));
    img2.copyTo(half);
    imshow(warped_window, warpedIm);

    //composite result 
    Mat leftClone = Mat(Size(src_1.cols + src_2.cols, src_2.rows), img1.type());
    Mat leftHalf(leftClone, Rect(0, 0, src_1.size().width,
            src_1.size().height));
    img2.copyTo(leftHalf);
    //*****I tried to use this function to calculate the average, but it was slow!
    //    compositeResult(leftClone, warpPerspectiveResult);
//    Mat intersection = leftClone & warpPerspectiveResult;
    Mat composite;
    addWeighted(leftClone, 0.5, warpPerspectiveResult, 0.5, 0, composite);

    namedWindow(composite_window);
    imshow(composite_window, composite);
}

/**
 * Function to determine if a pixel is black
 * @param point
 * @return 
 */
bool isZero(Vec3b point) {
    return point[0] == 0 && point[1] == 1 && point[2] == 0;
}

/**
 * Function to average out two images of equal size
 * @param left
 * @param right
 * @return 
 */
Mat compositeResult(Mat left, Mat right) {
    Mat composite = Mat(left.size(), left.type());
    for (int i = 0; i < left.cols; i++) {
        for (int j = 0; j < left.rows; j++) {
            Vec3b pointOfLeft = left.at<Vec3b>(j, i);
            Vec3b pointOfRight = right.at<Vec3b>(j, i);
            Vec3b pointOfComposite = composite.at<Vec3b>(j, i);
            //if left and right has a value, average
            if (!isZero(pointOfLeft) && !isZero(pointOfRight)) {
                pointOfComposite[0] = (pointOfLeft[0] + pointOfRight[0]) / 2;
                pointOfComposite[1] = (pointOfLeft[1] + pointOfRight[1]) / 2;
                pointOfComposite[2] = (pointOfLeft[2] + pointOfRight[2]) / 2;
            } else if (!isZero(pointOfLeft)) {
                //otherwise, if left has a value
                pointOfComposite[0] = pointOfLeft[0];
                pointOfComposite[1] = pointOfLeft[1];
                pointOfComposite[2] = pointOfLeft[2];
            } else {
                //lastly, right has a value
                pointOfComposite[0] = pointOfRight[0];
                pointOfComposite[1] = pointOfRight[1];
                pointOfComposite[2] = pointOfRight[2];
            }
        }
    }
    return composite;
}

/**
 * Callback for the detected matches. This uses the top x % parameter.
 * @param 
 * @param 
 */
void plotMatches(int, void*) {
    //concatenate two images together
    Size img1Sz = src_1.size();
    Size img2Sz = src_2.size();
    int height = max(img1Sz.height, img2Sz.height);
    Mat plottedImg(height, img1Sz.width + img2Sz.width, img1.type());
    //    Mat plottedImg(height * 2, img1Sz.width + img2Sz.width, img1.type());
    Mat left(plottedImg, Rect(0, 0, img1Sz.width, img1Sz.height));
    img1.copyTo(left);
    Mat right(plottedImg, Rect(img1Sz.width, 0, img2Sz.width, img2Sz.height));
    img2.copyTo(right);

    //we will maintain only the top x% of the scores
    ScoreMap filteredScoreMap = filterScores(scoreMap, topX / 100.0);

    for (map<double, tuple<int, int>>::iterator it = filteredScoreMap.begin();
            it != filteredScoreMap.end(); ++it) {
        tuple<int, int> coord = it->second;
        int x = get<0>(coord);
        int y = get<1>(coord);

        cout << "Included score: " << it->first << endl;

        //max index contains the max score for coordinate 1 - 2
        //first point is the image on the right (coordinate 1)
        tuple<int, int> coordsOfLeftImg = corners_1.at(x);
        tuple<int, int> coordsOfRightImg = corners_2.at(y);
        Point pointOfLeftImg = Point(
                get<0>(coordsOfLeftImg), get<1>(coordsOfLeftImg));
        Point pointOfRightImg = Point(
                get<0>(coordsOfRightImg) + img1Sz.width,
                get<1>(coordsOfRightImg));
        line(plottedImg, pointOfLeftImg, pointOfRightImg, Scalar(0, 0, 255), 1);
        circle(plottedImg, pointOfLeftImg, 5, Scalar(0), 2, 8, 0);
        circle(plottedImg, pointOfRightImg, 5, Scalar(0), 2, 8, 0);
    }

    cout << "Number of matches " << filteredScoreMap.size() << endl;
    namedWindow(matches_window, CV_WINDOW_AUTOSIZE);
    imshow(matches_window, plottedImg);
}

/**
 * Demo function to show the detected corners
 * @param src
 * @param coordinates
 * @param window_name
 */
void showCorners(Mat src, Coordinates coordinates,
        char* window_name) {
    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    src.copyTo(dst);

    for (int i = 0; i < coordinates.size(); i++) {
        tuple<int, int> coords = coordinates.at(i);
        Point point = Point(
                get<0>(coords), get<1>(coords));
        circle(dst, point, 5, Scalar(0), 2, 8, 0);
    }
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    imshow(window_name, dst);
    waitKey(0);
}

/**
 * Callback to show the detected corners. The user can adjust the threshold
 * accordingly.
 * @param 
 * @param 
 */
void demoKeypoints(int, void*) {
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros(src_1.size(), CV_32FC1);

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cornerHarris(src_1, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    /// Normalizing (is this needed?)
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    //vector for patches
    Patches patches;

    Mat demo = src_1.clone();

    /// Extracting patches around the corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int) dst_norm.at<float>(j, i) > thresh) {
                circle(demo, Point(i, j), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    /// Showing the result
    namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    imshow(source_window, demo);
}

/**
 * @author Dylan Valerio
 * @date May 1, 2015
 * @description This file contains solutions to  PA3, Epipolar Geometry
 *              for Computer Vision Class (CS282).
 * 
 * @summary TODO
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
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream> 
#include <sstream> 
#include <string>
#include "utils.cpp"
#include <limits>

using namespace cv;
using namespace std;

typedef vector<Vec3f> Epilines;
typedef pair<Epilines, Epilines> PairEpilines;

double pointLineDistance(Point2f line_start, Point2f line_end, Point2f point);
vector<Vec3f> getEpilines(PairKeyPoints pairKeyPoints);
PairEpilines plotEpilines(PairKeyPoints pairKeyPoints, Mat img1, Mat img2);
void plotMatches(KeyPoints leftMatches, KeyPoints rightMatches, Mat img1, Mat img2);

/** @function main */
int main(int argc, char** argv) {
    /// Load source image and convert it to gray
    Mat src1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat src2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img1, img2;
    equalizeHist(src1, img1);
    equalizeHist(src2, img2);

    //load matches file
    Mat matches;
    ifstream file(argv[3]);
    string line;
    int rows = 0;
    if (file.is_open()) {
        double x;
        while (getline(file, line)) {
            std::istringstream in(line);
            while (in >> x) {
                matches.push_back(x);
            }
            rows++;
        }
        file.close();
    }
    matches = matches.reshape(1, rows);
    cout << "Rows: " << matches.rows << endl;
    cout << "Cols: " << matches.cols << endl;

    //get the first two columns as left keypoints
    Mat leftKeypoints = Mat(matches, Rect(0, 0, 2, matches.rows));
    //get the first two columns as right keypoints        
    Mat rightKeypoints = Mat(matches, Rect(2, 0, 2, matches.rows));

    cout << "Size: " << leftKeypoints.size() << endl;
    cout << "Size: " << rightKeypoints.size() << endl;

    //convert matches for the fundamental matrix
    KeyPoints selPointsLeft, selPointsRight;
    for (int j = 0; j < leftKeypoints.rows; j++) {
        Point2f leftPoint = Point2f(leftKeypoints.at<double>(j, 0),
                leftKeypoints.at<double>(j, 1));
        selPointsLeft.push_back(leftPoint);
        Point2f rightPoint = Point2f(rightKeypoints.at<double>(j, 0),
                rightKeypoints.at<double>(j, 1));
        selPointsRight.push_back(rightPoint);
    }

    //plotting the matches for checking
    plotMatches(selPointsLeft, selPointsRight, img1, img2);

    PairEpilines pairEpilines = plotEpilines(pair<KeyPoints, KeyPoints>(
            selPointsLeft, selPointsRight), img1.clone(), img2.clone());
    Epilines rightEpilines = pairEpilines.first;
    Epilines leftEpilines = pairEpilines.second;

    //show residuals between the points of the second image to the epi lines
    double residualAggregate = 0;
    for (int i = 0; i < selPointsRight.size(); i++) {
        Point2f start = Point2f(0, -(rightEpilines[i])[2] / (rightEpilines[i])[1]);
        Point2f end = Point2f(img2.cols, -((rightEpilines[i])[2]+
                (rightEpilines[i])[0] * img2.cols) / (rightEpilines[i])[1]);
        residualAggregate += pointLineDistance(start, end, selPointsRight[i]);
    }
    cout << "Residuals: " << residualAggregate << endl;

    //Now fit a fundamental matrix based on putative 
    //correspondences obtained by your code from Program-
    //ming Assignment 2. Because the set of putative matches 
    //includes outliers, you will need to use RANSAC.
    //For this part, use only the normalized fitting approach.
    Coordinates coords1 = extractFeaturePoints(img1, 110);
    Coordinates coords2 = extractFeaturePoints(img1, 110);

    //get feature descriptors
    Patches patches1 = extractPatches(coords1, img1);
    Patches patches2 = extractPatches(coords2, img2);

    //get feature matches
    PairKeyPoints keypoints = getKeyPoints(coords1, coords2, img1,
            img2, 50, 20);

    //get another fundamental matrix
    Mat pa2OutputMask;
    Mat pa2LeftMat = Mat(keypoints.first);
    Mat pa2RightMat = Mat(keypoints.second);
    cout << pa2LeftMat << endl;
    Mat pa2FundamentalMatrix = findFundamentalMat(
            pa2LeftMat, pa2RightMat, pa2OutputMask, FM_RANSAC);
    cout << pa2FundamentalMatrix << endl;

    PairEpilines pa2Epilines = plotEpilines(pair<KeyPoints, KeyPoints>(
            keypoints.first, keypoints.second), img1.clone(), img2.clone());

    return 0;
}

Epilines getEpilines(KeyPoints firstKeyPoints, KeyPoints secondKeyPoints) {
    //match points
    Mat outputMask;
    Mat leftPointsMat = Mat(firstKeyPoints);
    Mat rightPointsMat = Mat(secondKeyPoints);
    //very high integer value to include all points
    Mat fundamentalMatrix = findFundamentalMat(
            leftPointsMat, rightPointsMat, outputMask, FM_RANSAC,
            3);
    cout << fundamentalMatrix << endl;

    //compute epipolar lines
    vector<Vec3f> epilines;
    computeCorrespondEpilines(leftPointsMat, 1, fundamentalMatrix, epilines);
    return epilines;
}

PairEpilines plotEpilines(PairKeyPoints pairKeyPoints, Mat img1, Mat img2) {
    vector<Vec3f> rightEpilines = getEpilines(pairKeyPoints.first,
            pairKeyPoints.second);
    vector<Vec3f> leftEpilines = getEpilines(pairKeyPoints.second,
            pairKeyPoints.first);

    // for all epipolar lines
    for (vector<Vec3f>::const_iterator it = rightEpilines.begin();
            it != rightEpilines.end(); ++it) {
        // draw the epipolar line between first and last column
        line(img2, Point(0, -(*it)[2] / (*it)[1]),
                Point(img2.cols, -((*it)[2]+(*it)[0] * img2.cols) / (*it)[1]),
                Scalar(255, 255, 255));
    }
    for (vector<Vec3f>::const_iterator it = leftEpilines.begin();
            it != leftEpilines.end(); ++it) {
        // draw the epipolar line between first and last column
        line(img1, Point(0, -(*it)[2] / (*it)[1]),
                Point(img1.cols, -((*it)[2]+(*it)[0] * img1.cols) / (*it)[1]),
                Scalar(255, 255, 255));
    }

    // Display the images with points and epipolar lines
    Size img1Sz = img1.size();
    Size img2Sz = img2.size();
    int height = max(img1Sz.height, img2Sz.height);
    Mat plottedImg(height, img1Sz.width + img2Sz.width, img1.type());
    Mat left(plottedImg, Rect(0, 0, img1Sz.width, img1Sz.height));
    img1.copyTo(left);
    Mat right(plottedImg, Rect(img1Sz.width, 0, img2Sz.width, img2Sz.height));
    img2.copyTo(right);
    namedWindow("Epilines");
    imshow("Epilines", plottedImg);
    waitKey(0);

    return pair<vector<Vec3f>, vector < Vec3f >> (leftEpilines, rightEpilines);
}

double pointLineDistance(Point2f line_start, Point2f line_end, Point2f point) {
    double normalLength = norm(line_end - line_start);
    double distance = (double) ((point.x - line_start.x) *
            (line_end.y - line_start.y) - (point.y - line_start.y) *
            (line_end.x - line_start.x)) / normalLength;
    return distance;
}

void plotMatches(KeyPoints leftMatches, KeyPoints rightMatches, Mat img1, Mat img2) {
    //create combination matrix
    Size img1Sz = img1.size();
    Size img2Sz = img2.size();
    int height = max(img1Sz.height, img2Sz.height);
    Mat plottedImg(height, img1Sz.width + img2Sz.width, img1.type());
    //    Mat plottedImg(height * 2, img1Sz.width + img2Sz.width, img1.type());
    Mat left(plottedImg, Rect(0, 0, img1Sz.width, img1Sz.height));
    img1.copyTo(left);
    Mat right(plottedImg, Rect(img1Sz.width, 0, img2Sz.width, img2Sz.height));
    img2.copyTo(right);

    //draw the matches
    for (int i = 0; i < leftMatches.size(); i++) {
        Point pointOfLeftImg = leftMatches[i];
        //offset for the first image
        Point pointOfRightImg = Point(rightMatches[i].x + img1.cols, 
                rightMatches[i].y);
        line(plottedImg, pointOfLeftImg, pointOfRightImg, Scalar(0, 0, 255), 1);
        circle(plottedImg, pointOfLeftImg, 5, Scalar(0), 2, 8, 0);
        circle(plottedImg, pointOfRightImg, 5, Scalar(0), 2, 8, 0);
    }
    namedWindow("Matches", CV_WINDOW_AUTOSIZE);
    imshow("Matches", plottedImg);
    waitKey(0);
}


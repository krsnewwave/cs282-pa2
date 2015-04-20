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

//type definitions
typedef pair<vector<Point2f>, vector<Point2f>> KeyPoints;
typedef map<double, tuple<int, int>, greater<double>> ScoreMap;
typedef vector<tuple<int, int>> Coordinates;
typedef vector<Mat> Patches;
typedef multimap<double, int, greater<double>> BFScoreMap;

/// Global variables
Mat src_1, src_2, img1, img2, filtered_coords;

int thresh = 150;
int max_thresh = 255;
int ransac_thresh = 1;
int topX = 20;

char* source_window = "Source image";
char* corners_window = "Corners detected";
char* matches_window = "Matches found";

vector<Point2f> leftImagePoints, rightImagePoints;
Coordinates corners_1, corners_2;
ScoreMap scoreMap;
vector<DMatch> bfMatches;
BFScoreMap sortedMap;

/// Function header
void demoKeypoints(int, void*);
void plotMatches(int, void*);
Patches extractPatches(Coordinates coordinates, Mat dst,
        int width = 5);
Mat euclideanMatches(Patches patches_1, Patches patches_2);
Coordinates extractFeaturePoints(Mat img, int threshold);

void showCorners(Mat src, Coordinates coordinates, char* window_name);
ScoreMap filterScores(ScoreMap scoreMap, double topX);
void demoRansac(int, void* userdata);
ScoreMap createScoreMap(Mat match_scores);
KeyPoints filterPoints(ScoreMap scoreMap, Coordinates coords1, Coordinates coords2);

KeyPoints myAlgorithm();

//revision: using BFMatcher

vector<DMatch> extractMatches(Patches leftDesc, Patches rightDesc);

/** @function main */
int main(int argc, char** argv) {
    /// Load source image and convert it to gray
    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    cvtColor(img1, src_1, CV_BGR2GRAY);
    cvtColor(img2, src_2, CV_BGR2GRAY);

    //equalizeHist(img1, src_1);
    //equalizeHist(img2, src_2);

    /// Create a window and a trackbar
    namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, demoKeypoints);
    imshow(source_window, src_1);

    demoKeypoints(0, 0);
    waitKey(0);
    destroyWindow(source_window);
    destroyWindow(corners_window);

    cout << "Extracting corners" << endl;

    corners_1 = extractFeaturePoints(src_1, thresh);
    corners_2 = extractFeaturePoints(src_2, thresh);

    showCorners(src_1, corners_1, "first image");
    showCorners(src_2, corners_2, "second image");

    KeyPoints keyPoints = myAlgorithm();

    leftImagePoints = keyPoints.first;
    rightImagePoints = keyPoints.second;

    //plot stitched image
    namedWindow("Ransac", CV_WINDOW_NORMAL);
    createTrackbar("Ransac threshold: ",
            "Ransac", &ransac_thresh, 300, demoRansac);
    waitKey(0);
    return 0;
}

void bestFirstMatchAlgorithm() {
    cout << "Executing opencv's BFMatcher algorithm" << endl;
    Patches patches_1 = extractPatches(corners_1, src_1, 20);
    Patches patches_2 = extractPatches(corners_2, src_2, 20);

    //flatten the patches vector to just one Mat
    Mat left, right;
    for (int i = 0; i < patches_1.size(); i++) {
        left.push_back(patches_1.at(i));
    }
    for (int i = 0; i < patches_2.size(); i++) {
        right.push_back(patches_2.at(i));
    }

    BFMatcher matcher = BFMatcher(NORM_L2);
    matcher.match(left, right, bfMatches);

    //good matches according to topX
    namedWindow(matches_window, CV_WINDOW_NORMAL);
    //    createTrackbar("Best %: ", matches_window, &topX, 100, plotBFMatches);
}

void plotBFMatches(int, void*) {
    vector<DMatch> topMatches;
    for (int i = 0; i < bfMatches.size(); i++) {
        sortedMap.insert(pair<double, int>(bfMatches[i].distance, i));
    }
    //the first values in the map are the lowest, remove until topx is satisfied
    int numberToRemove = round(bfMatches.size() * (1 - topX));
    int removedValues = 0;
    //the first values in the score map are the lowest
    for (map<double, tuple<int, int>>::iterator it = sortedMap.begin();
            it != sortedMap.end() && removedValues < numberToRemove;
            ++it, removedValues++) {
        cout << "Removed score: " << it->first << endl;
        sortedMap.erase(it);
    }
}

KeyPoints myAlgorithm() {
    cout << "Executing my algorithm" << endl;

    Patches patches_1 = extractPatches(corners_1, src_1, 20);
    Patches patches_2 = extractPatches(corners_2, src_2, 20);

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

void demoRansac(int, void*) {
    Mat H = findHomography(leftImagePoints, rightImagePoints, CV_RANSAC,
            ransac_thresh);
    Mat result;
    warpPerspective(img2, result, H, Size(src_1.cols + src_2.cols, src_2.rows));

    Mat half(result, Rect(0, 0, src_1.size().width,
            src_1.size().height));
    img1.copyTo(half);
    imshow("Ransac", result);
}

ScoreMap createScoreMap(Mat match_scores) {
    ScoreMap scoreMap;
    //now plot lines, assume that the match_scores' x & y correspond to
    //the x in coords1 and the y in coords2
    for (int x = 0; x < match_scores.size().width; x++) {
        //know match for the first image patch
        //max index is the max match for the second image
        int minIndex = 0;
        double minValue = HUGE_VAL;
        for (int y = 0; y < match_scores.size().height; y++) {
            double val = match_scores.at<double>(x, y);
            if (minValue > val) {
                minValue = val;
                minIndex = y;
            }
        }
        scoreMap.insert(pair<double, tuple<int, int>>(match_scores.
                at<double>(x, minIndex), make_tuple(x, minIndex)));
    }
    return scoreMap;
}

/**
 * 
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

KeyPoints filterPoints(ScoreMap scoreMap, Coordinates coords1,
        Coordinates coords2) {
    vector<Point2f> pointsOfLeftImg, pointsOfRightImg;

    for (map<double, tuple<int, int>>::iterator it = scoreMap.begin();
            it != scoreMap.end(); ++it) {
        tuple<int, int> coord = it->second;
        int x = get<0>(coord);
        int y = get<1>(coord);

        cout << "Included score: " << it->first << endl;

        //max index contains the max score for coordinate 1 - 2
        //first point is the image on the right (coordinate 1)
        tuple<int, int> coordsOfLeftImg = coords1.at(x);
        tuple<int, int> coordsOfRightImg = coords2.at(y);
        Point pointOfLeftImg = Point(
                get<0>(coordsOfLeftImg), get<1>(coordsOfLeftImg));
        //this may be the error
        Point pointOfRightImg = Point(
                get<0>(coordsOfRightImg) + src_1.size().width,
                get<1>(coordsOfRightImg));
        pointsOfLeftImg.push_back(pointOfLeftImg);
        pointsOfRightImg.push_back(pointOfRightImg);
    }
    return pair<vector<Point2f>, vector < Point2f >> (
            pointsOfLeftImg, pointsOfRightImg);
}

/**
 * Filters the scores
 * @param score
 * @param top - percent to be kept
 * @return 
 */
ScoreMap filterScores(ScoreMap scoreMap, double topX) {
    ScoreMap copyMap = scoreMap;
    int numberToRemove = round(scoreMap.size() * (1 - topX));
    int removedValues = 0;
    //the first values in the score map are the lowest
    for (map<double, tuple<int, int>>::iterator it = copyMap.begin();
            it != copyMap.end() && removedValues < numberToRemove;
            ++it, removedValues++) {
        cout << "Removed score: " << it->first << endl;
        copyMap.erase(it);
    }
    return copyMap;
}

/**
 * For each descriptor, compute euclidean distance to another.
 * @param patches_1
 * @param patches_2
 * @return vector with matches like so: dist(patches_1[0],patches_2[0])
 */
Mat euclideanMatches(Patches patches_1, Patches patches_2) {
    Mat scores = Mat::ones(Size(patches_1.size(), patches_2.size()), CV_64FC1);
    for (int i = 0; i < patches_1.size(); i++) {
        for (int j = 0; j < patches_2.size(); j++) {
            double dist = norm(patches_1.at(i), patches_2.at(j), NORM_L2);
            scores.at<double>(i, j) = dist;
        }
    }

    return scores;
}

/**
 * @param img
 * @param threshold - threshold for the feature keypoints' scores
 * @return vector of tuples with the coordinates of the feature keypoints
 */
Coordinates extractFeaturePoints(Mat img, int threshold) {
    Mat dst_norm;
    Mat dst = Mat::zeros(img.size(), CV_64FC1);
    Coordinates coordinates;

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cornerHarris(img, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    /// Normalizing (i've yet to study what this means)
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int) dst_norm.at<float>(j, i) > threshold) {
                //filteredDst.at<float>(j, i) = dst.at<float>(j, i);
                coordinates.push_back(make_tuple(i, j));
            }
        }
    }
    return coordinates;
}

/**
 * 
 * @param coordinates - container for the feature keypoints' coordinates
 * @param dst - image
 * @param width - width size of the feature vectors
 * @return vector of feature descriptors
 */
Patches extractPatches(Coordinates coordinates, Mat dst, int width) {
    //vector for patches
    Patches patches;

    /// Extracting patches around the corners
    for (int j = 0; j < coordinates.size(); j++) {
        int x, y;
        tie(x, y) = coordinates.at(j);
        Range rowRange = Range(x - width, x + width);
        Range colRange = Range(y - width, y + width);
        if (x - width < 0 || x + width > dst.cols ||
                y - width < 0 || y + width > dst.rows) {
            continue;
        }
        Mat roi = dst(colRange, rowRange);
        Mat patch = Mat::zeros(roi.size(), CV_64FC1);
        roi.copyTo(patch);
        Mat reshaped = patch.reshape(0, 1);
        //        imshow("test", reshaped);
        //        waitKey(0);
        patches.push_back(reshaped);
    }

    return patches;
}

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
 * 
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

    /// Extracting patches around the corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int) dst_norm.at<float>(j, i) > thresh) {
                circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    /// Showing the result
    namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
    imshow(corners_window, dst_norm_scaled);
}

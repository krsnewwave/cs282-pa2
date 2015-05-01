/**
 * @author Dylan Valerio
 * @date May 1, 2015
 * @description This file contains solutions to PA2 and PA3, 
 *              for Computer Vision Class (CS282).
 * 
 * @summary This is a library file containing the necessary code for both PA2
 *          and PA3. It comprises extracting patches, generating the score map
 *          necessary forgetting the top x% matches, computing the Euclidean
 *          distance scores for each patch, extracting patches, and getting
 *          the corners using the Harris corner detection algorithm.
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
#include "utils.hpp"

using namespace cv;
using namespace std;

/**
 * Extracts the patches for a given coordinate and width
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
        patches.push_back(reshaped);
    }

    return patches;
}

/**
 * Gets the top x% of scores in a BF score map
 * @param scoreMap - used for the bestFirstMatchAlgorithm
 * @param topX
 * @return 
 */
BFScoreMap filterScores(BFScoreMap scoreMap, double topX) {
    topX = topX / 100;
    BFScoreMap copyMap = scoreMap;
    int numberToRemove = round(scoreMap .size() * (1 - topX));
    int removedValues = 0;
    //the first values in the score map are the lowest
    for (map<double, tuple<int, int>>::iterator it = copyMap .begin();
            it != copyMap.end() && removedValues < numberToRemove;
            ++it, removedValues++) {
        //        cout << "Removed score: " << it->first << endl;
        copyMap.erase(it);
    }
    return copyMap;
}

/**
 * Creates a score map from a matrix of scores. This is necessary for
 * getting a ranking of the top scores.
 * @param match_scores
 * @return ScoreMap - the data structure used for myAlgorithm in main.cpp
 */
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
 * Using a score map, get the x and y values from the coordinates to create
 * a PairKeyPoints object, which is a more intuitive way to manipulate key points
 * @param scoreMap - used by myAlgorithm from main.cpp
 * @param coords1
 * @param coords2
 * @return 
 */
PairKeyPoints filterPoints(ScoreMap scoreMap, Coordinates coords1,
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
        Point pointOfRightImg = Point(
                get<0>(coordsOfRightImg),
                get<1>(coordsOfRightImg));
        pointsOfLeftImg.push_back(pointOfLeftImg);
        pointsOfRightImg.push_back(pointOfRightImg);
    }
    return pair<vector<Point2f>, vector < Point2f >> (
            pointsOfLeftImg, pointsOfRightImg);
}

/**
 * Filters the scores according to the top x%
 * @param score - used by myAlgorithm from main.cpp
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
 * For each descriptor, compute Euclidean distance to another.
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
 * Get the corners using Harris corner detection algorithm
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
 * Create a score map
 * @param bfMatches - used by the bestFirstMatchAlgorithm in main.cpp
 * @return 
 */
BFScoreMap createScoreMap(vector<DMatch> bfMatches) {
    vector<DMatch> topMatches;
    BFScoreMap sortedMap;
    for (int i = 0; i < bfMatches.size(); i++) {
        sortedMap.insert(pair<double, tuple<int, int>>(bfMatches[i].distance,
                make_tuple(bfMatches[i].queryIdx, bfMatches[i].trainIdx)));
    }
    return sortedMap;
}

/**
 * Extract key points using harris corner detection, and getting the top x%
 * Euclidean distance. This is used by PA3 to get the set of candidate matches.
 * @param leftCorners
 * @param rightCorners
 * @param src1
 * @param src2
 * @param threshold
 * @param width
 * @return 
 */
PairKeyPoints getKeyPoints(Coordinates leftCorners, Coordinates rightCorners,
        Mat src1, Mat src2, double threshold, int width) {
    Patches patches_1 = extractPatches(leftCorners, src1, width);
    Patches patches_2 = extractPatches(rightCorners, src2, width);

    //flatten the patches vector to just one Mat
    Mat left, right;
    for (int i = 0; i < patches_1.size(); i++) {
        left.push_back(patches_1.at(i));
    }
    for (int i = 0; i < patches_2.size(); i++) {
        right.push_back(patches_2.at(i));
    }

    vector<DMatch> bfMatches;
    BFMatcher matcher = BFMatcher(NORM_L2, true);
    matcher.match(left, right, bfMatches);

    BFScoreMap sortedMap = createScoreMap(bfMatches);
    BFScoreMap filteredMap = filterScores(sortedMap, threshold);

    vector<Point2f> leftPoints, rightPoints;
    //get the points from the score map
    for (map<double, tuple<int, int>>::iterator it = filteredMap.begin();
            it != filteredMap.end(); ++it) {
        tuple<int, int> coord = it->second;
        int x = get<0>(coord);
        int y = get<1>(coord);
        tuple<int, int> coordsOfLeftImg = leftCorners.at(x);
        tuple<int, int> coordsOfRightImg = rightCorners.at(y);
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
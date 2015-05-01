/* 
 * File:   utils.hpp
 * Author: dylan
 *
 * Created on April 26, 2015, 7:03 PM
 */

#ifndef UTILS_HPP
#define	UTILS_HPP



#endif	/* UTILS_HPP */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream> 
#include <sstream> 
#include <string>

using namespace cv;
using namespace std;

//type definitions
typedef vector<Point2f> KeyPoints;
typedef pair<KeyPoints, KeyPoints> PairKeyPoints;
typedef map<double, tuple<int, int>, greater<double>> ScoreMap;
typedef vector<tuple<int, int>> Coordinates;
typedef vector<Mat> Patches;
typedef multimap<double, tuple<int, int>, greater<double>> BFScoreMap;

Patches extractPatches(Coordinates coordinates, Mat dst, int width = 5);
ScoreMap filterScores(ScoreMap scoreMap, double topX);
ScoreMap createScoreMap(Mat match_scores);
PairKeyPoints filterPoints(ScoreMap scoreMap, Coordinates coords1, 
        Coordinates coords2);
Mat euclideanMatches(Patches patches_1, Patches patches_2);
Coordinates extractFeaturePoints(Mat img, int threshold);
PairKeyPoints getKeyPoints(Coordinates leftCorners,
        Coordinates rightCorners, Mat src1, Mat src2, double threshold = 0.5,
        int width = 20);
vector<DMatch> extractMatches(Patches leftDesc, Patches rightDesc);
BFScoreMap filterScores(BFScoreMap scoreMap, double topX);
BFScoreMap createScoreMap(vector<DMatch> bfMatches);

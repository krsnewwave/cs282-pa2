README

Please use like so: ./pa2 [image 1] [image 2] [optional: patch size]
 
To build this, please use CMAKE. Simply run "cmake ." then "make all". 
The solution is divided into two classes, main.cpp and
utils.cpp. There is also the main3.cpp, which correspond to
Programming Assignment 3. Lastly, a bonus.cpp file is included as
reference to another solution using SURF features.

On running, there is a sequence of windows asking for the user's inputs. 
These are:

1.) The first window that appears is a UI on the detected corners.
The default value is already the best found by the author. Press any button
to continue.

2.) The second window is to determine the top x% of matches used for the
stitching. The default value is already the best found by the author. Press
any button to continue.

3.) The third step is a combination of four windows. The first window is to 
determine the RANSAC threshold value for the inliers. The default value is 
already the best found by the author. The second window displays the inlier
matches for both images. The third window shows the composite images. The last
window show the affine transformation using three key points with the best
match scores. If the RANSAC threshold slider is modified, the three windows 
will also update. In the console, the average residuals and the number of 
inlier points are displayed.
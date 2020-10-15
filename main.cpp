#include <iostream>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <unistd.h> // for getting the current dir
#include <string>


using namespace std;
using namespace cv;
const int thresh_slider_max = 255;
int thresh_slider = 61;
int maxArea = 20000;
int minArea = 50;

struct object
{
    double cx;
    double cy;
}target;

void TrackerCallback(int, void*);
void DrawBBoxes(Mat& frame, Mat& Stats, Mat& Centroids);


int main(int, char**) {

    Mat LabelledImage;
    Mat Stats;
    Mat Centroids;

    cout << "Hello, world!\n";
    VideoCapture cap("123.mp4",0);
    namedWindow("Gray Scale", WINDOW_AUTOSIZE);

    createTrackbar("Thresh Adjust", "Gray Scale", 
                    &thresh_slider, thresh_slider_max, TrackerCallback);

    
    
    while(1)
    {

        Mat frame;
        Mat orig;
        // Capture frame-by-frame
        cap >> frame;
        orig = frame;
        frame.convertTo(frame,CV_8U);
        cv::resize(frame, frame, cv::Size(), 0.50, 0.50);
        cv::resize(orig, orig, cv::Size(), 0.50, 0.50);

        cvtColor(frame, frame, COLOR_BGR2GRAY);

        // If the frame is empty, break immediately
        if (frame.empty())
            break;



        //imgproc here:
        threshold(frame,frame, thresh_slider, 255, THRESH_BINARY);
        int output = connectedComponentsWithStats(frame, LabelledImage,Stats,
                                                 Centroids, 4, CV_16U);

        DrawBBoxes(orig, Stats, Centroids);                                       

        // Display the resulting frame
        imshow("Gray Scale", orig);

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
        break;
    }
 
  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();
  cout<<"end"<<endl;
	
}

void TrackerCallback(int, void*)
{
    cout << thresh_slider << endl;
}

void DrawBBoxes(Mat& frame, Mat& Stats, Mat& Centroids)
{
    for(int i=0; i<Stats.rows; i++)
    {
        int area = Stats.at<int>(i, CC_STAT_AREA);
        if(area > minArea && area < maxArea)
        {
            int x = Stats.at<int>(Point(0, i));
            int y = Stats.at<int>(Point(1, i));
            int w = Stats.at<int>(Point(2, i));
            int h = Stats.at<int>(Point(3, i));
            target.cx = Centroids.row(i).at<double>(0);
            target.cy = Centroids.row(i).at<double>(1);


            std::cout << "x=" << x << " y=" << y << " w=" << w << " h=" << h << std::endl;
            printf("Cx: %f    Cy: %f", target.cx, target.cy);
            Scalar color(255,255,0);
            Rect rect(x,y,w,h);
            cv::rectangle(frame, rect, color);
        }
    }
}
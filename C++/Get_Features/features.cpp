/**
 Commands To Run
-> g++ -std=c++11 features.cpp `pkg-config --libs --cflags opencv` -o features
-> ./features

**/

#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;


double brightness(Mat img)
{


     float ratio = 90000 /(img.cols * img.rows);
     Size size(300,300);
       Mat image;
       resize(img,image, size, ratio, ratio);
       Rect myROI(80,80,220,220);
       Mat image1=image(myROI);
       Mat gray ;
       cvtColor(image1,gray, COLOR_BGR2GRAY);
    //copying all element of matrix in temp

return mean(gray)[0];

}



int eyedetect(Mat img){
  CascadeClassifier eyesCascade("haarcascade_eye.xml");
  // Detect eyes
  std::vector<Rect> eyes;
  float ratio = 90000 /(img.cols * img.rows);
  Size size(300,300);
    Mat image;
    resize(img,image, size, ratio, ratio);
  eyesCascade.detectMultiScale( image, eyes, 1.2, 3, 0 |CASCADE_SCALE_IMAGE, Size(20, 20) );
  for( size_t j = 0; j < eyes.size(); j++ )
     {

     }


  return eyes.size();
}

double sharpness(Mat img,int w, int h)
{
  double variance;

  float ratio = 90000 /(img.cols * img.rows);
  Size size(300,300);
    Mat image;
    resize(img,image, size, ratio, ratio);
    Rect myROI(80,80,220,220);
    Mat image1=image(myROI);
    Mat gray ;
    cvtColor(image1,gray, COLOR_BGR2GRAY);
    Mat laplacianImage;
    Laplacian(gray, laplacianImage, CV_64F);
    Scalar mean, stddev;

    meanStdDev(laplacianImage, mean, stddev);
    variance = stddev.val[0] * stddev.val[0];

    return variance;

}


int main(int argc, char* argv[] ){
  string snapPath,good,number,snapID,distance,badmatches,total,confidence,faceID,top,bot,left,right,userEstimate,extra;
  string snapDirectory="/home/wmk/IQA/snaps/"; //needs to be changed
  Mat snapImg;
  int size,t,l,w,h,eyeNum;
  double sharp,bright;
  ifstream ip("snap_scores.csv");//name of the file containing snap data
  ofstream myfile;
  myfile.open ("out.csv");// output file for features
  if(!ip.is_open())std::cout<<"Could Not Open!"<< "\n";
// read through csv
  while(ip.good()){
    getline(ip,number,',');
    getline(ip,snapID,',');
    getline(ip,distance,',');
    getline(ip,badmatches,',');
    getline(ip,total,',');
    getline(ip,confidence,',');
    getline(ip,faceID,',');
    getline(ip,top,',');
    getline(ip,bot,',');
    getline(ip,left,',');
    getline(ip,right,',');
    getline(ip,userEstimate,'\n');

    if(snapID.find('"')<=snapID.size()){
      snapID.erase(snapID.find('"'),1);
    }



    if((userEstimate== "0" )||(userEstimate == "1")||(userEstimate== "2" )||(userEstimate <= "3")||(userEstimate== "4" )||(userEstimate <= "5")){
      cout << "SnapID: " <<snapID << endl;
      snapPath=snapDirectory+snapID+".jpeg";
      snapImg=imread(snapPath);
      Mat img = imread(snapPath,CV_LOAD_IMAGE_COLOR);

      //prevent face from exeeding image boundries
      t=max(stoi(top), 0);
      l=max(stoi(left), 0);
      w=min(stoi(right), img.size().width )-l;
      h=min(stoi(bot), img.rows)-t;
      size  = w*h;

     Rect crop=Rect(l,t,w,h);
     Mat croppedImage=img(crop);

     size  = w*h;
     sharp=sharpness(croppedImage,w,h);
     eyeNum=eyedetect(croppedImage);
     bright=brightness(croppedImage);

     cout << "Size: " <<size << endl;
     cout << "Sharpness: " <<sharp << endl;
     cout << "Number of eyes: " <<eyeNum << endl;
     cout << "Brightness: " <<bright << endl;

     if((userEstimate == "3")||(userEstimate== "4" )||(userEstimate == "5")){
       good="1";
     }else{
       good="0";
     }
//write features to csv file
     myfile << snapID<<","<< confidence<<","<<sharp<<","<<size<<","<<bright<<","<<eyeNum<<","<<good<<"\n";
  }
        // close the window


  }

  ip.close();
myfile.close();
return 0;
}

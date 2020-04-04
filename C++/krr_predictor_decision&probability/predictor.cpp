/**
 Commands To Run
-> g++ -std=c++11 -O3 -I.. ../dlib/all/source.cpp -lpthread -lX11 predictor.cpp `pkg-config --libs --cflags opencv` -o predictor

-> ./predictor

**/

#include <opencv2/opencv.hpp>
#include <fstream>
#include <dlib/svm.h>



using namespace std;
using namespace cv;
using namespace dlib;


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

return mean(gray)[0];

}



int eyedetect(Mat img){
  CascadeClassifier eyesCascade("haarcascade_eye.xml");
  std::vector<Rect> eyes;

  float ratio = 90000 /(img.cols * img.rows);
  Size size(300,300);
  Mat image;

  resize(img,image, size, ratio, ratio);
  eyesCascade.detectMultiScale( image, eyes, 1.2, 3, 0 |CASCADE_SCALE_IMAGE, Size(20, 20) );

  return eyes.size();
}

double sharpness(Mat img,int w, int h)
{
  double variance;

  float ratio = 90000 /(img.cols * img.rows);
  Size size(300,300);
    Mat image,gray;
    resize(img,image, size, ratio, ratio);

    Rect myROI(80,80,220,220);
    Mat image1=image(myROI);
    cvtColor(image1,gray, COLOR_BGR2GRAY);
    Mat laplacianImage;
    Laplacian(gray, laplacianImage, CV_64F);
    Scalar mean, stddev;

    meanStdDev(laplacianImage, mean, stddev);
    variance = stddev.val[0] * stddev.val[0];

    return variance;

}

double pfunction(string snapPath,string confidence,string top,string left,string right,string bot){
  int size,t,l,w,h,eyeNum;
  double sharp,bright,probability;
  typedef matrix<double, 5, 1> sample_type;
  typedef radial_basis_kernel<sample_type> kernel_type;
  typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
  typedef normalized_function<probabilistic_funct_type> pfunct_type;
  pfunct_type learned_pfunct;
  deserialize("trained_pmodal.dat") >> learned_pfunct;
  sample_type sample;



    Mat img = imread(snapPath,CV_LOAD_IMAGE_COLOR);

    t=max(stoi(top), 0);
    l=max(stoi(left), 0);
    w=min(stoi(right), img.size().width )-l;
    h=min(stoi(bot), img.rows)-t;

   Rect crop=Rect(l,t,w,h);
   Mat faceImage=img(crop);

   size  = w*h;
   sharp=sharpness(faceImage,w,h);
   eyeNum=eyedetect(faceImage);
   bright=brightness(faceImage);
   sample(0) = stod(confidence);
   sample(1) = sharp;
   sample(2) = size;
   sample(3) = bright;
   sample(4) = eyeNum;
probability=learned_pfunct(sample);
   cout << "Probability: " << probability << endl;

return probability;

}

double dfunction(string snapPath,string confidence,string top,string left,string right,string bot){
  int size,t,l,w,h,eyeNum;
  double sharp,bright,decision;
  typedef matrix<double, 5, 1> sample_type;
  typedef radial_basis_kernel<sample_type> kernel_type;
  typedef decision_function<kernel_type> dec_funct_type;
  typedef normalized_function<dec_funct_type> funct_type;
  funct_type learned_function;

  deserialize("trained_modal.dat") >> learned_function;
  sample_type sample;



    Mat img = imread(snapPath,CV_LOAD_IMAGE_COLOR);

    t=max(stoi(top), 0);
    l=max(stoi(left), 0);
    w=min(stoi(right), img.size().width )-l;
    h=min(stoi(bot), img.rows)-t;

   Rect crop=Rect(l,t,w,h);
   Mat faceImage=img(crop);

   size  = w*h;
   sharp=sharpness(faceImage,w,h);
   eyeNum=eyedetect(faceImage);
   bright=brightness(faceImage);

   sample(0) = stod(confidence);
   sample(1) = sharp;
   sample(2) = size;
   sample(3) = bright;
   sample(4) = eyeNum;

   decision=learned_function(sample);
   cout << "Decision: " << decision << endl;

return decision;

}

int main(int argc, char* argv[] ){
  string snapPath,good,number,snapID,distance,badmatches,total,confidence,faceID,top,bot,left,right,userEstimate,extra;
  string snapDirectory="../snaps/";
  ifstream ip("snap_scores.csv");
  double probability,decision;


  if(!ip.is_open())std::cout<<"Could Not Open!"<< "\n";

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

snapPath=snapDirectory+snapID+".jpeg";



// The decision function will return values >= 0 for samples it predicts
// are in the +1 class and numbers < 0 for samples it predicts to be in the -1 class.
decision=dfunction(snapPath, confidence, top, left, right, bot);

// The probability function will return a well conditioned probability
// instead of just a number > 0 for the +1 class and < 0 for the -1 class.
probability=pfunction(snapPath, confidence, top, left, right, bot);


}




ip.close();
return 0;
}

# Face_Quality_Predictor

The Face Quality Predictor can be used to determine if a face is either good or bad for facial recognition by  generating a decision or a well conditioned probability.

## Prerequisites

### Python

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following libraries.
pip for Python 2
pip3 for Python 3

```bash
pip3 install opencv-python
pip3 install dlib
pip3 install imutils
pip3 install numpy
pip3 install pandas
pip3 install csv
pip3 install collections
pip3 install os
```
### C++
Install [opencv](https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/) by following the guide in the link.

Install DLib
```bash
wget http://dlib.net/files/dlib-19.6.tar.bz2
tar xvf dlib-19.6.tar.bz2
cd dlib-19.6/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
pkg-config --libs --cflags dlib-1
cd ..
```
## Installing
In both the Python and C++ folders there exists a "snaps" folder, this folder needs to contain the images that are listed in the snap_scores.csv file. The snaps folder was not included in the repository due to the folder size.

The Linux Application files can execute the already compiled C++ code by simply calling ./krr ./predictor or ./features in the terminal.

The Python scripts can also be executed in the terminal after the prerequisite libraries have been installed. In feature.py the "snapFolder'' needs to contain the path to the snaps and "path" needs to be changed to where snap_scores.csv is stored.

## Deployment

![Program Flow](https://octodex.github.com/images/yaktocat.png)

### Features

The features.cpp script reads the labeled snap data from snap_scores.cpp that has the following format:

Number|SnapId|Distance|Bad Matches|Total Matches|Confidence|FaceId|Top|Bottom|Left|Right|UserEstimate
------|------|--------|-----------|-------------|----------|------|---|------|----|-----|------------
1|5c389ba0152b4f2eea00a51d| 'New'| 'New'| 'New'|1.0695158243179321|0c9b16cc-e3dc-4288-b235-ed76e9011adc|195|407|343|555|3

From this data I use the snapId to get the image and the coordinates (Top, Bottom, Left, Right) to crop out the face. From the cropped out face we can get the following features: sharpness (blurriness) ,size ,Brightness and the number of eyes.
```C++
Rect crop=Rect(l,t,w,h);
Mat croppedImage=img(crop);

size  = w*h;
sharp=sharpness(croppedImage,w,h);
eyeNum=eyedetect(croppedImage);
bright=brightness(croppedImage);
```

 To get the sharpness accurately I scale the image to 300x300 pixels then I focussed on a smaller rectangle to reduce the impact that the background will have on the sharpness score.
```C++
float ratio = 90000 /(img.cols * img.rows);
Size size(300,300);
Mat image;
resize(img,image, size, ratio, ratio);

Rect myROI(80,80,220,220);
Mat image1=image(myROI);
```
The features are then saved line by line into the out.csv file to be used in the trainer script. UserEstimate ranging from 1-5 was the outcome of the labeling I did on the images, 1 and 2 being bad images and 3-5 being usable to good images. The label "good" was used to indicate a good or bad image 1 being good and 0 being bad. The out.csv had the following structure.
SnapId|Confidence|Sharpness|Size|Brightness|Eye Number|Good
------|----------|---------|----|----------|----------|----
5c389ba0152b4f2eea00a51d|1.0695158243179321|4.09426|44944|98.9971|4|0

### krr classification trainer

In this script I read in the out.csv file line by line and add the features to the sample matrix and the corresponding label.
```C++
amp(0) = stod(confidence);
samp(1) = stod(sharp);
samp(2) = stod(size);
samp(3) = stod(bright);
samp(4) = stod(eyeNum);
samples.push_back(samp);
if (good=="1")
          labels.push_back(+1);
      else
          labels.push_back(-1);
```
Next I normalize the samples and then train the model with the best gamma value that ended up being 0.078125
```C++
normalizer.train(samples);

for (unsigned long i = 0; i < samples.size(); ++i)
    samples[i] = normalizer(samples[i]);

trainer.set_kernel(kernel_type(0.078125));

learned_function.normalizer = normalize;  
learned_function.function = trainer.train(samples, labels);
```
The trained model can now be saved and imported in other scripts.
```C++
serialize("trained_pmodal.dat") << learned_pfunct;
serialize("trained_modal.dat") << learned_function;
```
### Predictor

This is the script that can be modified and used in production. The script combines parts of the feature and classification scripts. In the main function there are two functions that can be called, the pfunction and the dfunction. They both do exactly the same but the dfunction returns a double value between -1 and 1 and the pfunction returns a double value between 0 and 1.

The dfunction loads the decision trained model and the output value is a decision based on whether the value is positive or negative.

Probability|Score
-----------|-----
Low probability good image|0.1259
High probability good image|0.9956
Low probability bad image|-0.2659
High probability bad image|-0.9815

The pfunction loads the probability trained model and returns a probability from 0 to 1.

Probability|Score
-----------|-----
Low probability bad image|0.08945
High probability good image|0.68154

Any one of these functions can be used and their cut off can be adjusted based on need.
The two functions need 6 parameters and the script can be adapted to read them from json or other data structures.

* They need the full path to the snap or image.
* They need the confidence score of the image.
* They need the top coordinate of the persons face (y)
* They need the left coordinate of the persons face (x)
* They need the right coordinate of the persons face (x1)
* They need the bot coordinate of the persons face (y1)



```C++
decision=dfunction(snapPath, confidence, top, left, right, bot);
probability=pfunction(snapPath, confidence, top, left, right, bot);
```
## Improvements

* More features can still be added to the trainer to improve the accuracy. More labeled data is needed to increase the performance of the classifier. Instead of having one model with 20 000 labeled data for two different camera types it would be better to have two models of 10 000 each for every camera type.

* I labeled faces with objects blocking facial landmarks as bad images but at the moment we have no feature that can accurately detect this. This can cause the model to be less accurate because it has an image that has good features but that has a bad label for no reason that can be determined.

* After many attempts to create a pose estimation feature no accurate feature could be achieved. I settled for an eye detector that works quite well but in some cases a good image resulted in no eyes being detected. An accurate pose estimator could significantly improve the prediction model and should be considered in future versions.

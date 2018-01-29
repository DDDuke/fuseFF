/*
 * Copyright (c) 2017 Dane Brown <d.brown@ru.ac.za>
 * All rights reserved. No warranty, explicit or implicit, provided.
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator> // for ostream_iterator

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"
#include <numeric>
#include <stdio.h>
#include "../cmakeStyle/lbp.hpp"
using namespace lbp;
using namespace cv;
using namespace std;
using cv::CLAHE;

Ptr<FaceRecognizer> model;

string fn_csv;
string fn_csv2;
int firstSet = 1;
int samples;
int noClasses;
float startThresh = 0;
int testClassSize = 0;
string matcher;
int enhanceType;	
string reductionType;	
float varRed;
int weigh;
int cnt = 0;
string outname = "unknown";

vector<Mat> trainImages;
vector<int> trainLabels;
vector<string> trainPaths;
vector<Mat> testImages;
vector<int> testLabels;
vector<string> testPaths;

//modalities 
vector < vector<Mat> > trainModality (4, vector<Mat>(0)); //4 modalities
vector < vector<Mat> > testModality (4, vector<Mat>(0)); //4 modalities

vector <Mat> trainFused;
vector <Mat> testFused;
vector <Mat> pcaResult;
Size imageSize = Size(100,100);
Size size(30, 30);
int finalSize = 30;
Size origSize;
int resizeType = 3;
float lbpRadiusSize = 4;
float lbpNeighbourSize = 31;

// float svmC = 100000000.0; //principal components 
// float svmC = 1e0; //logistic re
float svmC = 1e4; //svm
float svmRegressionBias = 7.5;

// float eigenThresh = 10000; // dir:fpir def value 30000 150x150 face
// float eigenThresh = 3500; // dir:fpir def value 10000 55x55 fing (proportional to image size)
float eigenThresh = 0; // mcyt4000 fvc4500

int eigenNum = 200; //principal components
// int lbphRadius = 2; // iris 
int lbphRadius = 4; //4 face iris palm(L1)
// int lbphRadius = 6; //6 fing 
// int lbphRadius = 12; //6 fing + face
// int lbphRadius = 2; //2  iitd palm + casia iris
// int lbphRadius = 1; //face+palm ave
// int lbphRadius = 6; //face+palm cat

// int lbphNeighbours = 4;//fing 
// int lbphNeighbours = 6;//6 face palm(L1)
// int lbphNeighbours = 3;//3 iris
int lbphNeighbours = 6;//fing + face
// int lbphNeighbours = 9;// face+palm ave 
// int lbphNeighbours = 7;// face+palm 

// float lbphThresh = 20;// dir:fpir def 900 20
// float lbphThresh = 8.5;// dir:fpir def 2 up to 20
// float lbphThresh = 0.05;// iris

//metrics
float lbphThresh = 0;// finger 2.5
float dir = 0;
float fpir = 0;
float frr = 0;
float mr = 0;
int zero = 0;
int zeroFive = 0;
int one = 0;
int five = 0;
int ten = 0;
int thirty = 0;
float bestFalseAcceptPercent = 0;
float bestFalseRejectPercent = 0;
float bestEER = 999;
float finalEER = 0;
int threshIterations = 100;

float modalityWeight1 = 0.5;//is 2nd for sily reason
float modalityWeight2 = 0.5;//is first

//svm
string svmTrainFile;
string svmTestFile;

template <typename T> string tostr(const T& t) { 
   ostringstream os; 
   os<<t; 
   return os.str(); 
} 

void saveTrainData (vector <Mat> fusedData)
{
   // 	cout<<"svmC "<<svmC<<endl;		
   
   
   // Open the file in write mode.
   ofstream trainOutputFile;
   svmTrainFile = outname + "svmtrain"; // add the output extension
   
   trainOutputFile.open(svmTrainFile.c_str(),ios::trunc);
   
   for(int i = 0;i < fusedData.size();i++)
   {
      int cnt = 1;//libsvm bug (0)
      
      threshold(fusedData[i],fusedData[i], 60, 255,THRESH_BINARY | CV_THRESH_OTSU);
      
      trainOutputFile << trainLabels[i]+1; //label image
      
      // Iterate through pixels.
      for (int r = 0; r < fusedData[i].rows; r++)
      {
         for (int c = 0; c < fusedData[i].cols; c++)
         {
            int pixel = fusedData[i].at<uchar>(r,c);
            if (pixel == 255)
               pixel = 1;
            trainOutputFile << " "<< cnt << ":"<<pixel;
            cnt++;
         }
      }
      
      trainOutputFile << endl;
      
      
   }
   
   //run svm-train
   string trainString;
   // 	./train -q -s 0 -n 4 -c 10e-3
   // 	 trainString = "./svm-train -q -c " + tostr(svmC) + " -t 0 " + svmTrainFile + " " + svmTrainFile +".model";
   //  trainString = "./train -q -c " + tostr(svmC) + " -s 4 " + svmTrainFile + " " + svmTrainFile +".model";
   
   if(matcher == "SVM")
      trainString = "./train -q -c " + tostr(svmC) + " -s 2 -n 4 "  + " " + svmTrainFile + " " + svmTrainFile +".model";//svm
//       trainString = "./train -q -c " + tostr(svmC) + " -s 2 -n 4 -e " + tostr(svmRegressionBias) + " " + svmTrainFile + " " + svmTrainFile +".model";//svm
      else			
         trainString = "./train -q -c " + tostr(svmC) + " -s 0 -n 4 -e " + tostr(svmRegressionBias) + " " + svmTrainFile + " " + svmTrainFile +".model";//lr
         
         system(trainString.c_str());
      
      
      // Close the file.
      trainOutputFile.close();
   
   // 	svmC /= 1e4;
}

void saveTestData (vector <Mat> fusedData2)
{
   // Open the file in write mode.
   ofstream testOutputFile;
   svmTestFile = outname + "svmtest"; // add the output extension   
   testOutputFile.open(svmTestFile.c_str(),ios::trunc);
   
   for(int i = 0;i < fusedData2.size();i++)
   {
      int cnt = 1; //libsvm bug (0)
      
      threshold(fusedData2[i],fusedData2[i], 60, 255,THRESH_BINARY | CV_THRESH_OTSU);
      
      testOutputFile << testLabels[i]+1; //label image
      
      
      // Iterate through pixels.
      for (int r = 0; r < fusedData2[i].rows; r++)
      {
         for (int c = 0; c < fusedData2[i].cols; c++)
         {
            int pixel = fusedData2[i].at<uchar>(r,c);
            if (pixel == 255)
               pixel = 1;
            testOutputFile << " "<< cnt << ":"<<pixel;
            cnt++;
         }
      }
      
      testOutputFile << endl;
      
      //probablity estimates
      
//       double fApB = decision_value*A+B;
//       if (fApB >= 0)
//       return Math.exp(-fApB)/(1.0+Math.exp(-fApB));
//       else
//       return 1.0/(1+Math.exp(fApB)) ;
      
      
   }
   
   //run svm-test
   string testString;
   // 	 testString = "./svm-predict " + svmTestFile + " " + svmTrainFile + ".model " + svmTestFile + "output";
   // 	if(matcher == "SVM")
//    testString = "./predict -b 1 " + svmTestFile + " " + svmTrainFile + ".model " + svmTestFile + "output";
   testString = "./predict " + svmTestFile + " " + svmTrainFile + ".model " + svmTestFile + "output";
   system(testString.c_str());
//    cout<<"testString"<<testString<<endl;
   // Close the file.
   testOutputFile.close();
}


static void optimalResize(Mat &reImage, Size finalSize)
{
   
   if (origSize.width >= finalSize.width && (matcher != "L"))
   {
      // 		                  cout<< "decreasing Size" << endl;
      resizeType = 3;
      resize(reImage,reImage, finalSize,resizeType); 
      
   }
   if (origSize.width < finalSize.width && (matcher != "L"))
   {
      // 		cout<< "increasing Size" << origSize.width<<endl;
      resizeType = 4;            
      resize(reImage,reImage, finalSize,1.5,1.5,resizeType);            
      
   }
   if (matcher == "L")
   {
      //                   cout<< "increasing Size" << endl;
      resizeType = 4;            
      resize(reImage,reImage, finalSize,1.5,1.5,resizeType);            //finger
//       resize(reImage,reImage, Size(100,100),1.5,1.5,resizeType);            //finger
//       resize(reImage,reImage, Size(200,200),1.5,1.5,resizeType);         //face 
      
   }
   
   
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, vector<string>& paths, char separator = ';') {
   
   float sharpHigh = 94.0;
   float sharpLow = 58.0;
   
   int gKernel = 17;
   int lKernel = 7;
   
   // 	if(finalSize<75)    
   {      
      //finger
      // 		lbpRadiusSize = 4/(55.0/(float)finalSize);
      // 		lbpNeighbourSize = 31/(55.0/(float)finalSize);
      //face
      
      lbpRadiusSize = 2/(150.0/(float)finalSize);
      lbpNeighbourSize = 23/(150.0/(float)finalSize);
      
      //fusion
      
//       lbpRadiusSize = 6;
//       lbpNeighbourSize = 4;
   }
   
   ifstream file(filename.c_str(), ifstream::in);
   if (!file) {
      string error_message = "No valid input file was given, please check the given filename.";
      CV_Error(CV_StsBadArg, error_message);
   }
   string line, path, classlabel;
   while (getline(file, line)) {
      stringstream liness(line);
      getline(liness, path, separator);
      getline(liness, classlabel);
      if(!path.empty() && !classlabel.empty()) {
         Mat input = imread(path, 0);
         
         Mat temp;
         
         //          if(cnt == 0)
         origSize = Size(input.cols,input.rows);
         
         //          if(origSize.width != finalSize) 
         optimalResize(input, Size(finalSize,finalSize));
         
         
         //          if(cnt == 0)
         //             cout<<lbpSize<<" lbpSize"<<endl;
         if (enhanceType == 1)
         {
            normalize(input, input, 0.0, 255.0, NORM_MINMAX, CV_8U);
            
//             if (matcher == "L")
//             {
//                					lbphRadius = 1; //6 fing 4 face start at 1 -- 6)
//                					lbphNeighbours = 7;//4 fing 6 face (start at 3 -- 9)
//             }
            // 		
            images.push_back(input);
            
         }
         else
            if (enhanceType == 2)
            {
               equalizeHist(input,input);
               // 					Ptr<CLAHE> clahe = createCLAHE();
               // 					clahe->setClipLimit(2);							
               // 					clahe->apply(input,input);
               
               images.push_back(input);
               // 					imshow("hist Image", input);
               
            }
            
            else
               if (enhanceType == 3 || enhanceType == 4)
               {
                  if (enhanceType == 3)
                  {
                     equalizeHist(input,input);
                     
                     // 							Ptr<CLAHE> clahe = createCLAHE();
                     // 							clahe->setClipLimit(2);							
                     // 							clahe->apply(input,input);
                  }
                  
                  if (matcher == "L")
                  {
                     gKernel = 7;
                     lKernel = 5;
                     sharpHigh = 0.5;
                     sharpLow = 2;
                  }
                  else
                  {
                     gKernel = 17;
                     lKernel = 7;
                  }
                  
                  GaussianBlur (input, input, Size(gKernel,gKernel), 0,0);			
                  Laplacian( input, temp, CV_8U, lKernel, 1, 0, BORDER_DEFAULT );
                  
                  // 							addWeighted(input, sharpHigh, temp, sharpLow*-1, 0, temp);
                  images.push_back(temp);
                  
                  
               }
               
               else
                  if (enhanceType == 5 || enhanceType == 6)							
                  {
                     if (enhanceType == 6)
                     {
                        if (matcher == "L")
                        {
                           gKernel = 7;
                           lKernel = 5;	
                           sharpHigh = 0.5;
                           sharpLow = 2;
                        }
                        else
                        {
                           gKernel = 17;
                           lKernel = 7;
                           equalizeHist(input, input);
                           
                        }
                        
                        GaussianBlur (input, input, Size(gKernel,gKernel), 0,0);			
                        Laplacian( input, temp, CV_8U, lKernel, 1, 0, BORDER_DEFAULT );
                        
                     }
                     else
                        input.copyTo(temp);
                     
                     Mat temp2;
                     //                      ELBP(temp, temp2, lbpRadiusSize, lbpNeighbourSize);   //19 better than 31 for face .. visa versa for fing
                     VARLBP(temp, temp2, lbpRadiusSize, lbpNeighbourSize);   //19 better than 31 for face .. visa versa for fing                     
                     // 											OLBP(temp,temp2);                     
                     
                     normalize(temp2, temp2, 0, 255, NORM_MINMAX, CV_8U);
                     
                     //                      resize(input,input, size,INTER_AREA);
                     
                     optimalResize(temp2, Size(finalSize,finalSize));
                     temp2 = (input - temp2)*1.1; //orig - bad features
                     
                     images.push_back(temp2);
                     
                  }
                  else
                     if (enhanceType == 7)							
                     {
                        
                        Mat temp2;
//                         								ELBP(input,temp, lbpRadiusSize, lbpNeighbourSize);   //19 better than 31 for face .. visa versa for fing
                        VARLBP(input,temp, lbpRadiusSize, lbpNeighbourSize);   //19 better than 31 for face .. visa versa for fing
                        //                         VARLBPDane(input,temp, lbpRadiusSize, lbpNeighbourSize);   //19 better than 31 for face .. visa versa for fing
                        //                                                 OLBP(input,temp);                        
                        
                        normalize(temp, temp, 0, 255, NORM_MINMAX, CV_8U);
                        
                        optimalResize(input, Size(finalSize,finalSize));
                        optimalResize(temp, Size(finalSize,finalSize));
                        //                         imshow("lbp Image orig", temp);
                        
                        // 								if(origSize.width > finalSize + 20) //for varlbp
                        // 								{
                        temp = (input - temp)*1.1; //orig - bad features
                        // 									
                        // 								}
                        //                         temp = temp2 - temp;
                        
                        //                                                 absdiff(input,temp, temp);                        
                        //                         absdiff(temp,temp2, temp);
                        
                        //                         imshow("orig Image", input);
                        //                         imshow("lbp Image", temp);
                        //                         waitKey();
                        //                         
                        
                        if (matcher == "L")
                        {
                           gKernel = 7;
                           lKernel = 5;	
                           sharpHigh = 0.5;
                        }
                        else
                        {
                           gKernel = 17;
                           lKernel = 7;
                           
                        }
                        
                        GaussianBlur (temp, temp, Size(gKernel,gKernel), 0,0);			
                        Laplacian( temp, input, CV_8U, lKernel, 1, 0, BORDER_DEFAULT );
                        
                        //                         resize(input,input, size, INTER_AREA);
                        
                        images.push_back(input);
                        
                        
                        
                        
                     }
                     
                     
                     labels.push_back(atoi(classlabel.c_str()));
                     paths.push_back(path);
      } 
      
   }
   
   //    cnt++;
}


//assign train and test samples from total images

void assignImages(vector<Mat> images,vector<int> labels,vector<string> paths, int samples)
{
   int cntTrain = 0;
   int prevLabel = 0;
   testClassSize = 0;
   trainLabels.clear();
   trainPaths.clear();
   testLabels.clear();
   testPaths.clear();
   for (int i = 0; i < images.size();i++)
   {
      if(labels[i] == prevLabel)
      {
         more:
         
         int realLabel = labels[i];
         if(labels[i] >= noClasses)
         {
            realLabel = -2;
         }
         
            
            
         if (cntTrain < samples && labels[i] < noClasses)
         {
            trainImages.push_back(images[i]);
            trainLabels.push_back(realLabel);
            trainPaths.push_back(paths[i]);
            cntTrain++;
//             cout<< "assigning train sample"<<paths[i]<<endl;
            
         }
         
         else
            if (cntTrain < samples && noClasses == 1 && labels[i] >= noClasses) //verification
            {
               
               trainImages.push_back(images[i]);
               trainLabels.push_back(realLabel);
               trainPaths.push_back(paths[i]);
               cntTrain++;
//                cout<< "assigning train sample"<<paths[i]<<endl;
            }
         
         else //assign rest to test sample
         {
      
            if(matcher == "SVM" || matcher == "LR")
            {
               if(labels[i] < noClasses)
               {
                  testImages.push_back(images[i]);
                  testLabels.push_back(realLabel);
                  testPaths.push_back(paths[i]);
               }
            }

            
            else
            {
               testImages.push_back(images[i]);
               testLabels.push_back(realLabel);
               testPaths.push_back(paths[i]);
               // 				 cout<< "assigning test sample"<<paths[i]<<endl;
               if(labels[i] < noClasses)
               {
                  testClassSize++; //true size
               }
            }
            
            
         }
      }
      
      else //zero train counter
      {
         cntTrain = 0;
         
         goto more;
         
      }
      prevLabel = labels[i];
   }
}

static  Mat formatImagesForPCA(const vector<Mat> &data)
{
   Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_64F); //assuming all images are the size of the first image
   for(unsigned int i = 0; i < data.size(); i++)
   {
      Mat image_row = data[i].clone().reshape(1,1);
      Mat row_i = dst.row(i);
      image_row.convertTo(row_i,CV_64F);
   }
   return dst;
}

static vector <Mat> pcaReduction(Mat data, vector <Mat> fusedPCA, float retainedVariance)
{
   
   //    cout << "\nRetained Variance = "<<retainedVariance*100<<"%"<<endl;
   // perform PCA
   PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, retainedVariance); //  a common value for retainedVariance
   
   vector <Mat> reconstruction;
   for(int i = 0;i < fusedPCA.size();i++)
   {
      Mat point = pca.project(data.row(i)); // project into the eigenspace, thus the image becomes a "point"
      
      Mat temp;
      
      temp = pca.backProject(point);// re-create the image from the "point"
      temp = temp.reshape(fusedPCA[i].channels(), fusedPCA[i].rows); // reshape from a row vector into image shape
      // 		temp.convertTo(temp,CV_8UC1);
      
      // 				normalize(temp, temp, 0, 255, NORM_MINMAX, CV_8UC1); // re-scale for displaying purposes
      // 		threshold(temp, temp, 60, 255,THRESH_BINARY | CV_THRESH_OTSU);
      convertScaleAbs( temp, temp );
      // 		equalizeHist(temp, temp);
      // 		GaussianBlur (temp, temp, Size(3, 3), 0,0);			
      
      
      reconstruction.push_back(temp); 
      
   }
   //    cout << "done!   # of principal components: " << pca.eigenvectors.rows << endl;
   
   
   return reconstruction;
}

Mat enhance(Mat image)
{
   int denoiseStr = 7;
   int templateSize = 7;
   int searchSize = 21;
   
   int sharpHigh = 3;
   int sharpLow = 2;
   
   int filterX = 17;
   int filterY = 11;	
   
   Mat denoise;
   image.copyTo(denoise);
   
   //denoise
   fastNlMeansDenoising(image,denoise, denoiseStr, templateSize, searchSize); //remove gaussian white noise
   
   //sharpen
   
   GaussianBlur(denoise, image, Size(filterX, filterY), 0,0);
   addWeighted(denoise, sharpHigh, image, sharpLow*-1, 0, image);
   
   return image;
}

Mat gaborFilter(Mat &image, int varGabor, int lam)
{	
   int kernel_size=33; //We observe that there is no effect of the size of the convolution kernel on the output image. This also implies that the convolution kernel is scale invariant, since scaling the kernel’s size is analogous to scaling the size of the image.
   
   // 	int pos_sigma= 3; //width of the Gaussian envelope used in the Gabor kernel.
   // 	int pos_lm = lam; //wavelength of the sinusoidal factor in the above equation.
   // 	int pos_th = varGabor;// kind of features the filter responds to. For example, giving theta a value of zero means that the filter is responsive only to horizontal features only.. 90 is vertical.
   // 	int pos_gamma= 3; //Gamma controls the ellipticity of the gaussian. When gamma = 1, the gaussian envelope is circular.
   // 	int pos_psi = 274; //the phase offset.
   
   int pos_sigma= 1;
   int pos_lm = 8;
   int pos_th = 0;
   float pos_gamma= 0.5;
   int pos_psi = 274;
   
   // 		int pos_sigma= 2;
   // 		int pos_lm = 6;
   // 		int pos_th = 0;
   // 		float pos_gamma= 3;		
   // 		int pos_psi = 0;
   
   Size KernalSize(kernel_size,kernel_size);
   double Sigma = pos_sigma;
   double Lambda = 0.5+pos_lm/100.0;
   double Theta = pos_th*CV_PI/180;
   double psi = pos_psi*CV_PI/180;;
   double Gamma = pos_gamma;
   
   Mat gaborKernel = getGaborKernel(KernalSize, pos_sigma, Theta, Lambda,Gamma,psi);
   filter2D(image, image, CV_32F, gaborKernel);
   // 	convertScaleAbs( image, image );
   
   image.convertTo(image, CV_32F, 1.0/255, 0);
   // 	normalize(image, image, 0.0, 255.0, NORM_MINMAX, CV_8U);
   
   return image;
   
}

void featureExtraction()
{
   trainFused.clear();
   testFused.clear();
   
   Mat trainDataPCA,trainDataPCA2,testDataPCA,testDataPCA2;
   trainDataPCA = formatImagesForPCA(trainModality[0]);
   testDataPCA = formatImagesForPCA(testModality[0]);
   
   if(fn_csv.compare(fn_csv2) != 0)
   {
      trainDataPCA2 = formatImagesForPCA(trainModality[1]);
      testDataPCA2 = formatImagesForPCA(testModality[1]);
   }
   
   if(fn_csv.compare(fn_csv2) != 0)
   {
      Mat hConTrain,hConTest;
      
      if ((reductionType == "PCAb" || reductionType == "PCAa")) //PCA
      {
         
         
         if (reductionType == "PCAa")
         {    
            for (int i = 0; i < testModality[0].size();i++)
            {	
               
               //horizontal concatenation
               //weighted 50 50
               if(weigh == 1)
               {
                  if(i < trainModality[0].size())
                  {
                     trainFused.push_back (modalityWeight2 * trainModality[0][i] + modalityWeight1 * trainModality[1][i]);
                  }
                  
                  testFused.push_back (modalityWeight2 * testModality[0][i] + modalityWeight1 * testModality[1][i]);
                  
               }
               else
               {
                  if(i < trainModality[0].size())
                  {
                     
                     
                     hconcat(trainModality[0][i], trainModality[1][i],hConTrain);
                     GaussianBlur (hConTrain, hConTrain, Size(3, 3), 0,0); 
                     resize(hConTrain,hConTrain, Size(hConTrain.cols-1,hConTrain.rows-1),resizeType); 
                     
                     trainFused.push_back(hConTrain);
                  }
                  
                  hconcat(testModality[0][i], testModality[1][i],hConTest);               
                  GaussianBlur (hConTest, hConTest, Size(3, 3), 0,0);
                  resize(hConTest,hConTest, Size(hConTest.cols-1,hConTest.rows-1),resizeType); 
                  
                  testFused.push_back(hConTest);
                  
               }             
               
            }          
            
            Mat trainDataPCAa = formatImagesForPCA(trainFused);
            vector <Mat> pcaResult3 (pcaReduction (trainDataPCAa,trainFused,varRed/100));
            Mat testDataPCAa = formatImagesForPCA(testFused);               
            vector <Mat> pcaResult3t (pcaReduction (testDataPCAa,testFused,varRed/100));
            
            trainFused.clear();
            testFused.clear();
            trainFused.swap (pcaResult3);   
            testFused.swap (pcaResult3t);   
            
            if(matcher == "SVM" || matcher == "LR")
            {
               saveTrainData(trainFused);   
               //             cout << "SVM Train Model Saved" << endl;
               saveTestData(testFused);   
               //             cout << "SVM Test Model Saved" << endl;
               
            }
            
            else
               model->train(trainFused, trainLabels);
            
         }
         
      }      
      else if (reductionType == "Gabor") //Gabor after fusion
      {
         
         vector <Mat> gaborImages;
         vector <Mat> gaborImages2;
         
         for (int i = 0; i < testModality[0].size();i++)
         {	
            //horizontal concatenation
            
            //weighted 50 50
            if(weigh == 1)
            {
               if(i < trainModality[0].size())
               {
                  trainFused.push_back (modalityWeight2 * trainModality[0][i] + modalityWeight1 * trainModality[1][i]);
               }
               
               testFused.push_back (modalityWeight2 * testModality[0][i] + modalityWeight1 * testModality[1][i]);
               
            }
            else
            {
               if(i < trainModality[0].size())
               {
                  
                  
                  hconcat(trainModality[0][i], trainModality[1][i],hConTrain);
                  GaussianBlur (hConTrain, hConTrain, Size(3, 3), 0,0); 
                  resize(hConTrain,hConTrain, Size(hConTrain.cols-1,hConTrain.rows-1),resizeType); 
                  
                  trainFused.push_back(hConTrain);
               }
               
               hconcat(testModality[0][i], testModality[1][i],hConTest);               
               GaussianBlur (hConTest, hConTest, Size(3, 3), 0,0);
               resize(hConTest,hConTest, Size(hConTest.cols-1,hConTest.rows-1),resizeType); 
               
               testFused.push_back(hConTest);
               
            }
            
            trainFused[i] = gaborFilter(trainFused[i], 0, varRed);
            testFused[i] = gaborFilter(testFused[i], 0, varRed);
            
            
            
         }
         
         
         
         if(matcher == "SVM" || matcher == "LR")
         {
            saveTrainData(trainFused);   
            //             cout << "SVM Train Model Saved" << endl;
            saveTestData(testFused);   
            //             cout << "SVM Test Model Saved" << endl;
            
         }
         
         else
            model->train(trainFused, trainLabels);
         //             model->train(trainFused, trainLabels);
         
      }
      
      else if (reductionType == "None")
      { //none
         
         for (int i = 0; i < testModality[0].size();i++)
         {	
            //horizontal concatenation
            
            //weighted 50 50
            if(weigh == 1)
            {
               if(i < trainModality[0].size())
               {
                  trainFused.push_back (modalityWeight2 * trainModality[0][i] + modalityWeight1 * trainModality[1][i]);
               }
               
               testFused.push_back (modalityWeight2 * testModality[0][i] + modalityWeight1 * testModality[1][i]);
               
            }
            else
            {
               if(i < trainModality[0].size())
               {
                  
                  
                  hconcat(trainModality[0][i], trainModality[1][i],hConTrain);
                  GaussianBlur (hConTrain, hConTrain, Size(3, 3), 0,0); 
                  resize(hConTrain,hConTrain, Size(hConTrain.cols-1,hConTrain.rows-1),resizeType); 
                  
                  trainFused.push_back(hConTrain);
               }
               
               hconcat(testModality[0][i], testModality[1][i],hConTest);               
               GaussianBlur (hConTest, hConTest, Size(3, 3), 0,0);
               resize(hConTest,hConTest, Size(hConTest.cols-1,hConTest.rows-1),resizeType); 
               
               testFused.push_back(hConTest);
               
            }
            
         }
         //fusion
         
//          cout<<"test albles "<< testLabels[0] << endl;
//          cout<<"test albles2 "<< trainFused.size() << endl;
         if(matcher == "SVM" || matcher == "LR")
         {
            saveTrainData(trainFused);   
            //             cout << "SVM Train Model Saved" << endl;
            saveTestData(testFused);   
            //             cout << "SVM Test Model Saved" << endl;
            
         }
         
         else
            model->train(trainFused, trainLabels);
         
      }
   }
   
}

int main(int argc, const char *argv[]) {
   // Check for valid command line arguments, print usage
   // if no arguments were given.
   if (argc != 13) {
      cout << "usage: " << argv[0] << " <csv1.ext>" << " <csv2.ext>" << " <samples per person> " << "<model name> "  << "<Matching algorithm (E, F or L)> " << "<1: Normal, 2: Equalize Hist, 3: +LOG, 4: LOG only > " << "<none, PCAa, PCAb, LDA or Gabor  > " << "<Reduction variance %> "<< "<ave or concat %> "<<"<finalSize roi> "<<"noClasses"<<endl;
      exit(1);
   }
   // Get the path to your CSV.
   fn_csv = string(argv[1]);
   fn_csv2 = string(argv[2]);
   samples = atoi(argv[3]);
   matcher = string(argv[5]);
   enhanceType = atoi(argv[6]);	
   reductionType = string(argv[7]);	
   varRed = atof(argv[8]);	
   weigh = atoi(argv[9]);
   finalSize = atoi(argv[10]);
   noClasses = atoi(argv[11]);
   startThresh = atof(argv[12]);
   // These vectors hold the images and corresponding labels.
   eigenThresh = startThresh; // iris
   lbphThresh = startThresh; // iris
   
   if(noClasses == -1 || matcher == "SVM" || matcher == "LR")
      threshIterations = 1;
   
   vector<string> paths;
   vector<string> paths2;
   
   vector<Mat> images;
   vector<Mat> images2;
   
   
   vector<int> labels;
   vector<int> labels2;
   
   // Read in the data. This can fail if no valid
   // input filename is given.
   try {      
      if ((fn_csv.compare(fn_csv2) != 0))
      {
         read_csv(fn_csv, images, labels, paths);
         firstSet = 0;
         read_csv(fn_csv2, images2, labels2, paths2);
      }
      else
         if ((fn_csv.compare(fn_csv2) == 0))
            read_csv(fn_csv, images, labels, paths); 
   } 
   catch (cv::Exception& e) 
   {
      cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
      //       cerr << "Error opening file \"" << fn_csv2 << "\". Reason: " << e.msg << endl;
      // nothing more we can do
      exit(1);
   }
   // Quit if there are not enough images for this demo.
   if(images.size() <= 1) {
      string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
      CV_Error(CV_StsError, error_message);
   }
   // Get the height from the first image. We'll need this
   // later in code to reshape the images to their original
   // size:
   // 	int height = images[0].rows;
   if(noClasses == -1)//closed-set identification
      noClasses = labels[labels.size()-1]+1;
   cout<<noClasses<<endl;
   assignImages(images,labels,paths, samples);  
   
   trainModality[0].swap (trainImages);   
   testModality[0].swap (testImages);  
   
//       cout<<"total Images1 "<< images.size() << endl;
//       cout<<"train Images1 "<< trainModality[0].size() << endl;
//       cout<<"test Images1 "<< testModality[0].size() << endl;
   
   if ((fn_csv.compare(fn_csv2) != 0))
   {
      
      
      assignImages(images2,labels2,paths2, samples);
      
      trainModality[1].swap (trainImages);  
      testModality[1].swap (testImages);   
      
      //       cout<<"\ntotal Images2 "<< images.size() << endl;
      //       cout<<"train Images2 "<< trainModality[0].size() << endl;
      //       cout<<"test Images2 "<< testModality[0].size() << endl;
   }
   
   
   double current_threshold;
   
   for(int i = 0;i<threshIterations;i++)//thresh...uncomment to activate
   {
      if (matcher == "E")
      {
         if(i == threshIterations-1) //no thresh
            model =  createEigenFaceRecognizer(eigenNum);
         else
            model =  createEigenFaceRecognizer(eigenNum, eigenThresh);
         
         
         // 		model =  createEigenFaceRecognizer();
//          			cout<<"eigenThresh "<<eigenThresh<<endl;			
         
         // 		current_threshold = model->getDouble("threshold");
         // 		cout << "current_threshold " << current_threshold << endl;
         //       cout << "EIGEN Model " << argv[4] << " Saved" << endl;
      }
      
      if (matcher == "F")
      {
         
         if(i == threshIterations-1) //no thresh
            model =  createFisherFaceRecognizer(eigenNum*1.1);
         else
            model = createFisherFaceRecognizer(eigenNum, eigenThresh);
         
         // 			model = createFisherFaceRecognizer(eigenNum*2);
         // 		model = createFisherFaceRecognizer();
         // 		current_threshold = model->getDouble("threshold");
         
         //       cout << "FISHER Model " << argv[4] << " Saved" << endl;
         
      }
      
      if (matcher == "L")
      {
         if(i == threshIterations-1) //no thresh
            model =  createLBPHFaceRecognizer(lbphRadius, lbphNeighbours,lbphNeighbours,lbphNeighbours);
         else
            model =  createLBPHFaceRecognizer(lbphRadius, lbphNeighbours,lbphNeighbours,lbphNeighbours, lbphThresh);
         
         
         
         // 		current_threshold = model->getDouble("threshold");
         // 			cout<<"LBPH lbphThresh "<<lbphThresh<<endl;			
         
         //       		model =  createLBPHFaceRecognizer();
         //       cout << "LBP Histogram Model " << argv[4] << " Saved" << endl;
         
      }
      
      string basename = argv[4]; // take the whole path, including dat extension (DAT EXTENSION!)
      basename.resize(basename.length() - 3); // remove dat extension (hah!)
      outname = basename + "txt"; // add the output extension	
      
      featureExtraction();
      
      
      
      if ((fn_csv.compare(fn_csv2) == 0))//no fusion
      {
           
         if(matcher == "SVM" || matcher == "LR")
         {
            saveTrainData(trainModality[0]);   
            //          cout << "SVM Train Model Saved" << endl;
            saveTestData(testModality[0]);   
            //          cout << "SVM Test Model Saved" << endl;
            
         }
         else
            model->train(trainModality[0], trainLabels);
         
      }  
      // 	save the model to eigenfaces_at.yaml
      //    model->save(argv[4]);
      
      
      
      
      //test
      
      if(matcher != "SVM" && matcher != "LR")
      {
         
         ofstream file1;
         file1.open(outname.c_str(),ios::trunc);
         
         int cnt2 = 0;
         
         vector < vector<int> > matrix;
         int matrixN = 0;
         vector<int> falseMode(testModality[0].size(), 0);
         
         
         vector<int> matchResult;
         float detectionAndIdentification = 0;
         float genuineReject = 0;
         float falseAccept = 0;
         float falseReject = 0;
         float miss = 0;
         int predictedLabel = -1;
         int testLabel = -1;
         // 			int oldTestLabel = 0;
         int oldTestLabel = testLabels [0];
         
         
         double confidence = 0.0;
         
         
         //          cout<<"test Images1 "<< testModality[0].size() << endl;
//          cout<<"test Images2 "<< testModality[1].size() << endl;
         int check1 = 0;
         int check2 = 0;
         for(int i = 0;i < testModality[0].size();i++)
         {
            
            Mat testSample;
            
            
            
            if ((fn_csv.compare(fn_csv2) == 0))//no fusion
               testSample = testModality[0] [i];
            else
               testSample = testFused [i];
            
//             imshow("test",testFused [i]);
//             waitKey();
            testLabel = testLabels [i];
            
            predictedLabel = model->predict(testSample);
//             cout<<"works till here "<<predictedLabel<<endl;
            // To get the confidence of a prediction call the model with:
            //
            // 		model->predict(testSample, predictedLabel, confidence);
            // 		cout << "confidence "<<confidence << endl;
            
//             cout << predictedLabel<< " testLabel "<<testLabel << endl;
            
            
            //test results
            if(predictedLabel == testLabel) //definitely correct
            {
               if(testLabel == oldTestLabel)
               {
                  file1 << predictedLabel << " ";
                  // 					cout << predictedLabel << " ";
               }
               else
               {
//                   file1 <<"" << endl;
//                   // 					cout <<"" << endl;
//                   file1 << predictedLabel << " ";
                  // 					cout << predictedLabel << " ";
//                   genuineReject++;
                  
               }
               
               file1 <<"" << endl;
               
               detectionAndIdentification++;
               
               check1++;
            }
            
            
            
            else //either genuineReject, false or miss
            {
               if (predictedLabel == -1)
               {
                  if(testLabel == -2) //correctly predict impostor
                  {
                     genuineReject++;
                     
                  }               
                  else //incorrectly reject a class
                  {
                     falseReject++;
                     
                  }
               }
               else //false accept impostor or wrong guess of a class
               {
                  if (testLabel != -2) //missclassify (wrong guess of a class)
                  miss++;
               else  //falsely accept impostor
                  falseAccept++;
               check2++;
               }
               
            }
            
            
            
         } //end scoring
//          cout<<"check1 "<< check1 << endl;
//          cout<<"check2 "<< check2 << endl;
         
         
//          if(noClasses = -1)
//             testClassSize = testModality[0].size();

         float nonClassSize = testModality[0].size() - testClassSize;
//          cout<<"test Images1 "<< testModality[0].size() << endl;
         
//          cout <<"\n Class test images " << testClassSize << endl;
         
//          cout <<"detectionAndIdentification " << detectionAndIdentification << endl;
//          cout <<"miss " << miss <<endl;         
//          cout <<"genuineReject " << genuineReject << endl;
         
//          cout <<"\n Non-Class test images " << nonClassSize  << endl;
         
//          cout <<"falseAccept " << falseAccept << endl;         
//          cout <<"falseReject " << falseReject <<endl;
         

         
         //classes
            dir = ((detectionAndIdentification) / (float)testClassSize) * 100;
            mr = (miss / (float)testClassSize) * 100;
         
            //non-classes
//             frr = (falseReject / (float)nonClassSize) * 100;
            frr = 100 - (dir+mr);            
            if(nonClassSize>0)
               fpir = (falseAccept / (float)nonClassSize) * 100;
            
//             eigenThresh += 45 * (finalSize/50);
            eigenThresh += 25 * (finalSize/40);//palm
//             lbphThresh += 0.05;
//             lbphThresh += 0.1 * (finalSize/50);//finger palm
//             lbphThresh += 0.1 * (finalSize/40);//face
//             lbphThresh += 0.01 * (finalSize/40);//iris
            lbphThresh += 0.1 * (finalSize/40);//fusion
            svmRegressionBias +=0.1 * (finalSize/50);

//          eigenThresh *= 1.15;
//          lbphThresh *= 1.35;
//          svmRegressionBias /=1.1;
         if(matcher == "E" || matcher == "F") 
         {
            // 				cout<<"eigenNum "<<eigenNum<<endl;			
            // 				eigenNum += 50;
         }
         
         if(matcher == "L") 
         {
            // 				cout<<"LBPH lbpNeighbourSize "<<lbphNeighbours<<endl;			
            // 				cout<<"LBPH lbphRadius "<<lbphRadius<<endl;			
            // 				lbphNeighbours++;
            // 								lbphRadius ++;
            
         }
         
         cnt2++;
         
         
         //       file1.close();	
         
         //       file1.open(outname.c_str(),ios::app);
  
         std::cout << std::setprecision(1) << std::fixed;
         
         if(i == threshIterations-1)               
            cout <<mr <<",\t "<<100<<",\t "<< dir << endl;
         
         if(fpir > 0 && fpir < 0.45 && zero ==0)
         {
            if(nonClassSize>0)
               cout <<mr<<",\t "<<fpir<<",\t "<< dir << endl;
            //          cout <<mr + fpir <<", "<< dir << endl;
            zero = 1;
         }
         
         if(fpir > 0.45 && fpir < 0.9 && zeroFive ==0)
         {
            cout <<mr<<",\t "<<fpir<<",\t "<< dir << endl;
            zeroFive = 1;
         }
         
         if(fpir > 0.9 && fpir < 1.5 && one ==0)
         {
            cout <<mr<<",\t "<<fpir<<",\t "<< dir << endl;
            one = 1;
         }
         
         if(fpir > 4.5 && fpir < 7 && five ==0)
         {
            cout <<mr<<",\t "<<fpir<<",\t "<< dir << endl;
            five = 1;
         } 
         
         if(fpir > 9 && fpir < 13 && ten ==0)
         {
            cout <<mr<<",\t "<<fpir<<",\t "<< dir << endl;
            ten = 1;
         }
         if(fpir > 28  && thirty ==0)
         {
            cout <<mr<<",\t "<<fpir<<",\t "<< dir << endl;
            thirty = 1;
         }     
            
//              cout <<mr<<", "<<fpir<<", "<< dir << endl;    
         
//          cout << "detectionAndIdentification rate (DIR): " << dir << "%" << endl;
//          cout << "Missclassification rate (MIR): " << mr << "%" << endl;
//          file1 << "dir: " << dir << "%" << endl;
//          
//          cout << "False Acceptance/Alarm Rate: " << fpir << "%" << endl;
//          file1 << "fpir: " << bestFalseAcceptPercent << "%" << endl;
//          
//             cout << "False Rejection Rate: " << frr << "%" << endl;
//          file1 << "FRR: " << bestFalseRejectPercent << "%" << endl;
         
         file1.close();
         
      }
      
      if(i == threshIterations-1)
         break;
      
      if(fpir >30)
         i = threshIterations-2;
      
   
   }
   return 0;
}
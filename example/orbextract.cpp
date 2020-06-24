#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/aruco.hpp>

#include "include/ORBextractor.h"
#include "include/Vertex.h"
#include "include/triangle.h"
#include "include/delaunay.h"
#include "include/DTMunit.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

#define d_max_value 50      // 暴力匹配的阈值
#define m_max_value 5       // DTM边矩阵相似度阈值

#define d_ransac_value 50
#define threshold_value 4  // 15
/**
 * @brief DTM
 * 1.分别对两幅图像进行特征提取；
 * 2.进行特征匹配；
 * 3.对第一次匹配的good_matches进行构建DT网络；
 */

/// 主函数
int main()
{
    /// test
    string strSettingPath = "/home/lu/code/ORB-DTM/config/test.yaml";
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    cout << "Test: " << fx << endl;
    string test_name = fSettings["word"];
    cout << "name: " << test_name << endl;

//    cv::Mat markerImage;
//    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
//    cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);

//    struct timespec time1 = {0, 0};       // 用于计时
//    struct timespec time2 = {0, 0};

//    string file1 = "./data/desk1.png";    // 1500 18  12
//    string file2 = "./data/desk2.png";
//    string file1 = "./data/n1.png";    // 1500 18  12
//    string file2 = "./data/n2.png";
//    string file1 = "./data/flag1.png";      // 1000 28  15  18 12
//    string file2 = "./data/flag2.png";
//    string file1 = "./data/draw1.png";      // 2500 28  15
//    string file2 = "./data/draw2.png";
//    string file1 = "./data/graf/img4.ppm";      // 2500 28  15
//    string file2 = "./data/graf/img5.ppm";
//    string file1 = "./data/newspaper1.jpg";      // 2500 28  15
//    string file2 = "./data/newspaper2.jpg";
//    string file1 = "./data/barks/img3.ppm";      // 2500 28  15
//    string file2 = "./data/barks/img4.ppm";

    string file1 = "./data/tum/1305031127.411371.png";    // 1500 18  12
    string file2 = "./data/tum/1305031127.447333.png";

//    string depth_name = "./data/tum/1305031127.419650.png";
//    cv::Mat depth_image = cv::imread(depth_name, 0);
//    cout << "类型: " << depth_image.type() << endl;
//    cout << "\n深度值: " << depth_image.at<int>(cv::Point(73, 323)) << endl;    // float 7.05334e-30
    /**************** 配置信息 ******************/
    int nFeatures =2000;        // 特征点数量 800
    float fScaleFactor =1.2;    // 图像金字塔的缩放尺度
    int nLevels =8;             // 金字塔层数
    int fIniThFAST =18;         // 提取FAST角点的阈值  两个阈值进行选择 18  8
    int fMinThFAST =8;          // 此阈值越高,角点质量越好;选取范围0-255
    // 初始匹配应更严格,阈值应更高

    int level = 0;      // 特定层数得到的源图像

//    cout << "显示特征提取的基本信息：" << endl;

//    vector< vector<cv::KeyPoint> > mvvKeys1;
//    mvvKeys1.resize(8);
    /**************** 图片一：初始化信息 *********************/
    cv::Mat first_image = cv::imread(file1, 0);    // load grayscale image 灰度图
    cv::Mat feature1;
    std::vector<cv::Mat> mvImageShow1;   //图像金字塔
    vector<cv::KeyPoint> mvKeys1_all;        //一维特征点 所有特征点
    vector<cv::KeyPoint> mvKeys1;        //一维特征点 最底层的特征点
    cv::Mat mDescriptors1;               //描述子
    vector<int> mnFeaturesPerLevel1;     //金字塔每层的特征点数量
//    vector<vector<cv::KeyPoint>> mvvKeypoints1;  //每层的特征点
    cv::Mat mDes1;
    //    mDes1.convertTo(mDes1,CV_32F);

//    arrowedLine(first_image, cv::Point(100, 100), cv::Point(100, 100 - 30), Scalar(0, 0, 255), 1, 8);
//    arrowedLine(first_image, cv::Point(105, 100), cv::Point(105, 100 - 30), Scalar(0, 0, 255), 1, 8);
//    imshow("test",first_image);
//    waitKey(0);
    /**************** 图片一：提取特征点信息 ******************/
    auto *orb1 = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    (*orb1)(first_image,cv::Mat(),mvKeys1_all,mDescriptors1);

    mvImageShow1 = orb1->GetImagePyramid();   //获取图像金字塔
    mnFeaturesPerLevel1 = orb1->GetmnFeaturesPerLevel();  //获取每层金字塔的特征点数量

//    cout << "显示每层金字塔的特征点数目" << endl;
//    int count = 0, sum = 0;
//    for (auto &p:mnFeaturesPerLevel1)
//    {
//        cout << count++ << " : " << p << endl;
//        sum += p;
//    }
//    cout << "sum = " << sum << endl;

    int class_id = 0;
    for (auto &p:mvKeys1_all) {
        /// 取出level0的关键点,存在mvKeys1中; 并且进行重新编号
        if (p.octave == level && class_id < mnFeaturesPerLevel1[level]) {
            // mvKeys1.emplace_back(cv::KeyPoint(p.pt, p.size, p.angle, p.response, p.octave, p.class_id));
            mvKeys1.emplace_back(cv::KeyPoint(p.pt, p.size, p.angle, p.response, p.octave, class_id++));
        }
    }


//    mvvKeypoints1 = orb1->GetmvvKeypoints();
//    cout << "显示mvvKeypoints 的信息: " << endl;
//    for(auto &p:mvvKeypoints1)
//    {
//        cout << p.size() << endl;
//    }

    mDes1 = mDescriptors1.rowRange(0,mnFeaturesPerLevel1[level]).clone();

    cv::drawKeypoints(mvImageShow1[level], mvKeys1, feature1, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DEFAULT);//DEFAULT  DRAW_OVER_OUTIMG     DRAW_RICH_KEYPOINTS
//    imshow("Mat1", feature1);
//    waitKey(0);


    // todo : 使用高斯金字塔的尺度不变性,解决圆形ROI内汉明距离存在多个值相同的问题
    /**************** 图片二：初始化信息 *********************/
    cv::Mat second_image = cv::imread(file2, 0);    // load grayscale image 灰度图
    cv::Mat feature2;
    std::vector<cv::Mat> mvImageShow2;   //图像金字塔
    vector<cv::KeyPoint> mvKeys2_all;        //一维特征点 所有特征点
    vector<cv::KeyPoint> mvKeys2;        //一维特征点 最底层的特征点
    cv::Mat mDescriptors2;               //描述子
    vector<int> mnFeaturesPerLevel2;     //金字塔每层的特征点数量
    vector<vector<cv::KeyPoint>> mvvKeypoints2;  //每层的特征点
    cv::Mat mDes2;
    //    mDes2.convertTo(mDes2,CV_32F);
    /**************** 图片二：提取特征点信息 ******************/
    ORBextractor *orb2 = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    (*orb2)(second_image,cv::Mat(),mvKeys2_all,mDescriptors2);

    mvImageShow2 = orb2->GetImagePyramid();   //获取图像金字塔

    mnFeaturesPerLevel2 = orb2->GetmnFeaturesPerLevel();  //获取每层金字塔的特征点数量

    class_id = 0;
    for (auto &p:mvKeys2_all) {
        if (p.octave == level && class_id < mnFeaturesPerLevel2[level]){
            //            mvKeys2.emplace_back(cv::KeyPoint(p.pt,p.size,p.angle,p.response,p.octave,p.class_id));
            mvKeys2.emplace_back(cv::KeyPoint(p.pt,p.size,p.angle,p.response,p.octave,class_id++));
        }
    }

    //    mvvKeypoints2 = orb2->GetmvvKeypoints();
    //    mvKeys2 = mvvKeypoints2[level];
    mDes2 = mDescriptors2.rowRange(0,mnFeaturesPerLevel2[level]).clone();

    cv::drawKeypoints(mvImageShow2[level], mvKeys2, feature2, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DEFAULT);//DEFAULT  DRAW_OVER_OUTIMG     DRAW_RICH_KEYPOINTS

//    imshow("Mat2", feature2);
//    waitKey(0);
    /***************   克隆图片   ******************/
    //    Mat debugOne   = feature1.clone();
    //    Mat debugTwo   = feature2.clone();

    //    cout << "size of key1: " << mvKeys1.size() << endl;
    //    cout << "size of key2: " << mvKeys2.size() << endl;
    //    imshow("pic1", feature1);
    //    waitKey(0);
    //    imshow("pic2", feature2);
    //    waitKey(0);
    /***************   特征匹配   *************/
    //    vector<DMatch> good_matches( BFmatchFunc(mDes1,mDes2,d_max_value) );
    //    cout <<"init size:\t" << good_matches.size() << endl;
    //    vector<DMatch> good_matches( KNNmatchFunc(mDes1, mDes2) );
    /***************  构建DT网络  ******************************/
    //    vector<DMatch> new_matches(ComputeDTMunit(m_max_value, good_matches, mvKeys1, mvKeys2, debugOne, debugTwo) );   //5
    //    cout <<"size one:\t" << new_matches.size() << endl;
    /***************  RANSAC 实验对照组  ******************************/
//    cout << "\n采用RANSAC作为control group的实验结果：";
    //    clock_gettime(CLOCK_REALTIME, &time1);
    vector<DMatch> control_matches( BFmatchFunc(mDes1,mDes2,d_ransac_value) );
    //    vector<DMatch> control_matches( KNNmatchFunc(mDes1, mDes2) );

//    cv::evaluateFeatureDetector();

    UsingRansac(threshold_value,feature1,feature2,mvKeys1,mvKeys2,mDes1,mDes2,control_matches);
    //    clock_gettime(CLOCK_REALTIME, &time2);
    //    cout << "time passed is: " << (time2.tv_sec - time1.tv_sec)*1000 + (time2.tv_nsec - time1.tv_nsec)/1000000 << "ms" << endl;
    /****************************************/
    cout << "\nmain end, see you...";
    return 0;
}
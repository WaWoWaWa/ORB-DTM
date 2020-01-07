//
// Created by lu on 19-5-11.
//
#include "DTMunit.h"

/**
 * @brief 构建DTM的基本函数
 * @param initGood_matches
 * @param mvKeys1
 * @param mvKeys2
 * @param feature1
 * @param feature2
 * @return newGood_matches
 */
vector<DMatch> ComputeDTMunit(int threshold, const vector<DMatch> &initGood_matches, const vector<cv::KeyPoint> &mvKeys1, const vector<cv::KeyPoint> &mvKeys2, cv::Mat &feature1, cv::Mat &feature2 )
{
    if (initGood_matches.empty())
        return initGood_matches;
    Mat feature3 = feature1.clone();
    Mat feature4 = feature2.clone();
    ///delaunay one
//    cout << "DT one:" << endl;
    vector<Vertex<float > > points1;
    for(const auto &p:initGood_matches)
    {
        points1.emplace_back(Vertex<float>(mvKeys1[p.queryIdx].pt.x , mvKeys1[p.queryIdx].pt.y , p.queryIdx ));
//        cout << "id1: " << p.queryIdx << endl;
    }

    Delaunay<float> triangulation1;
    const std::vector<Triangle<float> > triangles1 = triangulation1.Triangulate(points1);  //逐点插入法
    triangulation1.ComputeEdgeMatrix();
//    std::cout << "\t\t" <<triangles1.size() << " triangles generated"<<endl;
    const std::vector<Edge<float> > edges1 = triangulation1.GetEdges();

    for(const auto &e : edges1)
    {
        line(feature1, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }

//    for(const auto &t : triangles1)
//    {
//        double sideLength;
//        sideLength = sqrt(  (t.mainpoint.x-t.circum.x)*(t.mainpoint.x-t.circum.x) + (t.mainpoint.y-t.circum.y)*(t.mainpoint.y-t.circum.y)  );
//        cout << "sidelength: " << sideLength << endl;
//        if (sideLength < MAX_ARROR_SIZE) {
//            circle(feature1, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
//            circle(feature1, Point(t.circum.x, t.circum.y), t.p1.dist(t.circum), Scalar(255, 0, 0));
//            arrowedLine(feature1, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),
//                        1, 8);
//        }
//    }

    ///delaunay two
//    cout << "DT two:" << endl;
    vector<Vertex<float > > points2;
    for(const auto &p:initGood_matches)
    {
        points2.emplace_back(Vertex<float>(mvKeys2[p.trainIdx].pt.x , mvKeys2[p.trainIdx].pt.y , p.trainIdx ));
//        cout << "id2: " << p.trainIdx << endl;
    }

    Delaunay<float> triangulation2;
    const std::vector<Triangle<float> > triangles2 = triangulation2.Triangulate(points2);  //逐点插入法
    triangulation2.ComputeEdgeMatrix();
//    std::cout << "\t\t" <<triangles2.size() << " triangles generated"<<endl;
    const std::vector<Edge<float> > edges2 = triangulation2.GetEdges();

    for(const auto &e : edges2)
    {
        line(feature2, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }

//    for(const auto &t : triangles2)
//    {
//        double sideLength;
//        sideLength = sqrt(  (t.mainpoint.x-t.circum.x)*(t.mainpoint.x-t.circum.x) + (t.mainpoint.y-t.circum.y)*(t.mainpoint.y-t.circum.y)  );
//        cout << "sidelength: " << sideLength << endl;
//        if (sideLength < MAX_ARROR_SIZE) {
//            circle(feature2, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
//            arrowedLine(feature2, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),
//                        1, 8);
//        }
//    }

    /**************** 显示匹配结果与初始DT网络 ******************/
//    cout << "\t匹配:" << endl;
//    cout << "\t\tmatch:" << initGood_matches.size()<<endl;

//    Mat beforeOpt;
//    cv::drawMatches(feature1,mvKeys1,feature2,mvKeys2,initGood_matches,beforeOpt);
//    imshow("before optimization",beforeOpt);
//    imwrite("./figure/beforeDTM.png",beforeOpt);
//    waitKey(0);

    return initGood_matches;
/*******************  构建边矩阵，并计算相似度(范数)，进行DT网络的优化  *********************/
//    cout << "\n计算DTM的相关信息：" << endl;
//    Eigen::MatrixXd::Index maxRow,maxCol;
    Eigen::MatrixXd edgeMatrix = Eigen::MatrixXd::Zero(sizeofEdgeMatrix,sizeofEdgeMatrix);  //ComputeEdgeMatrix() 在此处也修改了 20,20 ，需要同步修改，后期改进此处
    edgeMatrix = triangulation1.GetEdgeMatrix() - triangulation2.GetEdgeMatrix();
    //    double value =0;
    //    value = edgeMatrix_.norm();
    //    cout << "\tvalue: " << value <<  endl;      // 相似度

//    edgeMatrix.cwiseAbs().colwise().sum().maxCoeff(&maxRow,&maxCol);    // 边矩阵.绝对值.列.和.最大值(行序号,列序号)

//    cout << "提取候选外点：\t"  << maxCol << endl;
//    cout << "显示sum:\n" << edgeMatrix_.cwiseAbs().colwise().sum() << endl;
//    cout << "计算列和：\n" << edgeMatrix_.cwiseAbs().colwise().sum()<< endl;
//    cout << "显示边矩阵之差：\n"<< edgeMatrix_.cwiseAbs().col(maxCol).transpose() << endl;
//    cout << "二者之差：\n" << edgeMatrix_.cwiseAbs().colwise().sum() - edgeMatrix_.cwiseAbs().col(maxCol).transpose()<< endl;
//    cout << "候选外点：" << mvKeys2[good_matches[maxCol].trainIdx].pt << endl;

    // 通过DT网络的边矩阵之差的范数，删除列和较大的候选外点集
    vector<DMatch> newGood_matches(initGood_matches);

//    cout << "\nold size:\t" << newGood_matches.size()<<endl;
    for(int i = newGood_matches.size();i != 0 ;i--)
    {
        if((edgeMatrix.cwiseAbs().colwise().sum())(0,i-1) >= threshold )
        {
//            cout << (edgeMatrix_.cwiseAbs().colwise().sum())(0,i-1) << "\t,\t" << mvKeys1[newGood_matches[i-1].queryIdx].pt <<"\t,\t" << mvKeys2[newGood_matches[i-1].trainIdx].pt << endl;
            newGood_matches.erase(newGood_matches.begin()+i-1);
        }
    }
    cout << "new size:\t" << newGood_matches.size()<<endl;

    double angle(0);
    auto it=newGood_matches.begin(),ng_end = newGood_matches.end();
//    for (const auto &p:newGood_matches) {
    for (;it != ng_end; ++it) {
//        cout << mvKeys1[p.queryIdx].angle << "\t,\t" << mvKeys2[p.trainIdx].angle << "\t,\t" <<
//             mvKeys1[p.queryIdx].angle - mvKeys2[p.trainIdx].angle << endl;
        angle = mvKeys1[it->queryIdx].angle - mvKeys2[it->trainIdx].angle;
        if (angle > 180)
            angle -= 360;

        if ( angle < -12 || angle > 12){
            cout << angle << endl;
            newGood_matches.erase(it);
        }
    }

    /************ 显示优化后的DT网络 ****************/

    if (newGood_matches.empty())
        return newGood_matches;
    ///delaunay three
    std::vector<Vertex<float> > points3;
    for(const auto &g:newGood_matches)
    {
        points3.emplace_back(Vertex<float>(mvKeys1[g.queryIdx].pt.x , mvKeys1[g.queryIdx].pt.y , g.queryIdx ));
    }
    Delaunay<float> triangulation3;
    const std::vector<Triangle<float> > triangles3 = triangulation3.Triangulate(points3);  //逐点插入法
    triangulation3.ComputeEdgeMatrix();
//    std::cout << "\t\t" << triangles3.size() << " triangles generated"<<endl;
    const std::vector<Edge<float> > edges3 = triangulation3.GetEdges();
    for(const auto &e : edges3)
    {
        line(feature3, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }

//    for(const auto &t : triangles3)
//    {
//        double sideLength;
//        sideLength = sqrt(  (t.mainpoint.x-t.circum.x)*(t.mainpoint.x-t.circum.x) + (t.mainpoint.y-t.circum.y)*(t.mainpoint.y-t.circum.y)  );
//        cout << "sidelength: " << sideLength << endl;
//        if (sideLength < MAX_ARROR_SIZE) {
//            circle(feature3, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
//            circle(feature3, Point(t.circum.x, t.circum.y), t.p1.dist(t.circum), Scalar(0, 0, 255));
//            arrowedLine(feature3, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),
//                        1, 8);
//        }
//    }

    ///delaunay four

//    cout << "\tDT four:" << endl;
    std::vector<Vertex<float> > points4;
    for(const auto &g:newGood_matches)
    {
        points4.emplace_back(Vertex<float>(mvKeys2[g.trainIdx].pt.x , mvKeys2[g.trainIdx].pt.y , g.trainIdx ));
    }

    Delaunay<float> triangulation4;
    const std::vector<Triangle<float> > triangles4 = triangulation4.Triangulate(points4);  //逐点插入法
    triangulation4.ComputeEdgeMatrix();
//    std::cout << "\t\t" << triangles4.size() << " triangles generated"<<endl;
    const std::vector<Edge<float> > edges4 = triangulation4.GetEdges();

    for(const auto &e : edges4)
    {
        line(feature4, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }

//    for(const auto &t : triangles4)
//    {
//        double sideLength;
//        sideLength = sqrt(  (t.mainpoint.x-t.circum.x)*(t.mainpoint.x-t.circum.x) + (t.mainpoint.y-t.circum.y)*(t.mainpoint.y-t.circum.y)  );
//        cout << "sidelength: " << sideLength << endl;
//        if (sideLength < MAX_ARROR_SIZE) {
//            circle(feature4, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
//            circle(feature4, Point(t.circum.x, t.circum.y), t.p1.dist(t.circum), Scalar(0, 0, 255));
//            arrowedLine(feature4, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),
//                        1, 8);
//        }
//    }

    Mat afterOpt;
    cv::drawMatches(feature3,mvKeys1,feature4,mvKeys2,newGood_matches,afterOpt);
    imshow("after optimization",afterOpt);
    imwrite("./figure/DTM.png",afterOpt);
    waitKey(0);

    /***********************************************/
//    cout << "Finished in function!!!" << endl;
    return newGood_matches;
}

/**
 * @brief 获取剩余点集
 *
 * 输入
 * @param sizeofLevel             剩余点个数
 * @param good_matches
 * @param mvKeys1
 * @param mvKeys2
 * @param mDesc1
 * @param mDesc2
 *
 * 输出
 * @param mvKeys1_new
 * @param mvKeys2_new
 * @param mDes1_new
 * @param mDes2_new
 */
void UpdateKey(const vector<DMatch> &good_matches, const vector<cv::KeyPoint> &mvKeys1, const vector<cv::KeyPoint> &mvKeys2, const cv::Mat &mDes1, const cv::Mat &mDes2,
               vector<cv::KeyPoint> &mvKeys1_new, vector<cv::KeyPoint> &mvKeys2_new, cv::Mat &mDes1_new, cv::Mat &mDes2_new)
{
    //   cv::Mat中没有删除某一列或者行的函数
    //   只能构造新的Mat，在删除某一列后，将后边的复制到新的Mat当中去
    //   新的解决方案是：将Mat转换为vector，使用back() pop()等操作处理后，再转换成Mat
    //   注意：由于删除的是列，而转换成vector后操作的是行，因此可以对Mat进行转置后，再进行转换操作，即Mat.t()
    //   在循环外边完成Mat到vector的转换工作，进行循环操作并退出后，再进行转换回来
    vector<int> order1,order2;
//    cout << "Size of goodmatchs:  " << good_matches.size() << endl;
    // 更新特征点
    for(const auto &g:good_matches)
    {
        order1.emplace_back(g.queryIdx);
        order2.emplace_back(g.trainIdx);
    }
    sort(order1.begin(),order1.end());
    sort(order2.begin(),order2.end());

    // 更新描述子
    int dele_temp_1=0;
    int dele_temp_2=0;
    int dele_temp_count1=0;
    int dele_temp_count2=0;
    for (int i = 0; i < 217; ++i)  //mvKeys1.size()
    {
        if(i == *(order1.begin()+dele_temp_count1))     // 如果与order中的序号相同，则跳过该点
            dele_temp_count1++;
        else
        {
            mvKeys1_new.insert(mvKeys1_new.end(),mvKeys1.begin()+i,mvKeys1.begin()+i+1);
            mDes1.row(i).copyTo(mDes1_new.row(dele_temp_1));
            dele_temp_1++;
        }

        if(i == *(order2.begin()+dele_temp_count2))
            dele_temp_count2++;
        else
        {
            mvKeys2_new.insert(mvKeys2_new.begin()+dele_temp_2,mvKeys2.begin()+i,mvKeys2.begin()+i+1);
            mDes2.row(i).copyTo(mDes2_new.row(dele_temp_2));
            dele_temp_2++;
        }

    }
//    cout << "Sizes of mvKeys1_new: \t" << mvKeys1_new.size() << endl;
//    cout << "Sizes of mDes1_new:\t\t" << mDes1_new.size << endl;
//    cout << "Sizes of mvKeys2_new: \t" << mvKeys2_new.size() << endl;
//    cout << "Sizes of mDes2_new:\t\t" << mDes2_new.size << endl;

}

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

// sort()时，自定义的排序条件
// 用于对vector对象内的指定成员进行排序
//inline bool cmpTrainIdxUp(const DMatch first, const DMatch second)
inline bool cmpTrainIdxUp(const DMatch first, const DMatch second)
{
    return first.trainIdx < second.trainIdx;
}

inline bool cmpTrainIdxDown(const DMatch first, const DMatch second)
{
    return first.trainIdx > second.trainIdx;
}

// unique()时，自定义的去重条件
// 用于对vector对象内的指定成员进行去重
//inline bool cmpTrainIdxEqual(const DMatch first,const DMatch second)
inline bool cmpTrainIdxEqual(const DMatch first,const DMatch second)
{
    return first.trainIdx == second.trainIdx;
}
/**
 * @brief 使用BF匹配
 * @param mDes1
 * @param mDes2
 * @return
 */
vector<DMatch> BFmatchFunc(const cv::Mat &mDes1, const cv::Mat &mDes2, int threshold)
{
//    cout << "\n显示第一次特征匹配的基本信息：" << endl;
    vector<DMatch> matches,good_matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(mDes1,mDes2,matches);

    //计算最大与最小距离
    double min_dist = 10000,max_dist = 0;

    for (int k = 0; k < mDes1.rows; k++)
    {
        double dist = matches[k].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }

//    cout << "\tmin_dist:" << min_dist << endl;
//    cout << "\tmax_dist:" << max_dist << endl;

    //筛选匹配
    int temp=0;
    for (int l = 0; l < mDes1.rows; l++)
    {
        if(matches[l].distance <= threshold )
        {
            matches[l].imgIdx=temp;
            good_matches.emplace_back(matches[l]);
            temp++;
        }
    }
    temp=0;

    sort(good_matches.begin(), good_matches.end(), cmpTrainIdxUp);   //排序
    good_matches.erase(unique(good_matches.begin(),good_matches.end(),cmpTrainIdxEqual),good_matches.end());    //去重

    // 对新的排列重新赋值index
    for(int i =0 ;i < good_matches.size();i++)
    {
        good_matches[i].imgIdx = i;
    }

    return good_matches;
}

/**
 * @brief 使用KNN匹配
 * @param mDes1
 * @param mDes2
 * @return
 */
vector<DMatch> KNNmatchFunc(cv::Mat &mDes1, cv::Mat &mDes2)
{
    if( mDes1.type()!=CV_32F )
    {
        mDes1.convertTo( mDes1, CV_32F );
        mDes2.convertTo( mDes2, CV_32F );
    }

    const float minRatio = 1.f/1.2f;
    const int k = 2;

    vector<vector<DMatch> > knnmatches;
    vector<DMatch> good_matches;

//    const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams();
//    const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams() ;
//    const Ptr<flann::IndexParams>& indexParams = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1);

    FlannBasedMatcher matcher;
//    matcher
    matcher.knnMatch(mDes1, mDes2, knnmatches, k);

    for (std::size_t i = 0; i < knnmatches.size(); ++i)
    {
        const DMatch& bestMatch = knnmatches[i][0];
        const DMatch& betterMatch = knnmatches[i][1];
        float distanceRatio = bestMatch.distance/betterMatch.distance;
        if (distanceRatio < minRatio)
            good_matches.emplace_back(bestMatch);
    }

    sort(good_matches.begin(), good_matches.end(), cmpTrainIdxUp);   //排序
    good_matches.erase(unique(good_matches.begin(),good_matches.end(),cmpTrainIdxEqual),good_matches.end());    //去重

    // 对新的排列重新赋值index
    for(int i =0 ;i < good_matches.size();i++)
    {
        good_matches[i].imgIdx = i;
    }

    return good_matches;
}

// sort()时，自定义的排序条件
// 用于对vector对象内的指定成员进行排序
inline bool cmpQueryIdxUp(const DMatch first, const DMatch second)
{
    return first.queryIdx < second.queryIdx;
}

inline bool cmpQueryIdxDown(const DMatch first, const DMatch second)
{
    return first.queryIdx > second.queryIdx;
}

// unique()时，自定义的去重条件
// 用于对vector对象内的指定成员进行去重
inline bool cmpQueryIdxEqual(const DMatch first,const DMatch second)
{
    return first.queryIdx == second.queryIdx;
}

/**
 * @brief 封装成函数
 *
 * 输入：debugOne,mvKeys1,debugTwo,mvKeys2,control_matches
 * 输出：筛选后的匹配数目
 */
void UsingRansac(const int threshold_value,
                 const cv::Mat &feature1,
                 const cv::Mat &feature2,
                 const vector<cv::KeyPoint> &mvKeys1,
                 const vector<cv::KeyPoint> &mvKeys2,
                 const cv::Mat &mDes1,
                 const cv::Mat &mDes2,
                 const vector<DMatch> &control_matches)
{
    vector<Vertex<float > > points1,points2;
    vector<cv::KeyPoint> mvKeys1_(mvKeys1), mvKeys2_(mvKeys2);

//    cout << "size of mvKeys: " << mvKeys1.size() << " , " << mvKeys2.size() << endl;
//    cout << "size of mvKeys_: " << mvKeys1_.size() << " , " << mvKeys2_.size() << endl;


    vector<DMatch> temp_matches(control_matches);

    sort(temp_matches.begin(), temp_matches.end(), cmpQueryIdxDown);   // query ID 降序
    for (const auto &p:temp_matches)                    // 剔除初始匹配点队
        mvKeys1_.erase(mvKeys1_.begin()+p.queryIdx);

    sort(temp_matches.begin(), temp_matches.end(), cmpTrainIdxDown);   // train ID 降序
    for (const auto &p:temp_matches)                    // 剔除初始匹配点队
        mvKeys2_.erase(mvKeys2_.begin()+p.trainIdx);





















//    for (const auto &p:mvKeys1_)
//    {
//        cout << p.class_id << endl;
//    }

//    sort(control_matches.begin(), control_matches.end(), cmpQueryIdxUp);   //排序
//    cout << endl;
//    for (auto &p:control_matches)
//    {
//        mvKeys1_.erase(mvKeys1_.begin()+p.queryIdx-1);
//        mvKeys2_.erase(mvKeys2_.begin()+p.trainIdx-1);
////        cout << "index: " << p.queryIdx << "\t,\t" << p.trainIdx << "\t,\t" << p.distance << endl;
//    }

//    for (auto &p:mvKeys1_)
//    {
////        cout << p.class_id << endl;     // 原始的特征点
//        for (auto &q:control_matches)
//        {
////            if (p.class_id == q.queryIdx)
////                cout << "ERROR" << endl;
//        }
//    }

//    cout << "\n\nRANSAC:" << endl;
//    for (auto &q:control_matches)
//    {
////        cout << q.queryIdx << endl;
////            if (p.class_id == q.queryIdx)
////                cout << "ERROR" << endl;
//    }

    cout << "size of control_matches: " << control_matches.size() << endl;
    cout << "size of mvKeys_: " << mvKeys1_.size() << " , " << mvKeys2_.size() << endl;

    Mat Debug_one = feature1.clone();
    Mat Debug_two = feature2.clone();

    /***************  RANSAC 实验对照组  ******************************/
    // 保存匹配对序号
    vector<int> queryIdxs( control_matches.size() ), trainIdxs( control_matches.size() );
    for( size_t i = 0; i < control_matches.size(); i++ )
    {
        queryIdxs[i] = control_matches[i].queryIdx;
        trainIdxs[i] = control_matches[i].trainIdx;
    }

    vector<Point2f> CGpoints1; KeyPoint::convert(mvKeys1, CGpoints1, queryIdxs);
    vector<Point2f> CGpoints2; KeyPoint::convert(mvKeys2, CGpoints2, trainIdxs);
    int ransacReprojThreshold = 5;  //拒绝阈值 35

//    cout << "Debug: " << CGpoints1.size() << endl;
    cout << endl << endl;

    // 计算单应矩阵H homography matrix
    Mat homography_matrix = findHomography( Mat(CGpoints1), Mat(CGpoints2), CV_RANSAC, ransacReprojThreshold );
    vector<char> matchesMask( control_matches.size(), 0 );
    Mat points1t;
    perspectiveTransform(Mat(CGpoints1), points1t, homography_matrix);  // 透视变换处理
    int count = 0;
    for( size_t i1 = 0; i1 < CGpoints1.size(); i1++ )  //保存‘内点’
    {
        if( norm(CGpoints2[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记
        {
            matchesMask[i1] = 1;    // 做标记,在drawmatch时会当做mask作为参数输入
            // 取出内点,用于构造DT网络
            points1.emplace_back(Vertex<float>(mvKeys1[control_matches[i1].queryIdx].pt.x , mvKeys1[control_matches[i1].queryIdx].pt.y , control_matches[i1].queryIdx ));
            points2.emplace_back(Vertex<float>(mvKeys2[control_matches[i1].trainIdx].pt.x , mvKeys2[control_matches[i1].trainIdx].pt.y , control_matches[i1].trainIdx ));
            count++;
        }
        else    // 保存外点
        {
//            mvKeys1_.emplace_back(mvKeys1[control_matches[i1].queryIdx]);
//            mvKeys2_.emplace_back(mvKeys2[control_matches[i1].trainIdx]);
//            cout << "index: " << control_matches[i1].queryIdx << "\t,\t" << control_matches[i1].trainIdx << "\t,\t" << control_matches[i1].distance << endl;
        }
    }
    cout << "初始结果: " << count << endl;  // 显示内点数目

    /********************  计算基础矩阵E,得到R,t    ****************************/
    // 计算基础矩阵E Essential
    Mat essential_matrix = findEssentialMat(Mat(CGpoints1), Mat(CGpoints2));
    Mat mat_R,mat_t;
    recoverPose(essential_matrix, Mat(CGpoints1), Mat(CGpoints2), mat_R, mat_t);
    //    cout << "\nessential matrix:\n" << essential_matrix << endl;

    Eigen::Matrix3d R;
    for (int i=0; i<mat_R.rows; ++i) {
        for (int j=0; j<mat_R.cols; ++j) {
            R(i,j) = mat_R.at<double>(i,j);
        }
    }

    Eigen::Vector3d t;
    for (int i=0; i<mat_t.rows; ++i) {
        t(i,0) = mat_t.at<double>(i,0);
    }

    //    cout << "\nR:\n" << R << endl;
    //    cout << "\nt:\n" << t << endl;

    /*********************  使用R,t作为先验,引导特征匹配  ***************************/
    /*

    // 获取尚未匹配的特征点集合     已实现:在内外点判别时,已经进行保存 mvKeys1_ mvKeys2_
//    cout << "size of mvKey1_: " << mvKeys1_.size() << endl;
//    cout << "size of mvKey2_: " << mvKeys2_.size() << endl;

    // 实现BFmatcher
    vector<DMatch> new_matches;
    for (size_t i1 = 0; i1 < mvKeys1_.size(); ++i1)     // 遍历所有未匹配的特征点mvKeys1_
    {
//        Mat Debug_one = feature1.clone();
//        Mat Debug_two = feature2.clone();
        Mat d1 = mDes1.row(mvKeys1_[i1].class_id);

        Eigen::Vector3d p1(mvKeys1_[i1].pt.x,mvKeys1_[i1].pt.y,1);
//        circle(Debug_one, cv::Point(p1(0),p1(1)), 30, Scalar(255,0,255));
        Eigen::Vector3d p2 = R*p1 + t;
//        circle(Debug_two, cv::Point(p2(0),p2(1)), 30, Scalar(255,0,255));

        float radius(0), dx, dy;
        int bestDist = INT_MAX, bestDist2 = INT_MAX, bestIdx2 = -1;
        for (size_t i2 = 0; i2 < mvKeys2_.size(); ++i2)
        {
            dx = p2(0) - mvKeys2_[i2].pt.x;
            dy = p2(1) - mvKeys2_[i2].pt.y;
            radius = dx*dx + dy*dy;         // 计算距离
                                            // TODO: 剔除重复点对

//            Mat Debug_one = feature1.clone();
//            Mat Debug_two = feature2.clone();
//            circle(Debug_one, cv::Point(p1(0),p1(1)), 30, Scalar(255,0,255));
//            circle(Debug_two, cv::Point(p2(0),p2(1)), 30, Scalar(255,0,255));

            if (radius <= 30*30)
            {
                Mat d2 = mDes2.row(mvKeys2_[i2].class_id);  // 提取特征点对应的描述子
                int dist = DescriptorDistance(d1,d2);       // 计算两个描述子之间的汉明距离

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = i2;
                }
                else if (dist < bestDist2)
                    bestDist2 = dist;

                //                arrowedLine(Debug_two, Point(mvKeys2_[i2].pt.x, mvKeys2_[i2].pt.y), Point(mvKeys2_[i2].pt.x, mvKeys2_[i2].pt.y-30), Scalar(0, 0, 255), 1, 8);
            }
        }

        if (bestDist <= 50)
        {
            if (bestDist < (float)bestDist2*0.6)
            {
                new_matches.emplace_back(mvKeys1_[i1].class_id, mvKeys2_[bestIdx2].class_id, bestDist);
            }
        }
//        if (dist <= 30)
//            new_matches.emplace_back(mvKeys1_[i1].class_id, mvKeys2_[i2].class_id, dist);



//        cout << endl;

//        Mat afterOpt;   //滤除‘外点’后
//        drawMatches(Debug_one,mvKeys1,Debug_two,mvKeys2,control_matches,afterOpt,Scalar(0,255,0),Scalar::all(-1),matchesMask);
//        imshow("Debug",afterOpt);
//        waitKey(0);

    }

//    Eigen::Vector3d p1(387,139,1);
//    circle(feature1, cv::Point(p1(0),p1(1)), 10, Scalar(255,0,255));
//    Eigen::Vector3d p2 = R*p1 + t;
//    circle(feature2, cv::Point(p2(0),p2(1)), 10, Scalar(255,0,255));
*/
    /****************  构建DT网络  ************************/
    ///delaunay one
//    Delaunay<float> triangulation1;
//    const std::vector<Triangle<float> > triangles1 = triangulation1.Triangulate(points1);  //逐点插入法
//    triangulation1.ComputeEdgeMatrix();
//    const std::vector<Edge<float> > edges1 = triangulation1.GetEdges();
//    for(const auto &e : edges1)
//    {
//        line(feature1, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
//    }

    ///delaunay two
//    Delaunay<float> triangulation2;
//    const std::vector<Triangle<float> > triangles2 = triangulation2.Triangulate(points2);  //逐点插入法
//    triangulation2.ComputeEdgeMatrix();
//    const std::vector<Edge<float> > edges2 = triangulation2.GetEdges();
//    for(const auto &e : edges2)
//    {
//        line(feature2, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
//    }

    /*******************  显示匹配结果  **********************/
    Mat afterOpt;   //滤除‘外点’后
    drawMatches(feature1,mvKeys1,feature2,mvKeys2,control_matches,afterOpt,Scalar(0,255,0),Scalar::all(-1),matchesMask);
    imshow("control group",afterOpt);
    imwrite("./figure/RANSAC.png",afterOpt);
    waitKey(0);

//    cout << "增加结果: " << new_matches.size() << endl;  // 显示内点数目
//    Mat newOpt;   //滤除‘外点’后
//    drawMatches(Debug_one,mvKeys1,Debug_two,mvKeys2,new_matches,newOpt,Scalar(0,255,0));
//    imshow("newOpt",newOpt);
//    imwrite("./figure/add.png",newOpt);
//    waitKey(0);
}
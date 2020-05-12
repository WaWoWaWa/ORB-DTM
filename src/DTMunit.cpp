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

//    return initGood_matches;
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
 *
 * @details
 * 0. 在函数前,使用暴力匹配得到初始解(初始的匹配点对),control_matches
 * 1. 保存未匹配的两组特征点集合
 * 2. 对control_matches进行RANSAC误匹配剔除,得到优化后的内点CGpoints1,CGpoints2
 * 3. 通过两组内点,计算E矩阵,并恢复R,t
 * 4. 对未匹配的内点再次进行暴力匹配,但是借助E矩阵(或H矩阵),得到圆形ROI区域,在ROI内进行特征匹配,得到汉明距离最小的解
 * 5. 最终匹配的点对为:RANSAC内点+ROI内恢复的点对
 *
 * @bug 圆形ROI内出现多个汉明距离相等的解,如何筛选,使用DT网络进行约束?
 * @test 还需要使用真值进行验证,评估匹配的准确率
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
    Eigen::Quaterniond q1(-0.2335, 0.6743, 0.6458, -0.2716), q2(-0.2344, 0.6734, 0.6456, -0.2734);
    q1.normalized();
    q2.normalized();
    Eigen::Vector3d t1(1.3211, 0.5853, 1.4421), t2(1.3185, 0.5850, 1.4430);

    Eigen::Isometry3d T1w(q1), T2w(q2);
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);
    Eigen::Isometry3d T21 = T2w * T1w.inverse();
    cout << endl;

    cout << "显示初始匹配与投影真值的距离: " << control_matches.size() << endl;
    vector<int> counts{0,0,0};  // <=5   5<=10   >10
    for(int index = 0; index < control_matches.size(); ++index) {
//    int index = 18;
        Eigen::Vector2d p1_debug(mvKeys1[control_matches[index].queryIdx].pt.x,
                                 mvKeys1[control_matches[index].queryIdx].pt.y);
        Eigen::Vector2d p2_debug(mvKeys2[control_matches[index].trainIdx].pt.x,
                                 mvKeys2[control_matches[index].trainIdx].pt.y);
        Eigen::Vector2d p2_estimate;

        Eigen::Vector3d pc1(p1_debug.x(), p1_debug.y(), 1);
        Eigen::Vector3d pc2 = T21 * pc1;

        float distance = sqrt(pow(p2_debug.x() - pc2.x(), 2) + pow(p2_debug.y() - pc2.y(), 2));
//        cout << "距离: " << distance << endl;

        if(distance <= 5)
            counts[0]++;
        else if(distance <=10)
            counts[1]++;
        else
            counts[2]++;

//        if(distance > 0.001)
//            continue;
//        else {
//            cout << "距离: " << distance << endl;

//              cout << "坐标: " << pc1(0) << " , " << pc1(1) << endl;

//            Mat temp_1 = feature1.clone();
//            arrowedLine(temp_1, cv::Point(pc1(0), pc1(1)), cv::Point(pc1(0), pc1(1) - 30), Scalar(0, 0, 255), 1, 8);        // 原始特征点
//            arrowedLine(temp_1, cv::Point(pc1(0), pc1(1)), cv::Point(pc1(0) - 30, pc1(1)), Scalar(0, 0, 255), 1, 8);
//              circle(temp_1, cv::Point(p1_debug(0),p1_debug(1)), 30, Scalar(255,0,255));     // 原图像
//            imshow("temp1", temp_1);
//            waitKey(0);

//            Mat temp_2 = feature2.clone();
//            arrowedLine(temp_2, cv::Point(pc2(0), pc2(1)), cv::Point(pc2(0), pc2(1) - 30), Scalar(0, 0, 255), 1, 8);                    // 投影点
//            arrowedLine(temp_2, cv::Point(pc2(0), pc2(1)), cv::Point(pc2(0) - 30, pc2(1)), Scalar(0, 0, 255), 1, 8);
//            arrowedLine(temp_2, cv::Point(p2_debug(0), p2_debug(1)), cv::Point(p2_debug(0), p2_debug(1) - 30), Scalar(0, 255, 0), 1, 8);     // 匹配点
//            arrowedLine(temp_2, cv::Point(p2_debug(0), p2_debug(1)), cv::Point(p2_debug(0) - 30, p2_debug(1)), Scalar(0, 255, 0), 1, 8);
//            imshow("temp2", temp_2);
//            waitKey(0);
//        }
    }
    cout << "初始的误匹配点数目(... 5 ... 10 ...): " << counts[0] << " , " << counts[1] << " , " << counts[2] << endl;
    /***************  获取初始匹配之外的点对   **************************/
    vector<Vertex<float > > points1,points2;
    vector<cv::KeyPoint> mvKeys1_(mvKeys1), mvKeys2_(mvKeys2);

    Mat Debug_one = feature1.clone();       // 克隆,用于增加额外的匹配点对(借助相机外参)
    Mat Debug_two = feature2.clone();

    vector<DMatch> temp_matches(control_matches);   // 用于排序的中间变量

    sort(temp_matches.begin(), temp_matches.end(), cmpQueryIdxDown);   // query ID 降序
    for (const auto &p:temp_matches)                    // 剔除初始匹配点队
        mvKeys1_.erase(mvKeys1_.begin()+p.queryIdx);

    sort(temp_matches.begin(), temp_matches.end(), cmpTrainIdxDown);   // train ID 降序
    for (const auto &p:temp_matches)                    // 剔除初始匹配点队
        mvKeys2_.erase(mvKeys2_.begin()+p.trainIdx);
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

    // 计算单应矩阵H homography matrix
    Mat homography_matrix = findHomography( Mat(CGpoints1), Mat(CGpoints2), CV_RANSAC, ransacReprojThreshold );
    //    cout << "\nhomography_matrix: \n" << homography_matrix << endl;
    Eigen::Matrix3d H12;
    H12 <<  homography_matrix.at<double>(0,0), homography_matrix.at<double>(0,1), homography_matrix.at<double>(0,2),
            homography_matrix.at<double>(1,0), homography_matrix.at<double>(1,1), homography_matrix.at<double>(1,2),
            homography_matrix.at<double>(2,0), homography_matrix.at<double>(2,1), homography_matrix.at<double>(2,2);

    Mat homography2_matrix = findHomography( Mat(CGpoints2), Mat(CGpoints1), CV_RANSAC, ransacReprojThreshold );
    //    cout << "\nhomography2_matrix: \n" << homography2_matrix << endl;
    Eigen::Matrix3d H21;
    H21 <<  homography2_matrix.at<double>(0,0), homography2_matrix.at<double>(0,1), homography2_matrix.at<double>(0,2),
            homography2_matrix.at<double>(1,0), homography2_matrix.at<double>(1,1), homography2_matrix.at<double>(1,2),
            homography2_matrix.at<double>(2,0), homography2_matrix.at<double>(2,1), homography2_matrix.at<double>(2,2);

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
        //        else    // 保存外点   待定,但是要改进保存方法(索引不正确)
        //        {
        //            mvKeys1_.emplace_back(mvKeys1[control_matches[i1].queryIdx]);
        //            mvKeys2_.emplace_back(mvKeys2[control_matches[i1].trainIdx]);
        //            cout << "index: " << control_matches[i1].queryIdx << "\t,\t" << control_matches[i1].trainIdx << "\t,\t" << control_matches[i1].distance << endl;
        //        }
    }
    cout << "\n初始结果: " << count << endl;  // 显示内点数目

    /********************  计算基础矩阵E,得到R,t    ****************************/
    // 计算基础矩阵E Essential
    Mat essential_matrix = findEssentialMat(Mat(CGpoints1), Mat(CGpoints2));
    Mat mat_R,mat_t;
    recoverPose(essential_matrix, Mat(CGpoints1), Mat(CGpoints2), mat_R, mat_t);

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
    // 获取尚未匹配的特征点集合     已实现:在内外点判别时,已经进行保存 mvKeys1_ mvKeys2_

    // 实现BFmatcher
    vector<DMatch> new_matches;
    for (size_t i1 = 0; i1 < mvKeys1_.size(); ++i1)     // 遍历所有未匹配的特征点mvKeys1_
    {
        Mat d1 = mDes1.row(mvKeys1_[i1].class_id);

        Eigen::Vector3d p1(mvKeys1_[i1].pt.x,mvKeys1_[i1].pt.y,1);
//        circle(Debug_one, cv::Point(p1(0),p1(1)), 30, Scalar(255,0,255));     // 可视化
//         Eigen::Vector3d p2 = R*p1 + t;       // 使用基础矩阵E计算坐标
        Eigen::Vector3d p2 = H12 * p1;          // 使用单应矩阵H计算坐标
//        circle(Debug_two, cv::Point(p2(0),p2(1)), 30, Scalar(255,0,255));     // 可视化
        // 是否应该采取两种矩阵均计算的方式,选取误差更小的结果???

        float radius(0), dx, dy;
        int bestDist = INT_MAX, bestDist2 = INT_MAX, bestIdx2 = -1;
        int ROIradius = 30;
        for (size_t i2 = 0; i2 < mvKeys2_.size(); ++i2)
        {
            dx = p2(0) - mvKeys2_[i2].pt.x;
            dy = p2(1) - mvKeys2_[i2].pt.y;
            radius = dx*dx + dy*dy;         // 计算距离
            // TO DO: 剔除重复点对 done
            // todo: 绘制出圆形ROI内的特征点,并显示的匹配值,以及真值的投影点

            // 对圆形ROI内的特征点,进行进一步处理
            if (radius <= ROIradius*ROIradius)
            {
                // Debug: 此处用于每帧图像均进行单独的可视化操作
//                 Mat Debug_one = feature1.clone();       // 克隆,用于增加额外的匹配点对(借助相机外参)
//                 Mat Debug_two = feature2.clone();
//                 circle(Debug_one, cv::Point(p1(0),p1(1)), ROIradius, Scalar(255,0,255));
//                 circle(Debug_two, cv::Point(p2(0),p2(1)), ROIradius, Scalar(255,0,255));

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

//                cout << "汉明距离: " << bestDist << endl;
//                arrowedLine(Debug_two, Point(mvKeys2_[i2].pt.x, mvKeys2_[i2].pt.y), Point(mvKeys2_[i2].pt.x, mvKeys2_[i2].pt.y-30), Scalar(0, 0, 255), 1, 8);
//                arrowedLine(Debug_one, Point(mvKeys1_[i1].pt.x, mvKeys1_[i1].pt.y), Point(mvKeys1_[i1].pt.x, mvKeys1_[i1].pt.y-30), Scalar(0, 0, 255), 1, 8);

//                Mat newOpt;   //滤除‘外点’后
//                drawMatches(Debug_one,mvKeys1,Debug_two,mvKeys2,new_matches,newOpt,Scalar(0,255,0));
//                imshow("Debug",newOpt);
//                waitKey(0);
            }
        }
//        cout << endl;

        if (bestDist <= 75)     /// 60  75  90
        {
            if (bestDist < (float)bestDist2*1.0)    // 不应该再使用比值来限制,因为已经限制在很小的ROI之中,比较个数很少
            {
                new_matches.emplace_back(mvKeys1_[i1].class_id, mvKeys2_[bestIdx2].class_id, bestDist);

                // Debug: 中间环节的可视化
//                vector<DMatch> temp_matches;
//                temp_matches.emplace_back(mvKeys1_[i1].class_id, mvKeys2_[bestIdx2].class_id, bestDist);
//                Mat Debug_one = feature1.clone();       // 克隆,用于增加额外的匹配点对(借助相机外参)
//                Mat Debug_two = feature2.clone();
//                Mat tempOpt;   //滤除‘外点’后
//                drawMatches(Debug_one,mvKeys1,Debug_two,mvKeys2,temp_matches,tempOpt,Scalar(0,255,0));
//                imshow("tempOpt",tempOpt);
//                waitKey(0);

                // 用于构建Delaunay Triangulation网络
                points1.emplace_back(Vertex<float>(mvKeys1[mvKeys1_[i1].class_id].pt.x ,
                                                   mvKeys1[mvKeys1_[i1].class_id].pt.y ,
                                                   mvKeys1_[i1].class_id ));

                points2.emplace_back(Vertex<float>(mvKeys2[mvKeys2_[bestIdx2].class_id].pt.x ,
                                                   mvKeys2[mvKeys2_[bestIdx2].class_id].pt.y ,
                                                   mvKeys2_[bestIdx2].class_id ));
            }
        }

    }

    /// 对新增点对进行正确性验证
    cout << "显示新增匹配与投影真值的距离:" << endl;
    counts = {0,0,0};
    for(int index = 0; index < new_matches.size(); ++index) {
//    int index = 18;
        Eigen::Vector2d p1_debug(mvKeys1[new_matches[index].queryIdx].pt.x,
                                 mvKeys1[new_matches[index].queryIdx].pt.y);
        Eigen::Vector2d p2_debug(mvKeys2[new_matches[index].trainIdx].pt.x,
                                 mvKeys2[new_matches[index].trainIdx].pt.y);
//        Eigen::Vector2d p2_estimate;

        Eigen::Vector3d pc1(p1_debug.x(), p1_debug.y(), 1);
        Eigen::Vector3d pc2 = T21 * pc1;

        float distance = sqrt(pow(p2_debug.x() - pc2.x(), 2) + pow(p2_debug.y() - pc2.y(), 2));
//        cout << "距离: " << distance << endl;

        if(distance <= 5)
            counts[0]++;
        else if(distance <=10)
            counts[1]++;
        else
            counts[2]++;

//        if (distance > 0.001)
//            continue;
//        else {
//            cout << "距离: " << distance << endl;

//    cout << "坐标: " << pc1(0) << " , " << pc1(1) << endl;

//            Mat temp_11 = feature1.clone();
//            arrowedLine(temp_11, cv::Point(pc1(0), pc1(1)), cv::Point(pc1(0), pc1(1) - 30), Scalar(0, 0, 255), 1, 8);
//            arrowedLine(temp_11, cv::Point(pc1(0), pc1(1)), cv::Point(pc1(0) - 30, pc1(1)), Scalar(0, 0, 255), 1, 8);
//    circle(temp_1, cv::Point(p1_debug(0),p1_debug(1)), 30, Scalar(255,0,255));     // 原图像
//            imshow("temp11", temp_11);
//        waitKey(0);

//            Mat temp_22 = feature2.clone();
//            arrowedLine(temp_22, cv::Point(pc2(0), pc2(1)), cv::Point(pc2(0), pc2(1) - 30), Scalar(0, 0, 255), 1, 8);       // 投影点
//            arrowedLine(temp_22, cv::Point(pc2(0), pc2(1)), cv::Point(pc2(0) - 30, pc2(1)), Scalar(0, 0, 255), 1, 8);
//            arrowedLine(temp_22, cv::Point(p2_debug(0), p2_debug(1)), cv::Point(p2_debug(0), p2_debug(1) - 30), Scalar(0, 255, 0), 1, 8);   // 匹配点
//            arrowedLine(temp_22, cv::Point(p2_debug(0), p2_debug(1)), cv::Point(p2_debug(0) - 30, p2_debug(1)), Scalar(0, 255, 0), 1, 8);
//            imshow("temp22", temp_22);
//            waitKey(0);

//            Mat newOpt;   //滤除‘外点’后
//            drawMatches(temp_11,mvKeys1,temp_22,mvKeys2,new_matches,newOpt,Scalar(255,0,0));
//            imshow("newOpt",newOpt);
//            imwrite("./figure/add.png",newOpt);
//            waitKey(0);
//        }
    }

    cout << "新增的误匹配点数目(... 5 ... 10 ...): " << counts[0] << " , " << counts[1] << " , " << counts[2] << endl;

    /****************  构建DT网络  ************************/
    ///delaunay one
//    Delaunay<float> triangulation1;
//    const std::vector<Triangle<float> > triangles1 = triangulation1.Triangulate(points1);  //逐点插入法
//    triangulation1.ComputeEdgeMatrix();
//    const std::vector<Edge<float> > edges1 = triangulation1.GetEdges();
//        for(const auto &e : edges1)
//        {
//            line(feature1, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
//        }

    ///delaunay two
//    Delaunay<float> triangulation2;
//    const std::vector<Triangle<float> > triangles2 = triangulation2.Triangulate(points2);  //逐点插入法
//    triangulation2.ComputeEdgeMatrix();
//    const std::vector<Edge<float> > edges2 = triangulation2.GetEdges();
//        for(const auto &e : edges2)
//        {
//            line(feature2, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
//        }
    /*******************  显示匹配结果  **********************/
//    Mat afterOpt;   //滤除‘外点’后
//    drawMatches(feature1,mvKeys1,feature2,mvKeys2,control_matches,afterOpt,Scalar(0,255,0),Scalar::all(-1),matchesMask);
//    imshow("control group",afterOpt);
//    imwrite("./figure/RANSAC.png",afterOpt);
//    waitKey(0);

    cout << "增加结果: " << new_matches.size() << endl;  // 显示内点数目
//    Mat newOpt;   //滤除‘外点’后
//    drawMatches(Debug_one,mvKeys1,Debug_two,mvKeys2,new_matches,newOpt,Scalar(0,255,0));
//    imshow("newOpt",newOpt);
//    imwrite("./figure/add.png",newOpt);
//    waitKey(0);
}
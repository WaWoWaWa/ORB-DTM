//
// Created by lu on 2019/12/16.
//
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;

int main()
{
    Eigen::MatrixXd::Index maxRow,maxCol;
    Eigen::MatrixXd edgeMatrix = Eigen::MatrixXd::Zero(3,3);
    edgeMatrix(1,2) = 5;
    edgeMatrix(0,1) = 3;
    edgeMatrix(2,1) = 4;
    edgeMatrix(2,2) = 2;
    cout << edgeMatrix << endl;

    edgeMatrix.row(2).maxCoeff(&maxRow,&maxCol);

    cout << "max locate: " << "( " << maxRow << " , " <<  maxCol << " )" << endl;
    cout << "max value: " << edgeMatrix(maxRow, maxCol) << endl;

    edgeMatrix.cwiseAbs().rowwise().sum().maxCoeff(&maxRow,&maxCol);
    cout << "max row: "  << maxRow << endl;
//        edgeMatrix.cwiseAbs().colwise().sum().maxCoeff(&maxRow,&maxCol);    // 边矩阵.绝对值.列.和.最大值(行序号?,列序号)

    cout << "\nmain() end, see you..." << endl;
    return 0;
}
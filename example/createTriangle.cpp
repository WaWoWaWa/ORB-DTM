//
// Created by lu on 2019/12/13.
//

#include <iostream>

#include "Vertex.h"
#include "edge.h"
#include "triangle.h"

using namespace std;

int main()
{
    Vertex<float> v1(30, 40);
    Vertex<float> v2(10, 50);
    Vertex<float> v3(10, 40);

    Triangle<float> t(v1, v2, v3);
}
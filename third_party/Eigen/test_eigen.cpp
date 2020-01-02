// This file is used to practice calculation using eigen library
//
//
//
//
//
//
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void test_conversion(){
    MatrixXf m(2,2);
    MatrixXf n(2,2);
    m<<1,2,
        3,4;
    n<<5,6,
        7,8;

    MatrixXf result = m * n;
    cout<< "Matrix m * n"<<endl<<result<<endl<<endl;
    result = m.array()*n.array();
    cout<<"-- Array m * n: "<<endl<<result<< endl<<endl;
    result = m.cwiseProduct(n);
    cout<<"-- With cwiseProduct: --"<<endl<<result <<endl<<endl;
    result  = m.array() + 4;
    cout<<"-- Array m+4: --"<<endl<<result<<endl<<endl;
}

void test_block(){
    MatrixXf m(4,4);
    m<<1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16;
    cout<<"Block in the middle" <<endl;
    // static
    cout<<m.block<2,2>(1,1)<<endl<<endl;
    //dynamic
    for(int i=1;i<=3;i++){
        cout<<"Block of size "<<endl;
        cout<<m.block(0,0,i,i)<<endl;
    }

    // col and row operator
    m.col(2) +=3*m.col(3);
    cout<<m<<endl;
}

void test_initializer(){
    // join with vector
    RowVectorXd vec1(3);
    vec1<<1,2,3;
    RowVectorXd vec2(4);
    vec2<<1,2,3,4;
    RowVectorXd joined(7);
    joined<<vec1, vec2;

    // join with matrix
    Matrix2d matA;
    matA<<1,2,3,4;
    Matrix4d matB;
    matB<<matA,matA/10,matA/10, matA;
    cout<<"matB: "<<matB<<endl;

    // fill a matrix
    Matrix3d m;
    m.row(0)<<1,2,3;
    m.block(1, 0, 2,2)<<4,5,7,8;
    m.col(2).tail(2)<<6,9;
    cout<<"m: "<<m<<endl;
}


int main(){
    // Matrix
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1,0) = 2.5;
    m(0, 1) = -1;
    m(1,1) = (1,0)+m(0,1);
    std::cout<<m<<std::endl;

    Matrix2d m2x2;
    // type, rows, cols
    Matrix<double, 2, 2> m2x2_;
    Vector4f v4;
    Matrix<float, 4, 1> v4_;
    RowVector3i v3;
    Matrix<int, 1, 3> v3_;

    // dynamic
    MatrixXf m_1(10, 10);
    VectorXf v(10);
    Matrix<float, Dynamic, Dynamic> m_2(10, 10);
    Matrix<float, Dynamic, 1> v_(10);

    // array multiple
    ArrayXXf a(2,2);
    ArrayXXf b(2,2);
    a<<1,2,
        3,4;
    b<<5,6,
        7,8;

    std::cout<<"a*b: "<<std::endl<<a*b<<std::endl;
    ArrayXf c= ArrayXf::Random(5);
    a*=2;
    std::cout<<"c = "<<std::endl
        <<c<<std::endl;
    std::cout<<"c.abs() = "<<std::endl
        <<c.abs()<<std::endl;
    std::cout<<"c.abs().sqrt() = "<<std::endl
        <<c.abs().sqrt()<<std::endl;

    // array and matrix
    test_conversion();

    test_block();

    test_initializer();

    return 0;
}

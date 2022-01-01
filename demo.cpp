#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Dense"
#include "Eigen/Core"
#include "PINV.h"
#include <iostream>

PINV* p;

void assert_true(bool cond, const char* const msg) {
    if(!cond) {
        std::cout << msg << std::endl;
    }
}

void test_simple() {

    Eigen::internal::set_is_malloc_allowed(true);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
    input << 3, 3.2, 3.5, 3.6;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(), input.rows());
    Eigen::internal::set_is_malloc_allowed(false);

    p->bd.input.setZero();
    p->bd.input.block(0, 0, 2, 2) = input;

    p->calculate_PINV();

    output = p->bd.output.block(0, 0, output.rows(), output.cols());

    Eigen::internal::set_is_malloc_allowed(true);
    assert_true(((input * output - Eigen::MatrixXd::Identity(input.rows(), output.cols())).array().abs() < TOL).all(), "Bad pseudoinverse");
    Eigen::internal::set_is_malloc_allowed(false);
}

 int main() {
    Eigen::internal::set_is_malloc_allowed(true);
    p = new PINV();
    Eigen::internal::set_is_malloc_allowed(false);

    std::cout << "test_simple" << std::endl;
    test_simple();
}
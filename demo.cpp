#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Core"
#include "Eigen/Dense"
#include "PINV.h"
#include <iostream>

// Low to allow for manual number inputs to be considered equal
#define TOL 1e-6

PINV *p;

void assert_true(bool cond, const char *const msg) {
  if (!cond) {
    std::cout << "FAIL: " << msg << std::endl;
  } else {
    std::cout << "PASS" << std::endl;
  }
}

void test_simple() {

  Eigen::internal::set_is_malloc_allowed(true);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
  input << 3, 3.2, 3.5, 3.6;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::internal::set_is_malloc_allowed(false);

  p->reset();
  p->bd.input.block(0, 0, input.rows(), input.cols()) = input;

  p->calculate();

  output = p->bd.output.block(0, 0, output.rows(), output.cols());

  Eigen::internal::set_is_malloc_allowed(true);
  // For square matrices, the pseudoinverse is the same as the inverse
  assert_true(
      ((input * output - Eigen::MatrixXd::Identity(input.rows(), output.cols()))
           .array()
           .abs() < TOL)
          .all(),
      "Bad pseudoinverse");
  Eigen::internal::set_is_malloc_allowed(false);
}

void test1() {

  Eigen::internal::set_is_malloc_allowed(true);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 2);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::internal::set_is_malloc_allowed(false);

  input << 7, 2, 3, 4, 5, 3;
  expected << 0.166667, -0.106061, 0.030303, -0.166667, 0.287879, 0.0606061;

  p->reset();
  p->bd.input.block(0, 0, input.rows(), input.cols()) = input;

  p->calculate();

  output = p->bd.output.block(0, 0, output.rows(), output.cols());

  Eigen::internal::set_is_malloc_allowed(true);
  assert_true((output - expected).array().abs().maxCoeff() < TOL,
              "Bad psuedoinverse");
  Eigen::internal::set_is_malloc_allowed(false);
}

void test2() {

  Eigen::internal::set_is_malloc_allowed(true);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 2);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::internal::set_is_malloc_allowed(false);

  input << -2, -1, 4, -1, -1, -1;
  expected << -0.112903, 0.177419, -0.0645161, -0.370968, -0.274194, -0.354839;

  p->reset();
  p->bd.input.block(0, 0, input.rows(), input.cols()) = input;

  p->calculate();

  output = p->bd.output.block(0, 0, output.rows(), output.cols());

  Eigen::internal::set_is_malloc_allowed(true);
  assert_true((output - expected).array().abs().maxCoeff() < TOL,
              "Bad psuedoinverse");
  Eigen::internal::set_is_malloc_allowed(false);
}

void test3() {

  Eigen::internal::set_is_malloc_allowed(true);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(6, 2);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(),
                                                                  input.rows());
  Eigen::internal::set_is_malloc_allowed(false);

  input << 0, 1, 1, 1, 2, 1, 3, 1, 3, 1, 4, 1;
  expected << -2.00000000e-01, -1.07692308e-01, -1.53846154e-02, 7.69230769e-02,
      7.69230769e-02, 1.69230769e-01, 6.00000000e-01, 4.00000000e-01,
      2.00000000e-01, 4.16333634e-17, 4.16333634e-17, -2.00000000e-01;

  p->reset();
  p->bd.input.block(0, 0, input.rows(), input.cols()) = input;

  p->calculate();

  output = p->bd.output.block(0, 0, output.rows(), output.cols());

  Eigen::internal::set_is_malloc_allowed(true);
  assert_true((output - expected).array().abs().maxCoeff() < TOL,
              "Bad psuedoinverse");
  Eigen::internal::set_is_malloc_allowed(false);
}

int main() {

  // Only time PINV will use the heap
  Eigen::internal::set_is_malloc_allowed(true);
  p = new PINV();
  Eigen::internal::set_is_malloc_allowed(false);

  std::cout << "test_simple" << std::endl;
  test_simple();
  std::cout << "test1" << std::endl;
  test1();
  std::cout << "test2" << std::endl;
  test2();
  std::cout << "test3" << std::endl;
  test3();
}
#include "PINV.h"
#include <iostream>

PINV::PINV() { _init_bdcsvd_mem(); }

void PINV::_init_bdcsvd_mem() {
  // for BDCSVD
  bd.input =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(ROWS, COLS);
  bd.output =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(COLS, ROWS);
  bd.copy =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::ColMajor>::Zero(bd.input.rows(), bd.input.cols());
  bd.bdg = new Eigen::internal::BandMatrix<double, -1, -1, 1, 0, 1>(COLS, COLS);
  bd.hh =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::ColMajor>::Zero(bd.input.rows(), bd.input.cols());
  bd.col_temp =
      Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(bd.input.rows(), 1);
  bd.ess_temp = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::ColMajor>::Zero(bd.input.rows(), 1);
  bd.bid_dense =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>::Zero(bd.input.cols(), bd.input.cols());
  bd.jacobi_temp_member = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::ColMajor>::Zero(8, 1);

  // Stuff for the divide function, n=17
  bd.jacobi_temp = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor>::Zero(9, 8);
  bd.UofSVD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::ColMajor>::Zero(18, 18);
  bd.VofSVD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::ColMajor>::Zero(17, 17);
  bd.singVals = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(17, 1);
  bd.workspace = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::ColMajor>::Zero(1, 17);
  bd.q0 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>::Zero(9, 1);
  bd.sigma_inv =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(17, 17);
  bd.sigma_svd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(17, 17);
  bd.b = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(17, 17);

  // objects themselves
  bd.svd =
      new Eigen::BDCSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(
          bd.input.rows(), bd.input.cols(),
          Eigen::ComputeThinU | Eigen::ComputeThinV);
  bd.jacobi = new Eigen::JacobiSVD<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
      9, 8, &bd.jacobi_temp_member, Eigen::ComputeFullU | Eigen::ComputeFullV);
}

void PINV::calculate() {
  // bidiagonal stuff...
  Eigen::internal::UpperBidiagonalization<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      bid(bd.copy, &bd.hh, bd.bdg, &bd.col_temp, &bd.ess_temp);

  bd.svd->compute(bd.input, &bd.copy, &bid, &bd.bid_dense, bd.jacobi,
                  &bd.jacobi_temp, &bd.singVals, &bd.UofSVD, &bd.VofSVD,
                  &bd.workspace, &bd.q0, &bd.ess_temp);

  Eigen::Matrix<double, COLS, 1> sigma = bd.svd->singularValues();
  double trunc = sigma(0, 0) * RCOND;
  Eigen::Matrix<bool, COLS, 1> sig_mask = sigma.array() > trunc;

  /*
   * Select says compute the inverse of Sigma, except where mask is false,
   * and for those indices use zero.
   */
  bd.sigma_inv =
      sig_mask.select(sigma.array().inverse(), 0).matrix().asDiagonal();
  bd.sigma_svd.noalias() = bd.svd->matrixV();
  bd.b.noalias() = bd.sigma_svd * bd.sigma_inv;
  bd.output.noalias() = bd.b * bd.svd->matrixU().adjoint();
}

void PINV::reset() {
    bd.input.setZero();
}
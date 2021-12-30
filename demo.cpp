#define EIGEN_DEFAULT_ROW_MAJOR
#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Dense"
#include "Eigen/Core"
#include <iostream>

#define x 17
#define y 2500

struct BDCSVD_Mem {
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > input;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > output;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        copy;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        hh;
    Eigen::Matrix< double, Eigen::Dynamic, 1 > col_temp;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        ess_temp;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >
        bid_dense;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        jacobi_temp_member;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        jacobi_temp;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        UofSVD;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        VofSVD;
    Eigen::Matrix< double, Eigen::Dynamic, 1 > singVals;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        workspace;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >
        q1;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > sigma_inv;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > sigma_svd;
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > b;
    Eigen::internal::BandMatrix< double, -1, -1, 1, 0, 1 >* bdg;

    //Objects themselves, pre-allocations at boot
    Eigen::BDCSVD< Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > >*
        svd;
    Eigen::JacobiSVD< Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor > >* jacobi;
};

BDCSVD_Mem bd;

void init_bdcsvd_mem() {
    //for BDCSVD
    bd.input = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >::Zero(
        y, x);
    bd.output = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >::Zero(
        x, y);
    bd.copy = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor >::Zero(bd.input.rows(),
                                                     bd.input.cols());
    bd.bdg = new Eigen::internal::BandMatrix< double, -1, -1, 1, 0, 1 >(
        x, x);
    bd.hh = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::ColMajor >::Zero(bd.input.rows(),
                                                   bd.input.cols());
    bd.col_temp =
        Eigen::Matrix< double, Eigen::Dynamic, 1 >::Zero(bd.input.rows(), 1);
    bd.ess_temp = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor >::Zero(bd.input.rows(), 1);
    bd.bid_dense = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor >::Zero(bd.input.cols(),
                                                          bd.input.cols());
    bd.jacobi_temp_member =
        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::ColMajor >::Zero(8, 1);

    //Stuff for the divide function, n=17
    bd.jacobi_temp = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::ColMajor >::Zero(9, 8);
    bd.UofSVD = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::ColMajor >::Zero(18, 18);
    bd.VofSVD = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::ColMajor >::Zero(17, 17);
    bd.singVals = Eigen::Matrix< double, Eigen::Dynamic, 1 >::Zero(17, 1);
    bd.workspace = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::ColMajor >::Zero(1, 17);
    bd.q1 = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::ColMajor >::Zero(9, 1);
    bd.sigma_inv =
        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >::Zero(17, 17);
    bd.sigma_svd =
        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >::Zero(17, 17);
    bd.b =
        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >::Zero(17, 17);

    //objects themselves
    bd.svd = new Eigen::BDCSVD<
        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > >(
        bd.input.rows(), bd.input.cols(),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    bd.jacobi = new Eigen::JacobiSVD< Eigen::Matrix<
        double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >(
        9, 8, &bd.jacobi_temp_member,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
}

/*
 * Pseudo-inverse via BDCSVD computation. Malloc-free. Caller is responsible
 * for setting input and getting output matrices via Memory.
 *
 * Python equivalent:
 * mem.bd.output = np.linalg.pinv(mem.bd.input)
 *
 * ----------
 * USAGE
 * ----------
 * //Check sizes are within buffer sizes
 * ASSERT(input_matrix.rows() <= mem.bd.input.rows());
 * ASSERT(input_matrix.cols() <= mem.bd.input.cols());
 * ASSERT(output_matrix.rows() <= mem.bd.output.rows());
 * ASSERT(output_matrix.cols() <= mem.bd.output.cols());
 *
 * //Set up input
 * mem.bd.input.setZero(); //To allow smaller problem sizes to be computed accurately
 * mem.bd.input.block(0, 0, input_matrix.rows(), input_matrix.cols()) =
 *  input_matrix;
 *
 * int32 ret_val = PINV();
 *
 * //Copy out output
 * output_matrix =
 *  mem.bd.output.block(0, 0, output_matrix.rows(), output_matrix.cols());
 *
 * ----------
 * NOTE
 * ----------
 * mem.bd.output and mem.bd.input are the only in and out buffers for PINV. This
 * allows the functions calling them to not need maximum output buffers every time.
 * The maximum input problem size is currently 2500x17; this can be adjusted inside
 * memory.cpp, where mem.bd is initialized. Any smaller problem size is also
 * allowed, given that the input is zeroed out otherwise.
 */

 int main() {
     bd.input.setZero();
 }
int32 PINV() {
    //bidiagonal stuff...
    Eigen::internal::UpperBidiagonalization< Eigen::Matrix<
        double64, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >
        bid(mem.bd.copy, &mem.bd.hh, mem.bd.bdg, &mem.bd.col_temp,
            &mem.bd.ess_temp);

    mem.bd.svd->compute(mem.bd.input, &mem.bd.copy, &bid, &mem.bd.bid_dense,
                        mem.bd.jacobi, &mem.bd.jacobi_temp, &mem.bd.singVals,
                        &mem.bd.UofSVD, &mem.bd.VofSVD, &mem.bd.workspace,
                        &mem.bd.q1, &mem.bd.ess_temp);

    Eigen::Matrix< double64, 17, 1 > sigma = mem.bd.svd->singularValues();
    double64 trunc = sigma(0, 0) * RCOND;
    Eigen::Matrix< bool, 17, 1 > sig_mask = sigma.array() > trunc;

    /*
     * Select says compute the inverse of Sigma, except where mask is falsy,
     * and for those indices use zero.
     */
    mem.bd.sigma_inv =
        sig_mask.select(sigma.array().inverse(), 0).matrix().asDiagonal();
    mem.bd.sigma_svd.noalias() = mem.bd.svd->matrixV();
    mem.bd.b.noalias() = mem.bd.sigma_svd * mem.bd.sigma_inv;
    mem.bd.output.noalias() = mem.bd.b * mem.bd.svd->matrixU().adjoint();

    m_error_code = NO_ERROR;
    return 0;
}

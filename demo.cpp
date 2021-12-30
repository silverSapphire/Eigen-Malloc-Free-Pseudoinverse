#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Dense"
#include "Eigen/Core"
#include <iostream>

#define ROWS 2500
#define COLS 17
#define RCOND 1e-15
#define TOL 1e-13

void assert_true(bool cond, const char* const msg) {
    if(!cond) {
        std::cout << msg << std::endl;
    }
}

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
        ROWS, COLS);
    bd.output = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >::Zero(
        COLS, ROWS);
    bd.copy = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor >::Zero(bd.input.rows(),
                                                     bd.input.cols());
    bd.bdg = new Eigen::internal::BandMatrix< double, -1, -1, 1, 0, 1 >(
        COLS, COLS);
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
 * bd.output = np.linalg.pinv(bd.input)
 *
 * ----------
 * USAGE
 * ----------
 * //Check sizes are within buffer sizes
 * ASSERT(input_matrix.rows() <= bd.input.rows());
 * ASSERT(input_matrix.cols() <= bd.input.cols());
 * ASSERT(output_matrix.rows() <= bd.output.rows());
 * ASSERT(output_matrix.cols() <= bd.output.cols());
 *
 * //Set up input
 * bd.input.setZero(); //To allow smaller problem sizes to be computed accurately
 * bd.input.block(0, 0, input_matrix.rows(), input_matrix.cols()) =
 *  input_matrix;
 *
 * int32 ret_val = PINV();
 *
 * //Copy out output
 * output_matrix =
 *  bd.output.block(0, 0, output_matrix.rows(), output_matrix.cols());
 *
 * ----------
 * NOTE
 * ----------
 * bd.output and bd.input are the only in and out buffers for PINV. This
 * allows the functions calling them to not need maximum output buffers every time.
 * The maximum input problem size is currently 2500x17; this can be adjusted inside
 * memory.cpp, where bd is initialized. Any smaller problem size is also
 * allowed, given that the input is zeroed out otherwise.
 */
void PINV() {
    //bidiagonal stuff...
    Eigen::internal::UpperBidiagonalization< Eigen::Matrix<
        double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >
        bid(bd.copy, &bd.hh, bd.bdg, &bd.col_temp,
            &bd.ess_temp);

    bd.svd->compute(bd.input, &bd.copy, &bid, &bd.bid_dense,
                        bd.jacobi, &bd.jacobi_temp, &bd.singVals,
                        &bd.UofSVD, &bd.VofSVD, &bd.workspace,
                        &bd.q1, &bd.ess_temp);

    Eigen::Matrix< double, COLS, 1 > sigma = bd.svd->singularValues();
    double trunc = sigma(0, 0) * RCOND;
    Eigen::Matrix< bool, COLS, 1 > sig_mask = sigma.array() > trunc;

    /*
     * Select says compute the inverse of Sigma, except where mask is falsy,
     * and for those indices use zero.
     */
    bd.sigma_inv =
        sig_mask.select(sigma.array().inverse(), 0).matrix().asDiagonal();
    bd.sigma_svd.noalias() = bd.svd->matrixV();
    bd.b.noalias() = bd.sigma_svd * bd.sigma_inv;
    bd.output.noalias() = bd.b * bd.svd->matrixU().adjoint();
}

void test_simple() {

    Eigen::internal::set_is_malloc_allowed(true);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
    input << 3, 3.2, 3.5, 3.6;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(), input.rows());
    Eigen::internal::set_is_malloc_allowed(false);

    bd.input.setZero();
    bd.input.block(0, 0, 2, 2) = input;

    PINV();

    output = bd.output.block(0, 0, output.rows(), output.cols());

    Eigen::internal::set_is_malloc_allowed(true);
    assert_true(((input * output - Eigen::MatrixXd::Identity(input.rows(), output.cols())).array().abs() < TOL).all(), "Bad pseudoinverse");
    Eigen::internal::set_is_malloc_allowed(false);
}

 int main() {
    Eigen::internal::set_is_malloc_allowed(true);
    init_bdcsvd_mem();
    Eigen::internal::set_is_malloc_allowed(false);

    std::cout << "test_simple" << std::endl;
    test_simple();
}
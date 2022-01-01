#ifndef PINV_H_
#define PINV_H_

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Dense"

#define ROWS 2500
#define COLS 17
#define RCOND 1e-15

class PINV {

public:
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
            q0;
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

    PINV();
    ~PINV() {}

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
    void calculate_PINV();

    private:

    void _init_bdcsvd_mem();
};

#endif
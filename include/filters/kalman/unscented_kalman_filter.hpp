/*
 * This is an extension of the kalman filter class to implement the extended kalman filter
 * It is templated on the state vector type and size, the control vector type and size,
 * the measurement vector type and size, and the scalar type.
 */

#pragma once

// Eigen includes
#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Cholesky>

// STD imports
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>

#include <filters/utils/circular_buffer.hpp>
#include <filters/utils/log.hpp>

/* #define DETAILED_LOG */

/**
 * @brief The UnscentedKalmanFilter class implements the unscented kalman filter for nonlinear systems with non additive noise
 * @tparam D_TYPE The datatype that the filter will use for calculations, float or double
 * @tparam STATE_SIZE The size of the state vector
 * @tparam CONTROL_SIZE The size of the control vector
 * @tparam MEASUREMENT_SIZE The size of the measurement vector
 * @tparam PROCESS_NOISE_SIZE The size of the process noise vector
 * @tparam MEASUREMENT_NOISE_SIZE The size of the measurement noise vector
 * @tparam HISTORY_SIZE The size of the history buffer
 */

template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
class UnscentedKalmanFilter
{

  public:
    using StateVector                 = Eigen::Matrix<D_TYPE, STATE_SIZE,                          1>;
    using StateAugmentedVectorProcess = Eigen::Matrix<D_TYPE, STATE_SIZE + PROCESS_NOISE_SIZE,     1>;
    using StateAugmentedVectorUpdate  = Eigen::Matrix<D_TYPE, STATE_SIZE + MEASUREMENT_NOISE_SIZE, 1>;

    using CovarianceAugmentedProcess = Eigen::Matrix<D_TYPE, STATE_SIZE + PROCESS_NOISE_SIZE,     STATE_SIZE + PROCESS_NOISE_SIZE>;
    using CovarianceAugmentedUpdate  = Eigen::Matrix<D_TYPE, STATE_SIZE + MEASUREMENT_NOISE_SIZE, STATE_SIZE + MEASUREMENT_NOISE_SIZE>;

    using SigmaPointsProcess = Eigen::Matrix<D_TYPE,           STATE_SIZE + PROCESS_NOISE_SIZE,     2 * (STATE_SIZE + PROCESS_NOISE_SIZE) + 1>;
    using SigmaPointsUpdate  = Eigen::Matrix<D_TYPE,           STATE_SIZE + MEASUREMENT_NOISE_SIZE, 2 * (STATE_SIZE + MEASUREMENT_NOISE_SIZE) + 1>;
    using PropogatedSigmaPointsProcess = Eigen::Matrix<D_TYPE, STATE_SIZE,                          2 * (STATE_SIZE + PROCESS_NOISE_SIZE) + 1>;
    using PropogatedSigmaPointsUpdate  = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE,                    2 * (STATE_SIZE + MEASUREMENT_NOISE_SIZE) + 1>;

    using ControlVector     = Eigen::Matrix<D_TYPE, CONTROL_SIZE,     1>;
    using MeasurementVector = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE, 1>;

    using ProcessNoise     = Eigen::Matrix<D_TYPE, PROCESS_NOISE_SIZE,     PROCESS_NOISE_SIZE>;
    using MeasurementNoise = Eigen::Matrix<D_TYPE, MEASUREMENT_NOISE_SIZE, MEASUREMENT_NOISE_SIZE>;

    using CrossCorrelationMatrix = Eigen::Matrix<D_TYPE, STATE_SIZE, MEASUREMENT_SIZE>;
    using CovarianceMatrix       = Eigen::Matrix<D_TYPE, STATE_SIZE, STATE_SIZE>;
    using InnovationMatrix       = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE, MEASUREMENT_SIZE>;
    using KalmanGainMatrix       = Eigen::Matrix<D_TYPE, STATE_SIZE, MEASUREMENT_SIZE>;

    using MeasurementAugmentedVector = Eigen::Matrix<D_TYPE, MEASUREMENT_SIZE + MEASUREMENT_NOISE_SIZE, 1>;

    /**
     * @brief UnscentedKalmanFilter Constructor which computes the weights and lambda and sets the state and covariance to zero
     */
    UnscentedKalmanFilter();
    /**
     * @brief UKF Desructor. Does nothing
     */
    ~UnscentedKalmanFilter();

    void predictionStep(const ControlVector &control, const D_TYPE dt);

    void updateStep(const MeasurementVector &measurement);

    /**
     * @brief predict Predicts the state and covariance forward using the process model
     * @param control The control vector
     * @param timestamp The time stamp of the control vector
     */
    void predict(const ControlVector& u, D_TYPE timestamp);

    /**
     * @brief update Updates the state and covariance using the measurement model
     * @param measurement The measurement vector
     * @param timestamp The time stamp of the measurement vector
     */
    void update(MeasurementVector z, D_TYPE timestamp);
    
    /**
     * @brief Inititalizes the filter by setting the initial state and covariance
     * @param x0 The initial state
     * @param P0 The initial covariance
     * @param timestamp The time stamp of the initial state
     */
    void init(const StateVector& x0, const CovarianceMatrix& P0, D_TYPE timestamp);

    /**
     * @brief Get the index that is the closest timestamp which is before the given timestamp
     *
     * @param timestamp The timestamp to find the closest index for
     * @param buffer The buffer to search through
     * @return int The index of the closest timestamp
     */
    int getClosestIndex(D_TYPE timestamp, CircularBuffer<D_TYPE, HISTORY_SIZE>& buffer);

    /**
     * @brief Computes the dt and checks if its negative, printing a warning in the console if it is
     *
     * @param timestamp_next The timestamp to compute the dt for
     * @param timestamp_prev The previous timestamp
     * @return D_TYPE The dt
     */
    D_TYPE computeDt(D_TYPE timestamp_next, D_TYPE timestamp_prev);

    /**
     * @brief Compute the weights and lambda for the process model, which needs to be done anytime we change the tunable parameters
     */
    void computeProcessLambdaWeights();

    /**
     * @brief Compute the weights for the measurement sigma points, which needs to be done anytime we change the tunable parameters
     */
    void computeMeasurementLambdaWeights();

    // Setter and getter for the tunable parameters
    void tunableAlpha(D_TYPE tunable_alpha) { _tunable_process_alpha = tunable_alpha; computeProcessLambdaWeights(); computeMeasurementLambdaWeights(); }
    D_TYPE tunableAlpha() const { return _tunable_process_alpha; }
    void tunableBeta(D_TYPE tunable_beta) { _tunable_process_beta = tunable_beta;     computeProcessLambdaWeights(); computeMeasurementLambdaWeights();}
    D_TYPE tunableBeta() const { return _tunable_process_beta; }
    void tunableKappa(D_TYPE tunable_kappa) { _tunable_process_kappa = tunable_kappa; computeProcessLambdaWeights(); computeMeasurementLambdaWeights();}
    D_TYPE tunableKappa() const { return _tunable_process_kappa; }

    void systemModel(MODEL& system_model) { _system_model = system_model; }
    MODEL& systemModel() { return _system_model; }

    void predictionFlag(bool prediction_flag) { _prediction_flag = prediction_flag; }
    bool predictionFlag() const { return _prediction_flag; }

    void noiseUKFProcess(const ProcessNoise& noise_ukf_process) { _noise_ukf_process = noise_ukf_process; }
    ProcessNoise& noiseUKFProcess() { return _noise_ukf_process; }

    void noiseUKFMeasurement(const MeasurementNoise& noise_ukf_measurement) { _noise_ukf_measurement = noise_ukf_measurement; }
    MeasurementNoise& noiseUKFMeasurement() { return _noise_ukf_measurement; }

    /**
     * @brief Setter and getters for state and covariance
     */
    // !< Setter for state
    void state(StateVector state) { _x_t = state; }
    // !< Getter for state
    StateVector state() const { return _x_t; }
    // !< Setter for covariance
    void covariance(CovarianceMatrix covariance) { _covariance = covariance; };
    // !< Getter for covariance
    CovarianceMatrix covariance() const { return _covariance; }

    // !< Setter fpr initialized
    void initialized(bool initialized) { _initialized = initialized; }
    // !< Getter for initialized
    bool initialized() const { return _initialized; }

  private:
    // !< Creating a circular buffer for state, covariance, control and timestamp
    CircularBuffer<StateVector,      HISTORY_SIZE> _states;
    CircularBuffer<CovarianceMatrix, HISTORY_SIZE> _covariances;
    CircularBuffer<ControlVector,    HISTORY_SIZE> _controls;
    CircularBuffer<D_TYPE,           HISTORY_SIZE> _timestamps;

    StateVector       _x_t;
    CovarianceMatrix  _covariance;

    StateAugmentedVectorProcess _augmented_state_process; //!> Augmented state vector for the process model that has the state vector augmented with the process noise
    StateAugmentedVectorUpdate  _augmented_state_update;  //!> Augmented state vector for the update model that has the state vector augmented with the measurement noise

    CovarianceAugmentedProcess _augmented_covariance_process; //!> Augmented covariance matrix for the process model that has the state vector augmented with the process noise
    CovarianceAugmentedUpdate _augmented_covariance_update;  //!> Augmented covariance matrix for the update model that has the state vector augmented with the measurement noise
    
    SigmaPointsProcess _sigma_points_process; //!> Sigma points for the process model
    SigmaPointsUpdate  _sigma_points_update;  //!> Sigma points for the update model
    
    PropogatedSigmaPointsProcess _propogated_sigma_points_process; //!> Sigma points for the process model after being propogated through the process model
    PropogatedSigmaPointsUpdate  _propogated_sigma_points_update;  //!> Sigma points for the update model after being propogated through the measurement model

    StateVector       _predicted_state;       //!> Predicted state vector, weighted average of the mean of the sigma points after being propogated through the process model
    MeasurementVector _predicted_measurement; //!> Predicted measurement vector, weighted average of the mean of the sigma points after being propogated through the measurement model

    ProcessNoise     _noise_ukf_process;     //!> Process noise matrix. Special for UKF for non additive noise models
    MeasurementNoise _noise_ukf_measurement; //!> Measurement noise matrix. Special for UKF for non additive noise models

    CrossCorrelationMatrix _unscented_measurement_matrix_C; //!> Unscented cross correlation matrix
    InnovationMatrix       _unscented_innovation_matrix_S; //!> Unscented innovation matrix
    KalmanGainMatrix       _unscented_kalman_gain_K;      //!> Unscented Kalman gain matrix

    D_TYPE _tunable_process_alpha = 0.001; //!> Alpha tunes the spread of the sigma points around the mean. Tunable parameter, default values works well for gaussian approximation. Dont change unless you know what you are doing
    D_TYPE _tunable_process_beta  = 2.0;   //!> Beta tunes the spread of the sigma points around the tail. Tunable parameter, default values works well in many cases. Dont change unless you know what you are doing
    D_TYPE _tunable_process_kappa = 1.0;   //!> Kappa tunes the spread of the sigma points around the head. Tunable parameter, default values works well in many cases. Dont change unless you know what you are doing
    D_TYPE _process_lambda;                //!> Lambda for the process model, computed using alpha, kappa and augmented state size
    
    D_TYPE _tunable_measurement_alpha = 0.001; //!> Alpha tunes the spread of the sigma points around the mean. Tunable parameter, default values works well for gaussian approximation. Dont change unless you know what you are doing
    D_TYPE _tunable_measurement_beta  = 2.0;   //!> Beta tunes the spread of the sigma points around the tail. Tunable parameter, default values works well in many cases. Dont change unless you know what you are doing
    D_TYPE _tunable_measurement_kappa = 1.0;   //!> Kappa tunes the spread of the sigma points around the head. Tunable parameter, default values works well in many cases. Dont change unless you know what you are doing
    D_TYPE _measurement_lambda;                //!> Lambda for the process model, computed using alpha, kappa and augmented state size


    D_TYPE _weight_mean_0_process;       //!> Corresponding weight for the mean of the sigma points at index 0 after being propogated through the process model
    D_TYPE _weight_mean_i_process;       //!> Corresponding weight for the mean of the sigma points at index i after being propogated through the process model
    D_TYPE _weight_covariance_0_process; //!> Corresponding weight for the covariance of the sigma points at index 0 after being propogated through the process model
    D_TYPE _weight_covariance_i_process; //!> Corresponding weight for the covariance of the sigma points at index i after being propogated through the process model

    D_TYPE _weight_mean_0_update;       //!> Corresponding weight for the mean of the sigma points at index 0 after being propogated through the measurement model
    D_TYPE _weight_mean_i_update;       //!> Corresponding weight for the mean of the sigma points at index i after being propogated through the measurement model
    D_TYPE _weight_covariance_0_update; //!> Corresponding weight for the covariance of the sigma points at index 0 after being propogated through the measurement model
    D_TYPE _weight_covariance_i_update; //!> Corresponding weight for the covariance of the sigma points at index i after being propogated through the measurement model

    uint8_t _measurement_source; //!> Source of the measurement, 0 for gps, 1 for barometer, 2 for rangefinder

    MODEL _system_model;
    
    const int _number_augmented_states_process = STATE_SIZE + PROCESS_NOISE_SIZE;               //!> Number of augmented states for the process model
    const int _number_augmented_states_update  = STATE_SIZE + MEASUREMENT_NOISE_SIZE;           //!> Number of augmented states for the update model
    const int _number_sigma_points_process     = (STATE_SIZE + PROCESS_NOISE_SIZE)     * 2 + 1; //!> Number of sigma points for the process model
    const int _number_sigma_points_update      = (STATE_SIZE + MEASUREMENT_NOISE_SIZE) * 2 + 1; //!> Number of sigma points for the update model

    bool _prediction_flag = false; //!> Flag to indicate if the prediction step has been performed
    bool _initialized = false;
#ifdef DETAILED_LOG
    std::stringstream _ss;
#endif
};

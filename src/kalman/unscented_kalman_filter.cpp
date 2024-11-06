#include <filters/kalman/unscented_kalman_filter.hpp> 

/**
 * @brief UnscentedKalmanFilter Constructor which computes the weights and lambda and sets the state and covariance to zero
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE, PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::UnscentedKalmanFilter()
{ 
  // Compute the weights and Lambda
  computeProcessLambdaWeights();
  computeMeasurementLambdaWeights();
  
  // initialize augmented state to 0
  _augmented_state_process      = StateAugmentedVectorProcess::Zero();
  _augmented_covariance_process = CovarianceAugmentedProcess::Zero();
  _augmented_state_update       = StateAugmentedVectorUpdate::Zero();
  _augmented_covariance_update  = CovarianceAugmentedUpdate::Zero();
}

template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE, 
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           predictionStep(const ControlVector &control, const D_TYPE dt)
{
  // !< Compute Augmented State
  _augmented_state_process.block(0, 0, STATE_SIZE, 1) = state();
  _augmented_state_process.block(STATE_SIZE, 0, PROCESS_NOISE_SIZE, 1) = Eigen::Matrix<D_TYPE, PROCESS_NOISE_SIZE, 1>::Zero();

  //!> Compute the augmented covariance matrix
  _augmented_covariance_process.block(0, 0, STATE_SIZE, STATE_SIZE) = covariance();
  _augmented_covariance_process.block(STATE_SIZE, STATE_SIZE, PROCESS_NOISE_SIZE, PROCESS_NOISE_SIZE) = _noise_ukf_process;

  //!> Doing cholesky decomposition on the augmented covariance matrix
  Eigen::LLT<Eigen::Matrix<D_TYPE, STATE_SIZE+PROCESS_NOISE_SIZE, STATE_SIZE+PROCESS_NOISE_SIZE>> lltOfA(_augmented_covariance_process);
  Eigen::Matrix<D_TYPE, STATE_SIZE+PROCESS_NOISE_SIZE, STATE_SIZE+PROCESS_NOISE_SIZE> lltMat = lltOfA.matrixL();

  //!> Compute Sigma points
  _sigma_points_process.col(0)            = _augmented_state_process; // Sigma points of the same size of the augmented state
  // !> Propogate the sigma points through the process model
  _propogated_sigma_points_process.col(0) = _system_model.stateTransitionFunction(_sigma_points_process.col(0), control, dt);
  // !> Predict the state for sigma point 0
  _predicted_state                        = _weight_mean_0_process * _propogated_sigma_points_process.col(0);
  for(int i = 0; i < STATE_SIZE+PROCESS_NOISE_SIZE; i++)
  {
    // !< Compute sigma points for +-sqrt(covariance)
    _sigma_points_process.col(i+1)                               = _augmented_state_process + std::sqrt(STATE_SIZE+PROCESS_NOISE_SIZE+_process_lambda)*lltMat.col(i);
    _sigma_points_process.col(i+1+STATE_SIZE+PROCESS_NOISE_SIZE) = _augmented_state_process - std::sqrt(STATE_SIZE+PROCESS_NOISE_SIZE+_process_lambda)*lltMat.col(i);
    
    // !< Propogate the sigma points through the process model
    _propogated_sigma_points_process.col(i+1)                               = _system_model.stateTransitionFunction(_sigma_points_process.col(i+1), control, dt); 
    _propogated_sigma_points_process.col(i+1+STATE_SIZE+PROCESS_NOISE_SIZE) = _system_model.stateTransitionFunction(_sigma_points_process.col(i+1+STATE_SIZE+PROCESS_NOISE_SIZE), control, dt); 

    // !< Predict the state for sigma point i
    _predicted_state += _weight_mean_i_process*_propogated_sigma_points_process.col(i+1);
    _predicted_state += _weight_mean_i_process*_propogated_sigma_points_process.col(i+1+STATE_SIZE+PROCESS_NOISE_SIZE);
  }
  state(_predicted_state);

  //!> Compute the predicted covariance
  //!> Covariance = Sigma from 0 to 2n+1 (Wi * (Zi - U) * (Zi - U)^T)
  //!> n  -> number of states + noise
  //!> Wi -> ith weight
  //!> Zi -> Propogated sigma point
  //!> U  -> mean computed above
  covariance(_weight_covariance_0_process * ((_propogated_sigma_points_process.col(0)-state()) * (_propogated_sigma_points_process.col(0)-state()).transpose()));
  for(auto i = 1; i < 2*(STATE_SIZE+PROCESS_NOISE_SIZE)+1; i++)
  {
    covariance(covariance() + _weight_covariance_i_process*(_propogated_sigma_points_process.col(i)-state())*(_propogated_sigma_points_process.col(i)-state()).transpose());
  }
}

template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE,int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE, 
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           updateStep(const MeasurementVector &measurement)
{
  _augmented_state_update.block(0, 0, STATE_SIZE, 1) = state();
  _augmented_state_update.block(STATE_SIZE, 0, MEASUREMENT_NOISE_SIZE, 1) = Eigen::Matrix<D_TYPE, MEASUREMENT_NOISE_SIZE, 1>::Zero();
  

  // Compute the augmented covariance matrix
  _augmented_covariance_update.block(0, 0, STATE_SIZE, STATE_SIZE) = covariance();
  _augmented_covariance_update.block(STATE_SIZE, STATE_SIZE, MEASUREMENT_NOISE_SIZE, MEASUREMENT_NOISE_SIZE) = _noise_ukf_measurement;

  // Doing cholesky decomposition on the augmented covariance matrix to find the square root of the augmented covariance matrix
  Eigen::LLT<Eigen::Matrix<D_TYPE, STATE_SIZE+MEASUREMENT_NOISE_SIZE, STATE_SIZE+MEASUREMENT_NOISE_SIZE>> lltOfA(_augmented_covariance_update);
  Eigen::Matrix<D_TYPE, STATE_SIZE+MEASUREMENT_NOISE_SIZE, STATE_SIZE+MEASUREMENT_NOISE_SIZE> lltMat = lltOfA.matrixL();

  // Compute Sigma points propoagatet them and compute the mean
  _sigma_points_update.col(0)            = _augmented_state_update;
  _propogated_sigma_points_update.col(0) = _system_model.measurementFunction(_sigma_points_update.col(0));
  _predicted_measurement                 = _weight_mean_0_update * _propogated_sigma_points_update.col(0);
  /* _propogated_sigma_points_update.col(0) = _system_model.measurementFunction(_sigma_points_update.col(0)); */
  /* _predicted_measurement = _weight_mean_0_update * _propogated_sigma_points_update.col(0); */
  for(int i = 0; i < STATE_SIZE+MEASUREMENT_NOISE_SIZE; i++)
  {
    _sigma_points_update.col(i+1)            = _augmented_state_update + std::sqrt(STATE_SIZE+MEASUREMENT_NOISE_SIZE+_measurement_lambda)*lltMat.col(i);
    _propogated_sigma_points_update.col(i+1) = _system_model.measurementFunction(_sigma_points_update.col(i+1));
    _predicted_measurement                  += _weight_mean_i_update * _propogated_sigma_points_update.col(i+1);
    
    _sigma_points_update.col(i+1+STATE_SIZE+MEASUREMENT_NOISE_SIZE)            = _augmented_state_update - std::sqrt(STATE_SIZE+MEASUREMENT_NOISE_SIZE+_measurement_lambda)*lltMat.col(i);
    _propogated_sigma_points_update.col(i+1+STATE_SIZE+MEASUREMENT_NOISE_SIZE) = _system_model.measurementFunction(_sigma_points_update.col(i+1+STATE_SIZE+MEASUREMENT_NOISE_SIZE));
    _predicted_measurement                                                    += _weight_mean_i_update * _propogated_sigma_points_update.col(i+1+STATE_SIZE+MEASUREMENT_NOISE_SIZE);
  }

  // Compute the measurement matrix and innovation covariance
  _unscented_measurement_matrix_C = _weight_covariance_0_update * (_sigma_points_update.col(0).block(0, 0, STATE_SIZE, 1) - state()) * (_propogated_sigma_points_update.col(0) - _predicted_measurement).transpose();
  _unscented_innovation_matrix_S = (_weight_covariance_0_update * (_propogated_sigma_points_update.col(0) - _predicted_measurement) * (_propogated_sigma_points_update.col(0) - _predicted_measurement).transpose());
  for(int i = 1; i < 2*(STATE_SIZE+MEASUREMENT_NOISE_SIZE)+1; i++)
  {
    _unscented_measurement_matrix_C += _weight_covariance_i_update * (_sigma_points_update.col(i).block(0, 0, STATE_SIZE, 1) - state()) * (_propogated_sigma_points_update.col(i) - _predicted_measurement).transpose();
    _unscented_innovation_matrix_S += (_weight_covariance_i_update * (_propogated_sigma_points_update.col(i) - _predicted_measurement) * (_propogated_sigma_points_update.col(i) - _predicted_measurement).transpose());
  }
  
  _unscented_kalman_gain_K = (_unscented_measurement_matrix_C * _unscented_innovation_matrix_S.inverse());
  covariance(covariance() - ((_unscented_kalman_gain_K * _unscented_innovation_matrix_S) * _unscented_kalman_gain_K.transpose()));
  state(state() + _unscented_kalman_gain_K * (measurement - _predicted_measurement));
}


/**
 * @brief predict Predicts the state and covariance forward using the process model
 * @param control The control vector
 * @param timestamp The time stamp of the control vector
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           predict(const ControlVector& u, D_TYPE timestamp)
{
  // <! Update the state, covariance and dt 
  D_TYPE dt = computeDt(timestamp, _timestamps.get(_states.size() - 1));
  // TODO: Handle this better
  if(dt < 0)
    return;
  _x_t = _states.get(_states.size() - 1);
  _covariance = _covariances.get(_covariances.size() - 1);

  predictionStep(u, dt);

  // <! Update the buffer of state, covariance and dt
  _states.add(state());
  _covariances.add(covariance());
  _controls.add(u);
  _timestamps.add(timestamp);
}

/**
 * @brief update Updates the state and covariance using the measurement model
 * @param measurement The measurement vector
 * @param timestamp The time stamp of the measurement vector
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           update(MeasurementVector z, D_TYPE timestamp)
{
  // !< Find the index for the timestamps 
  int index = getClosestIndex(timestamp, _timestamps);
#ifdef DETAILED_LOG
  _ss << "Update index is: " << index << ", input TS: " << std::setprecision(20) << timestamp << ", selected: " << _timestamps.get(index) << "\n";
  _ss << "At index 0:  " << std::setprecision(20) <<  _timestamps.get(0) << "\n";
  _ss << "At index -1: " << std::setprecision(20) <<  _timestamps.get(_timestamps.size() - 1) << "\n";
  vLog(2, _ss);
#endif

  // !< Compute dt
  D_TYPE dt = computeDt(timestamp, _timestamps.get(index));

  _x_t = _states.get(index);
  _covariance = _covariances.get(index);

  // !< Redoing prediction for the state and covariance at given timestamp
  predictionStep(_controls.get(index), dt);
  
  // !< Update step
  updateStep(z);

  // <! Update the buffer of state, covariance and dt
  _states.set(index, state());
  _covariances.set(index, covariance());
  _timestamps.set(index, timestamp);

  // <! Update the buffer of control and timestamp
  _states.removeTail(index);
  _covariances.removeTail(index);
  _timestamps.removeTail(index);
  _controls.removeTail(index);

  // !< Propogate the changes forward
  for(int k = 0; k < _states.size() - 1; k++)
  {
    _x_t = _states.get(k);
    _covariance = _covariances.get(k);
    ControlVector u = _controls.get(k);
    D_TYPE dt = computeDt(_timestamps.get(k+1), _timestamps.get(k));
    predictionStep(u, dt);
    _states.set(k+1, state());
    _covariances.set(k+1, covariance());
  }

}

/**
 * @brief Inititalizes the filter by setting the initial state and covariance
 * @param x0 The initial state
 * @param P0 The initial covariance
 * @param timestamp The time stamp of the initial state
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           init(const StateVector& x0, const CovarianceMatrix& P0, D_TYPE timestamp)
{
  if(_states.size() > 0)
  {
    std::cout << "States size is: << " << _states.size() << std::endl;
    std::cout << "ERROR NOT EMPTY FILTER\n";
    return;
  }
  _states.add(x0);
  _controls.add(ControlVector::Zero());
  _timestamps.add(timestamp);
  _covariances.add(P0);

  _initialized = true;
  std::cout << "KF Initialized\n";
}

/**
 * @brief Get the index that is the closest timestamp which is before the given timestamp
 *
 * @param timestamp The timestamp to find the closest index for
 * @param buffer The buffer to search through
 * @return int The index of the closest timestamp
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
int UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                          PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                          getClosestIndex(D_TYPE timestamp, CircularBuffer<D_TYPE, HISTORY_SIZE>& buffer)
{
  int index = buffer.size() - 1;
  for(; index >= 0 && (buffer.get(index) - timestamp) > 0; index--);
  return index;
}


/**
 * @brief Computes the dt and checks if its negative, printing a warning in the console if it is
 *
 * @param timestamp_next The timestamp to compute the dt for
 * @param timestamp_prev The previous timestamp
 * @return D_TYPE The dt
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
D_TYPE UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                             PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                             computeDt(D_TYPE timestamp_next, D_TYPE timestamp_prev)
{
  D_TYPE dt = timestamp_next - timestamp_prev;
  if(dt < 0)
  {
    std::cout << "dt is negative: " << dt << "\n";
    std::cout << "This should not happend. Check your timestamps.\n";
  }
  return dt;
}

/**
 * @brief Compute the weights and lambda for the process model, which needs to be done anytime we change the tunable parameters
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           computeProcessLambdaWeights()
{
  /*
   * lambda       = alpha^2                                      * (n                               + kappa)                  -  n
   */
  _process_lambda = _tunable_process_alpha*_tunable_process_alpha*(_number_augmented_states_process + _tunable_process_kappa) - _number_augmented_states_process;

  _weight_mean_0_process = _process_lambda/(STATE_SIZE+PROCESS_NOISE_SIZE+_process_lambda);
  _weight_mean_i_process = 1/(2*(STATE_SIZE+PROCESS_NOISE_SIZE+_process_lambda));
  _weight_covariance_0_process= _weight_mean_0_process + (1-_tunable_process_alpha*_tunable_process_alpha+_tunable_process_beta);
  _weight_covariance_i_process = _weight_mean_i_process;
}

/**
 * @brief Compute the weights for the measurement sigma points, which needs to be done anytime we change the tunable parameters
 */
template<class MODEL, typename D_TYPE, int STATE_SIZE, int CONTROL_SIZE, int MEASUREMENT_SIZE, int PROCESS_NOISE_SIZE, int MEASUREMENT_NOISE_SIZE, int HISTORY_SIZE>
void UnscentedKalmanFilter<MODEL, D_TYPE, STATE_SIZE, CONTROL_SIZE, MEASUREMENT_SIZE,
                           PROCESS_NOISE_SIZE, MEASUREMENT_NOISE_SIZE, HISTORY_SIZE>::
                           computeMeasurementLambdaWeights()
{
  /*
   * lambda           = alpha^2                                              * (n                              + kappa)                  -  n
   */
  _measurement_lambda = _tunable_measurement_alpha*_tunable_measurement_alpha*(_number_augmented_states_update + _tunable_measurement_kappa) - _number_augmented_states_update;
  
  _weight_mean_0_update = _measurement_lambda/(STATE_SIZE+MEASUREMENT_NOISE_SIZE+_measurement_lambda);
  _weight_mean_i_update = 1/(2*(STATE_SIZE+MEASUREMENT_NOISE_SIZE+_measurement_lambda));
  _weight_covariance_0_update= _weight_mean_0_update + (1-_tunable_measurement_alpha*_tunable_measurement_alpha+_tunable_measurement_beta);
  _weight_covariance_i_update = _weight_mean_i_update;
}



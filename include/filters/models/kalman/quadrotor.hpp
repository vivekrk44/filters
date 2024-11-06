/* This library fuses position information from the GPS with acceleration from the IMU
 * using an Unscented Kalman Filter. The filter is implemented in the UKF class from 
 * the zf_lib
 * The comments are made in the style of the doxygen documentation
 */

#include <eigen3/Eigen/Dense>

#define TYPE double
#define N_STATES 16
#define N_CONTROLS 6
#define N_MEASUREMENTS 7
#define N_PROCESS_NOISES 12
#define N_MEASUREMENT_NOISES 7
#define N_HISTORY 200


/*
 * @brief This class implements a system model for an Unscented Kalman Filter for a quadrotor with the following sensors
 *       - IMU
 *       - GPS
 *       - Barometer
 *       - Magnetometer
 *  The IMU provides acceleration and angular velocity measurements in the body frame as control input which is used in the prediction step to compute 
 *  the state transition. The GPS provides position, velocity and orientation information as we take the fused output of the GPS as measurement. Here, we 
 *  assume the GPS position Z axis measurement is not very accurate and tends to drift. We correct this using the barometer and the rangefinder.
 *  
 *  We use 16 state variables which are the following:
 *  - 3 position (X, Y, Z)
 *  - 3 velocity (X, Y, Z)
 *  - 4 orientation variables (quaternion W, X, Y, Z)
 *  - 3 accel bias variables
 *  - 3 gyro bias variables
 *
 *  There are 12 non additive noise variables in the prediction step which are the following:
 *  - 3 accel noise variables
 *  - 3 gyro noise variables
 *  - 3 accel bias noise variables
 *  - 3 gyro bias noise variables
 *
 *  There are 6 control variables which are the following:
 *  - 3 acceleration variables
 *  - 3 angular velocity variables
 *
 *  There are 10 measurement variables which are the following:
 *  - 3 position variables
 *  - 3 velocity variables
 *  - 4 orientation variables (quaternion W, X, Y, Z)
 *
 *  There are 10 measurement noise variables which are the following:
 *  - 3 position noise variables
 *  - 3 velocity noise variables
 *  - 4 orientation noise variables (quaternion W, X, Y, Z)
 *
 *  The state transition function is given by finding the differential of the state variables with respect to time. The state transition function is given by
 *   x_dot = f(x,u,n) where x is the state vector, u is the control vector and n is the process noise vector. The state transition function is given by
 *   
 *  x_dot = [v, Rbw * (u_a - n_a - b_a) + ge3, 0.5 * Q(u_w - n_g - b_g) * Rbw, n_ba, n_bw]
 *  where
 *  - v    is the velocity in the world frame, a direct linear relation to the velocity state
 *  - Rbw  is the rotation from the body frame to the world frame as a quaternion or the current orientation as a quaternion
 *  - Q    is a function that converts the angular velocity to a quaternion (see below for details)
 *  - u_a  is the acceleration in the body frame
 *  - n_a  is the acceleration noise
 *  - b_a  is the acceleration bias
 *  - ge3  is the gravity vector in the world frame
 *  - u_w  is the angular velocity in the body frame
 *  - n_g  is the angular velocity noise
 *  - b_g  is the angular velocity bias
 *  - n_ba is the acceleration bias noise
 *  - n_bw is the angular velocity bias noise
 *
 *  Q(wx, wy, wz)
 *  {
 *    mag = sqrt(wx^2 + wy^2 + wz^2);
 *    if (mag < 1e-5)
 *    {
 *    return [1, 0, 0, 0];
 *    }
 *    else
 *    {
 *    return [cos(mag/2), sin(mag/2) * wx/mag, sin(mag/2) * wy/mag, sin(mag/2) * wz/mag];
 *    }
 *  }
 *
 *  Here the noise is not additive but used in the prediction step to compute the state transition and thus the non additive noise kalman filter works best
 *
 *  The measurement model is given by
 *
 *  z = h(x,v) where z is the measurement vector, x is the state vector and v is the measurement noise vector
 *
 *  z = [p + n_p, v + n_v, phi + n_phi, theta + n_theta, psi + n_psi]
 *  where
 *  - p is the position in the world frame
 *  - n_p is the position noise
 *  - v is the velocity in the world frame
 *  - n_v is the velocity noise
 *  - phi is the roll angle
 *  - n_phi is the roll angle noise
 *  - theta is the pitch angle
 *  - n_theta is the pitch angle noise
 *  - psi is the yaw angle
 *  - n_psi is the yaw angle noise
 *
 *  However the rangefinder and the barometer use a slightly different model for the position Z measurement. The rangefinder model is given by
 *  z = [p_x + n_p_x, p_y + n_p_y, (p_z + n_p_z)/(cos(phi) * cos(theta)), v_x + n_v_x, v_y + n_v_y, v_z + n_v_z, phi + n_phi, theta + n_theta, psi + n_psi]
 *  This is different because the rangefinder measures the distance from the ground and not the altitude which changes depending on the orientation of the drone
 *
 *  Similarly the barometer measures the air pressure and we use the barometric formula to get the height from the air pressure. The barometer model is given by
 *  height = (T0 / L) * (1 - (p / p0)^(R * L / g0))
 *  where
 *  - T0 is the temperature at sea level
 *  - L is the temperature lapse rate
 *  - p is the air pressure
 *  - p0 is the air pressure at sea level
 *  - R is the universal gas constant
 *  - g0 is the acceleration due to gravity at sea level
 *  - height is the height from the sea level
 *
 *  The measurement model for the barometer is given by
 *  pressure = p0 * (1 - (L * height) / T0)^(g0 / (R * L))
 */

class UKFQuad
{

  public:

    UKFQuad()
    {
    }

    /**
     * @brief This function computes the state trasition given the augmented state matrix and control input
     *
     * @param x The state vector augmented with the noise variables
     * @param u The control input vector
     *
     * @return The predicted state given current state and control input
     */
    Eigen::Matrix<TYPE, N_STATES, 1> stateTransitionFunction(Eigen::Matrix<TYPE, N_STATES+N_PROCESS_NOISES, 1> x, const Eigen::Matrix<TYPE, N_CONTROLS, 1> u, TYPE dt=0)
    {
      // !< Update dt
      _dt = dt;

      /**
       * Precomputed to reduce computation time
       */
      
      _Rgyro_quat.w() = 0.0;
      Eigen::Matrix<TYPE, 3, 1> u_w;
      u_w << (u(3) - x(13) - x(19))/2.0,
             (u(4) - x(14) - x(20))/2.0,
             (u(5) - x(15) - x(21))/2.0;
      
      TYPE norm_u_w = u_w.norm();

      TYPE sin_norm = sin(norm_u_w/2.0);
      TYPE cos_norm = cos(norm_u_w/2.0);
      _Rgyro_quat.w() = norm_u_w < 1e-5 ? 1.0 : cos_norm;
      _Rgyro_quat.x() = norm_u_w < 1e-5 ? 0.0 : u_w(0)*sin_norm/norm_u_w;
      _Rgyro_quat.y() = norm_u_w < 1e-5 ? 0.0 : u_w(1)*sin_norm/norm_u_w;
      _Rgyro_quat.z() = norm_u_w < 1e-5 ? 0.0 : u_w(2)*sin_norm/norm_u_w;

      _quat_current.w() = x(6);
      _quat_current.x() = x(7);
      _quat_current.y() = x(8);
      _quat_current.z() = x(9);

      _x_dot.setZero(); //!< Initialize the state transition function to zero
      
      /**
       * Position dot = Velocity
       */
      _x_dot(0) = x(3);
      _x_dot(1) = x(4);
      _x_dot(2) = x(5);
      
      /**
       *  Velocity dot = Rotation_matrix * Body Acceleration + Gravity
       * Transform the acceleration from the body frame to the world frame
       *                          currentQuat  *   measured acc body   - bias acc body       - noise acc body
       */
      _x_dot.block<3, 1>(3, 0) = _quat_current * (u.block<3, 1>(0, 0) - x.block<3, 1>(10, 0) - x.block<3, 1>(16, 0));
      _x_dot(5) += _gravity;
      
      /**
       * Orientation dot = Rotation_matrix * Body Rotation_rate
       */
      Eigen::Quaternion<TYPE> _quat_dot;
      _quat_dot = _Rgyro_quat * _quat_current;

      _x_dot(6) = _quat_dot.w();
      _x_dot(7) = _quat_dot.x();
      _x_dot(8) = _quat_dot.y();
      _x_dot(9) = _quat_dot.z();

      /**
       * Accel Bias dot = noise bias accelerometer
       */
      _x_dot(10) = x(22);
      _x_dot(11) = x(23);
      _x_dot(12) = x(24);

      /**
       * Gyro Bias dot = noise bias gyroscope
       */
      
      _x_dot(13) = x(25);
      _x_dot(14) = x(26);
      _x_dot(15) = x(27);

      /**
       * Update the state vector
       * x = x + x_dot * dt
       */
      x.block(0, 0, N_STATES, 1) += (_x_dot * _dt);
      x.block(6, 0, 4, 1).normalize();

      return x.block(0, 0, N_STATES, 1);
    }

    /**
     * 
     * @brief This function takes the state vector augmented by the measurement noise to return the predicted measurement given the state
     *        and measurement noise
     * 
     * @param x The state vector augmented with the measurement noise
     * @return The predicted measurement as Eigen Matrix
     */
    Eigen::Matrix<TYPE, N_MEASUREMENTS, 1> measurementFunction(const Eigen::Matrix<TYPE, N_STATES+N_MEASUREMENT_NOISES, 1>& x)
    {
      /**
       * Here we can fuse the measurements from different sensors. Depending on the value of the measurement_source.
       * We can use different measurement model to fuse the measurements.
       * measurement_source = 0: GPS
       * measurement_source = 1: Baro
       * measurement_source = 2: Laser Rangefinder
       * measurement_source = 3: Raw GPS
       */

      Eigen::Matrix<TYPE, N_MEASUREMENTS, 1> z; //!< The predicted measurement
      
      /**
       * The state x consists of 15 elements (x, y, z, vx, vy, vz, qw, qx, qy, qz,
       * ax_bias, ay_bias, az_bias, wx_bias, wy_bias, wz_bias) and 9 noise elements
       * (position_noise, velocity_noise, orientation_noise) each as a 3x1 vector
       * The measurement z consists of 9 elements the position (x, y, z) and velocities (vx, vy, vz)
       * and orientation (phi, theta, psi).
       */
      z(0) = x(0) - x(16); //!< measured x position = x - x_noise
      z(1) = x(1) - x(17); //!< measured y position = y - y_noise
      z(2) = x(2) - x(18); //!< measured z position = z - z_noise

      /**
       * The orientation is a direct linear function of the state
       */
      z(6) = x(6) - x(22); //!< measured roll = roll - roll_noise
      z(7) = x(7) - x(23); //!< measured pitch = pitch - pitch_noise
      z(8) = x(8) - x(24); //!< measured yaw = yaw - yaw_noise
      z(9) = x(9) - x(25); //!< measured yaw = yaw - yaw_noise
      z.block(6, 0, 4, 1).normalize();
      return z;
    }

    /**
     * @brief Updates the timestep interval
     * @param dt The timestep interval in seconds
     */
    void dt(TYPE dt){ dt = dt; }
    /**
     * @brief Returns the timestep interval
     * @return The timestep interval in seconds
     */
    TYPE dt() const { return _dt; }

    void gravity(TYPE gravity){ _gravity = gravity; }
    TYPE gravity() const { return _gravity; }

    TYPE _dt;

  private: 
    TYPE _gravity = 9.81; //!< Gravity constant 
    Eigen::Matrix<TYPE, N_STATES, 1> _x_dot; //!< State derivative

    Eigen::Matrix<TYPE, 3, 3> _Rbw; //!< Rotation matrix from body to world frame
    Eigen::Quaternion<TYPE>   _Rgyro_quat; //!< Rotation matrix from body to world frame
    Eigen::Quaternion<TYPE>   _quat_current; //!< Current quaternion

};


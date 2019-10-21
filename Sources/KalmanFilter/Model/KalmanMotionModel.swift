import Surge
import StateSpace
import StateSpaceModel

// MARK: - Protocols

public protocol KalmanMotionModel: UncontrollableMotionModelProtocol & DifferentiableMotionModelProtocol
    where State == Vector<Double>,
          Jacobian == Matrix<Double>
{
    // Nothing
}

public protocol ControllableKalmanMotionModel: ControllableMotionModelProtocol & ControllableDifferentiableMotionModelProtocol
    where State == Vector<Double>,
          Control == Vector<Double>,
          Jacobian == Matrix<Double>
{
    // Nothing
}

// MARK: - Uncontrollable Motion Models

extension ZeroMotionModel: KalmanMotionModel {
    // Nothing
}

extension LinearMotionModel: KalmanMotionModel {
    // Nothing
}

extension BrownianMotionModel: KalmanMotionModel
    where MotionModel: KalmanMotionModel
{
    // Nothing
}

extension NonlinearMotionModel: KalmanMotionModel {
    // Nothing
}

// MARK: - Controllable Motion Models

extension BrownianMotionModel: ControllableKalmanMotionModel
    where MotionModel: ControllableKalmanMotionModel
{
    // Nothing
}

extension ControllableLinearMotionModel: ControllableKalmanMotionModel
    where MotionModel: KalmanMotionModel,
          ControlModel: KalmanControlModel
{
    // Nothing
}

extension ControllableNonlinearMotionModel: ControllableKalmanMotionModel {
    // Nothing
}

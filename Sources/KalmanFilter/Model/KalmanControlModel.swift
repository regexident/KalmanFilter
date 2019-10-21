import Surge
import StateSpace
import StateSpaceModel

// MARK: - Protocols

public protocol KalmanControlModel: ControlModelProtocol
    where State == Vector<Double>,
          Control == Vector<Double>
{
    // Nothing
}

// MARK: - Control Models

extension LinearControlModel: KalmanControlModel {
    // Nothing
}

import Surge
import StateSpace
import StateSpaceModel

// MARK: - Protocols

public protocol KalmanObservationModel: ObservationModelProtocol & DifferentiableObservationModelProtocol
    where State == Vector<Double>, Observation == Vector<Double>, Jacobian == Matrix<Double>
{
    // Nothing
}

// MARK: - Observation Models

extension TransparentObservationModel: KalmanObservationModel {
    // Nothing
}

extension LinearObservationModel: KalmanObservationModel {
    // Nothing
}

extension NonlinearObservationModel: KalmanObservationModel {
    // Nothing
}

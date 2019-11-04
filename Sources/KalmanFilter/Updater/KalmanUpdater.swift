import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public typealias MultiModalKalmanUpdater = MultiModalBayesUpdater

public protocol KalmanUpdaterProtocol: BayesUpdaterProtocol {
    associatedtype ObservationModel: KalmanObservationModel
}

public class KalmanUpdater<ObservationModel> {
    /// Observation model (used for correction).
    public var observationModel: ObservationModel

    /// Observation noise matrix (aka `R`)
    ///
    /// This matrix implies the observation error covariance,
    /// based on the amount of sensor noise.
    ///
    /// Default: zero matrix.
    public var observationNoise: Matrix<Double>

    private var cachedIdentity: Matrix<Double>? = nil

    public init(
        observationModel: ObservationModel,
        observationNoise: Matrix<Double>
    ) {
        self.observationModel = observationModel
        self.observationNoise = observationNoise
    }

    private func identity(size: Int) -> Matrix<Double> {
        let identity = self.cachedIdentity ?? Matrix.identity(size: size)

        assert(identity.rows == size, "Malformed identity matrix")
        assert(identity.columns == size, "Malformed identity matrix")

        self.cachedIdentity = identity

        return identity
    }
}

extension KalmanUpdater: DimensionsValidatable {
    public func validate(for dimensions: DimensionsProtocol) throws {
        if let validatableModel = self.observationModel as? DimensionsValidatable {
            try validatableModel.validate(for: dimensions)
        }

        guard self.observationNoise.columns == dimensions.observation else {
            throw MatrixError.invalidColumnCount(
                message: "Expected \(dimensions.observation) columns in `self.observationNoise`, found \(self.observationNoise.columns)"
            )
        }

        guard self.observationNoise.rows == dimensions.observation else {
            throw MatrixError.invalidRowCount(
                message: "Expected \(dimensions.observation) columns in `self.observationNoise`, found \(self.observationNoise.rows)"
            )
        }
    }
}

extension KalmanUpdater: Statable {
    public typealias State = Vector<Double>
}

extension KalmanUpdater: Observable {
    public typealias Observation = Vector<Double>
}

extension KalmanUpdater: Estimatable {
    public typealias Estimate = KalmanEstimate
}

extension KalmanUpdater: BayesUpdaterProtocol, KalmanUpdaterProtocol
    where ObservationModel: KalmanObservationModel
{
    /// Corrects the state error covariance based on innovation vector and Kalman update.
    ///
    /// Implements the following literature formulas:
    ///
    /// ```
    /// P'(k) = A * P(k-1) * At + Q
    /// K(k) = P'(k) * Ht * (H * P'(k) * Ht + R)^(-1)
    /// x(k) = x'(k) + K(k) * (z(k) - H * x'(k))
    /// ```
    ///
    /// - Parameters:
    ///   - prediction: The prediction used for prediction step.
    ///   - observation: The observation used for prediction step.
    ///   - control: The control used for prediction step.
    public func updated(
        prediction: Estimate,
        observation: Observation
    ) -> Estimate {
        let x = prediction.state
        let p = prediction.covariance

        let z = observation

        let r = self.observationNoise
        let i = self.identity(size: x.dimensions)

        // Calculate z prediction and H: z'(k), H
        let zP = self.observationModel.apply(state: x)
        let h = self.observationModel.jacobian(state: x)

        // Calculate transposed H:
        let hT = transpose(h)

        // Calculate innovation covariance matrix and its inverse:
        // S(k) = H * P'(k) * Ht + R
        let s = (h * p * hT) + r

        // Calculate inverse of S:
        // Si = S(k)^(-1)
        let sI = inv(s)

        // Update kalman gain:
        // K(k) = P'(k) * Ht * Si
        let k = p * hT * sI

        // Calculate innovation:
        // y(k) = z(k) - z'(k)
        let y = z - zP

        // Correct state using Kalman gain:
        // x(k) = x'(k) + K(k) * y(k)
        let xP = x + (k * y)

        // P(k) = (I - K(k) * H) * P'(k)
        let pP = (i - (k * h)) * p

        return KalmanEstimate(state: xP, covariance: pP)
    }
}

import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public typealias MultiModalKalmanPredictor = MultiModalBayesPredictor

public protocol KalmanPredictorProtocol: BayesPredictorProtocol {
    associatedtype MotionModel: KalmanMotionModel
}

public protocol ControllableKalmanPredictorProtocol: ControllableBayesPredictorProtocol {
    associatedtype MotionModel: ControllableKalmanMotionModel
}

public class KalmanPredictor<MotionModel> {
    /// Motion model (used for prediction).
    public var motionModel: MotionModel

    /// Process noise matrix (aka `Q`)
    ///
    /// This matrix implies the process noise covariance.
    ///
    /// Default: zero matrix.
    public var processNoise: Matrix<Double>

    public init(
        motionModel: MotionModel,
        processNoise: Matrix<Double>
    ) {
        self.motionModel = motionModel
        self.processNoise = processNoise
    }

    /// Predicts next state using current state and control and calculates probability estimate.
    ///
    /// Implements the following literature formulas:
    ///
    /// Uncontrollable:
    /// ```
    /// x'(k) = A * x(k-1)
    /// P'(k) = A * P(k-1) * At + Q
    /// ```
    ///
    /// Controllable:
    /// ```
    /// x'(k) = A * x(k-1) + B * u(k)
    /// P'(k) = A * P(k-1) * At + Q
    /// ```
    ///
    /// - Parameters:
    ///   - control: The control used for prediction step.
    private func predicted(
        estimate: Estimate,
        applyModel: (Vector<Double>) -> (Vector<Double>, Matrix<Double>)
    ) -> KalmanEstimate {
        let x = estimate.state
        let p = estimate.covariance
        
        let q = self.processNoise

        // Calculate x prediction and A:
        let (xP, a) = applyModel(x)

        let aT = transpose(a)

        // Calculate predicted probability estimate:
        // P'(k) = A * P(k-1) * At + Q
        let pP = (a * p * aT) + q

        return KalmanEstimate(state: xP, covariance: pP)
    }
}

extension KalmanPredictor: DimensionsValidatable {
    public func validate(for dimensions: DimensionsProtocol) throws {
        if let validatableModel = self.motionModel as? DimensionsValidatable {
            try validatableModel.validate(for: dimensions)
        }

        guard self.processNoise.columns == dimensions.state else {
            throw MatrixError.invalidColumnCount(
                message: "Expected \(dimensions.state) columns in `self.processNoise`, found \(self.processNoise.columns)"
            )
        }

        guard self.processNoise.rows == dimensions.state else {
            throw MatrixError.invalidRowCount(
                message: "Expected \(dimensions.state) rows in `self.processNoise`, found \(self.processNoise.rows)"
            )
        }
    }
}

extension KalmanPredictor: Statable {
    public typealias State = Vector<Double>
}

extension KalmanPredictor: Controllable {
    public typealias Control = Vector<Double>
}

extension KalmanPredictor: Estimatable {
    public typealias Estimate = KalmanEstimate
}

extension KalmanPredictor: BayesPredictorProtocol
    where MotionModel: KalmanMotionModel
{
    /// Predicts next state using current state and control and calculates probability estimate.
    ///
    /// Implements the following literature formulas:
    ///
    /// ```
    /// x'(k) = A * x(k-1)
    /// ```
    ///
    /// - Parameters:
    ///   - control: The control used for prediction step.
    public func predicted(estimate: Estimate) -> Estimate {
        return self.predicted(estimate: estimate) { (x: Vector<Double>) in
            let xP: Vector<Double> = self.motionModel.apply(state: x)
            let a: Matrix<Double> = self.motionModel.jacobian(state: x)
            return (xP, a)
        }
    }
}

extension KalmanPredictor: KalmanPredictorProtocol
    where MotionModel: KalmanMotionModel
{
    // Nothing
}

extension KalmanPredictor: ControllableBayesPredictorProtocol
    where MotionModel: ControllableKalmanMotionModel
{
    /// Predicts next state using current state and control and calculates probability estimate.
    ///
    /// Implements the following literature formulas:
    ///
    /// ```
    /// x'(k) = A * x(k-1) + B * u(k)
    /// ```
    ///
    /// - Parameters:
    ///   - control: The control used for prediction step.
    public func predicted(estimate: Estimate, control: Control) -> Estimate {
        let u = control
        return self.predicted(estimate: estimate) { x in
            let xP = self.motionModel.apply(state: x, control: u)
            let a = self.motionModel.jacobian(state: x, control: u)
            return (xP, a)
        }
    }
}

extension KalmanPredictor: ControllableKalmanPredictorProtocol
    where MotionModel: ControllableKalmanMotionModel
{
    // Nothing
}

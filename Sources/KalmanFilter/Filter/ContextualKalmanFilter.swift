import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

public struct Contextual<Context, Value> {
    let context: Context
    let value: Value
}

import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public class ContextualKalmanFilter<Context, MotionModel, ObservationModel>: EstimateReadWritable
    where Context: Hashable
{
    public typealias UpdaterClosure = (Context) -> KalmanUpdater<ObservationModel>

    public var estimate: Estimate

    public var predictor: KalmanPredictor<MotionModel>
    public var updaters: [Context: KalmanUpdater<ObservationModel>] = [:]
    public var closure: UpdaterClosure

    /// Creates a Kalman Filter with a given initial process state `estimate`.
    ///
    /// Unless a more appropriate initial `estimate` is available
    /// the following default provides reasonably good results:
    ///
    /// ```
    /// let state: Vector<Double> = .zero
    /// let covariance: Matrix<Double> = .init(
    ///     diagonal: <#variance#>,
    ///     size: <#state dimensions#>
    /// )
    /// let estimate = (state: state, covariance: covariance)
    /// let kalmanFilter = ContextualKalmanFilter(
    ///     estimate: estimate,
    ///     model: <#model#>
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - estimate: The initial process state estimate.
    ///   - predictor: The kalman filter's internal predictor.
    ///   - updater: The kalman filter's internal updater.
    public init(
        estimate: Estimate,
        predictor: KalmanPredictor<MotionModel>,
        closure: @escaping UpdaterClosure
    ) {
        self.estimate = estimate
        self.predictor = predictor
        self.updaters = [:]
        self.closure = closure
    }

    private func withUpdater<T>(
        for context: Context,
        _ closure: (KalmanUpdater<ObservationModel>) -> T
    ) -> T {
        let updater = self.updaters[context] ?? self.closure(context)
        let result = closure(updater)
        self.updaters[context] = updater
        return result
    }
}

extension ContextualKalmanFilter: DimensionsValidatable {
    public func validate(for dimensions: DimensionsProtocol) throws {
        try self.predictor.validate(for: dimensions)
        for updater in self.updaters.values {
            try updater.validate(for: dimensions)
        }
    }
}

extension ContextualKalmanFilter: Statable {
    public typealias State = Vector<Double>
}

extension ContextualKalmanFilter: Controllable {
    public typealias Control = Vector<Double>
}

extension ContextualKalmanFilter: Observable {
    public typealias Observation = Contextual<Context, Vector<Double>>
}

extension ContextualKalmanFilter: Estimatable {
    public typealias Estimate = (
        /// State vector (aka `x` in the literature)
        state: Vector<Double>,
        /// Estimate covariance matrix (aka `P`, or sometimes `Σ` in the literature)
        covariance: Matrix<Double>
    )
}

extension ContextualKalmanFilter: BayesPredictor
    where MotionModel: KalmanMotionModel
{
    public func predicted(
        estimate: Estimate
    ) -> Estimate {
        return self.predictor.predicted(
            estimate: estimate
        )
    }
}

extension ContextualKalmanFilter: ControllableBayesPredictor
    where MotionModel: ControllableKalmanMotionModel
{
    public func predicted(
        estimate: Estimate,
        control: Control
    ) -> Estimate {
        return self.predictor.predicted(
            estimate: estimate,
            control: control
        )
    }
}

extension ContextualKalmanFilter: BayesUpdater
    where ObservationModel: KalmanObservationModel
{
    public func updated(
        prediction: Estimate,
        observation: Observation
    ) -> Estimate {
        return self.withUpdater(for: observation.context) { updater in
            let estimate = updater.updated(
                prediction: prediction,
                observation: observation.value
            )
            return estimate
        }
    }
}

extension ContextualKalmanFilter: BayesFilter
    where MotionModel: KalmanMotionModel, ObservationModel: KalmanObservationModel
{
    public func filtered(
        estimate: Estimate,
        observation: Observation
    ) -> Estimate {
        let prediction = self.predicted(
            estimate: estimate
        )
        return self.updated(
            prediction: prediction,
            observation: observation
        )
    }
}

extension ContextualKalmanFilter: ControllableBayesFilter
    where MotionModel: ControllableKalmanMotionModel, ObservationModel: KalmanObservationModel
{
    public func filtered(
        estimate: Estimate,
        control: Control,
        observation: Observation
    ) -> Estimate {
        let prediction = self.predicted(
            estimate: estimate,
            control: control
        )
        return self.updated(
            prediction: prediction,
            observation: observation
        )
    }
}

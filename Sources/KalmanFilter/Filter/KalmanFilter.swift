import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public class KalmanFilter<Predictor, Updater>: EstimateReadWritable {
    public var estimate: Estimate

    public var predictor: Predictor
    public var updater: Updater

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
    /// let kalmanFilter = KalmanFilter(
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
        predictor: Predictor,
        updater: Updater
    ) {
        self.estimate = estimate
        self.predictor = predictor
        self.updater = updater
    }
}

extension KalmanFilter: DimensionsValidatable {
    public func validate(for dimensions: DimensionsProtocol) throws {
        if let predictor = self.predictor as? DimensionsValidatable {
            try predictor.validate(for: dimensions)
        }
        if let updater = self.updater as? DimensionsValidatable {
            try updater.validate(for: dimensions)
        }
    }
}

extension KalmanFilter: Statable {
    public typealias State = Vector<Double>
}

extension KalmanFilter: Controllable
    where Predictor: Controllable
{
    public typealias Control = Predictor.Control
}

extension KalmanFilter: Observable
    where Updater: Observable
{
    public typealias Observation = Updater.Observation
}

extension KalmanFilter: Estimatable {
    public typealias Estimate = (
        /// State vector (aka `x` in the literature)
        state: Vector<Double>,
        /// Estimate covariance matrix (aka `P`, or sometimes `Σ` in the literature)
        covariance: Matrix<Double>
    )
}

extension KalmanFilter: BayesPredictor
    where Predictor: BayesPredictor,
          Predictor.Estimate == Estimate
{
    public func predicted(estimate: Estimate) -> Estimate {
        return self.predictor.predicted(
            estimate: estimate
        )
    }
}

extension KalmanFilter: ControllableBayesPredictor
    where Predictor: ControllableBayesPredictor,
          Predictor.Estimate == Estimate
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

extension KalmanFilter: BayesUpdater
    where Updater: BayesUpdater,
          Updater.Estimate == Estimate
{
    public func updated(
        prediction: Estimate,
        observation: Observation
    ) -> Estimate {
        return self.updater.updated(
            prediction: prediction,
            observation: observation
        )
    }
}

extension KalmanFilter: BayesFilter
    where Predictor: BayesPredictor,
          Updater: BayesUpdater,
          Predictor.Estimate == Estimate,
          Updater.Estimate == Estimate
{
    public func filtered(
        estimate: Estimate,
        observation: Observation
    ) -> Estimate {
        let prediction = self.predictor.predicted(
            estimate: estimate
        )
        let estimate = self.updater.updated(
            prediction: prediction,
            observation: observation
        )
        return estimate
    }
}

extension KalmanFilter: ControllableBayesFilter
    where Predictor: ControllableBayesPredictor,
          Updater: BayesUpdater,
          Predictor.Estimate == Estimate,
          Updater.Estimate == Estimate
{
    public func filtered(
        estimate: Estimate,
        control: Control,
        observation: Observation
    ) -> Estimate {
        let prediction = self.predictor.predicted(
            estimate: estimate,
            control: control
        )
        let estimate = self.updater.updated(
            prediction: prediction,
            observation: observation
        )
        return estimate
    }
}

import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public class MultiModalKalmanUpdater<Context, Updater>
    where Context: Hashable
{
    public var updaters: [Context: Updater] = [:]
    public var closure: (Context) -> Updater

    public init(
        closure: @escaping (Context) -> Updater
    ) {
        self.closure = closure
    }

    private func withUpdater<T>(
        for context: Context,
        _ closure: (Updater) -> T
    ) -> T {
        let updater = self.updaters[context] ?? self.closure(context)
        let result = closure(updater)
        self.updaters[context] = updater
        return result
    }
}

extension MultiModalKalmanUpdater: DimensionsValidatable {
    public func validate(for dimensions: DimensionsProtocol) throws {
        for updater in self.updaters.values {
            guard let updater = updater as? DimensionsValidatable else {
                continue
            }
            try updater.validate(for: dimensions)
        }
    }
}

extension MultiModalKalmanUpdater: Statable {
    public typealias State = Vector<Double>
}

extension MultiModalKalmanUpdater: Observable {
    public typealias Observation = Contextual<Context, Vector<Double>>
}

extension MultiModalKalmanUpdater: Estimatable {
    public typealias Estimate = (
        /// State vector (aka `x` in the literature)
        state: Vector<Double>,
        /// Estimate covariance matrix (aka `P`, or sometimes `Σ` in the literature)
        covariance: Matrix<Double>
    )
}

extension MultiModalKalmanUpdater: BayesUpdater
    where Updater: BayesUpdater & Observable,
          Updater.Estimate == Estimate,
          Updater.Observation == Vector<Double>
{
    public func updated(
        prediction: Estimate,
        observation: Observation
    ) -> Estimate {
        return self.withUpdater(for: observation.context) { updater in
            return updater.updated(prediction: prediction, observation: observation.value)
        }
    }
}

extension MultiModalKalmanUpdater: KalmanUpdaterProtocol
    where Updater: KalmanUpdaterProtocol & Observable,
          Updater.Estimate == Estimate,
          Updater.Observation == Vector<Double>
{
    public typealias ObservationModel = Updater.ObservationModel
}

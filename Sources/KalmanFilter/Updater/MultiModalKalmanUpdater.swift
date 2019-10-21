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

extension MultiModalKalmanUpdater: Statable
    where Updater: Statable
{
    public typealias State = Updater.State
}

extension MultiModalKalmanUpdater: Observable
    where Updater: Observable
{
    public typealias Observation = Contextual<Context, Updater.Observation>
}

extension MultiModalKalmanUpdater: Estimatable
    where Updater: Estimatable
{
    public typealias Estimate = Updater.Estimate
}

extension MultiModalKalmanUpdater: BayesUpdater
    where Updater: BayesUpdater
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
    where Updater: KalmanUpdaterProtocol
{
    public typealias ObservationModel = Updater.ObservationModel
}

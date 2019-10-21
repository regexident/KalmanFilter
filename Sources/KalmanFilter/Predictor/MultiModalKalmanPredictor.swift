import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public class MultiModalKalmanPredictor<Model, Predictor>
    where Model: Hashable
{
    public var predictors: [Model: Predictor] = [:]
    public var closure: (Model) -> Predictor

    public init(
        closure: @escaping (Model) -> Predictor
    ) {
        self.closure = closure
    }

    private func withPredictor<T>(
        for model: Model,
        _ closure: (Predictor) -> T
    ) -> T {
        let predictor = self.predictors[model] ?? self.closure(model)
        let result = closure(predictor)
        self.predictors[model] = predictor
        return result
    }
}

extension MultiModalKalmanPredictor: DimensionsValidatable {
    public func validate(for dimensions: DimensionsProtocol) throws {
        for predictor in self.predictors.values {
            guard let predictor = predictor as? DimensionsValidatable else {
                continue
            }
            try predictor.validate(for: dimensions)
        }
    }
}

extension MultiModalKalmanPredictor: Statable
    where Predictor: Statable
{
    public typealias State = Predictor.State
}

extension MultiModalKalmanPredictor: Controllable
    where Predictor: Controllable
{
    public typealias Control = MultiModal<Model, Predictor.Control>
}

extension MultiModalKalmanPredictor: Estimatable
    where Predictor: Estimatable
{
    public typealias Estimate = Predictor.Estimate
}

extension MultiModalKalmanPredictor: ControllableBayesPredictor
    where Predictor: ControllableBayesPredictor
{
    public func predicted(estimate: Estimate, control: Control) -> Estimate {
        return self.withPredictor(for: control.model) { predictor in
            return predictor.predicted(estimate: estimate, control: control.value)
        }
    }
}

extension MultiModalKalmanPredictor: ControllableKalmanPredictorProtocol
    where Predictor: ControllableKalmanPredictorProtocol
{
    public typealias MotionModel = Predictor.MotionModel
}

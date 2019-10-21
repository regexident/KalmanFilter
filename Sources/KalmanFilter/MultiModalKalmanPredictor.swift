import Foundation

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

// swiftlint:disable all identifier_name

public class MultiModalKalmanPredictor<Context, Predictor>
    where Context: Hashable
{
    public var predictors: [Context: Predictor] = [:]
    public var closure: (Context) -> Predictor

    public init(
        closure: @escaping (Context) -> Predictor
    ) {
        self.closure = closure
    }

    private func withPredictor<T>(
        for context: Context,
        _ closure: (Predictor) -> T
    ) -> T {
        let predictor = self.predictors[context] ?? self.closure(context)
        let result = closure(predictor)
        self.predictors[context] = predictor
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
    public typealias Control = Contextual<Context, Predictor.Control>
}

extension MultiModalKalmanPredictor: Estimatable
    where Predictor: Estimatable
{
    public typealias Estimate = Predictor.Estimate
}

//extension MultiModalKalmanPredictor: BayesPredictor
//    where MotionModel: KalmanMotionModel
//{
//    public func predicted(estimate: Estimate) -> Estimate {
//        return self.withPredictor(for: observation.context) { updater in
//            return updater.updated(prediction: prediction, observation: observation.value)
//        }
//    }
//}
//
//extension MultiModalKalmanPredictor: KalmanPredictorProtocol
//    where MotionModel: KalmanMotionModel
//{
//    // Nothing
//}

extension MultiModalKalmanPredictor: ControllableBayesPredictor
    where Predictor: ControllableBayesPredictor
{
    public func predicted(estimate: Estimate, control: Control) -> Estimate {
        return self.withPredictor(for: control.context) { predictor in
            return predictor.predicted(estimate: estimate, control: control.value)
        }
    }
}

extension MultiModalKalmanPredictor: ControllableKalmanPredictorProtocol
    where Predictor: ControllableKalmanPredictorProtocol
{
    public typealias MotionModel = Predictor.MotionModel
}

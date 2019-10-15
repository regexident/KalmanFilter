import Foundation

import Surge

import BayesFilter

public struct Contextual<Context, Payload> {
    let context: Context
    let payload: Payload
}

public class ContextualKalmanFilter<Context: Hashable>: BayesFilter {
    public typealias Observation = KalmanFilter.Observation
    public typealias Control = Contextual<Context, KalmanFilter.Control>
    public typealias Estimate = KalmanFilter.Estimate
    
    public typealias Provider = (Context, Dimensions, Estimate) -> KalmanFilter
    
    public let dimensions: Dimensions
    public var estimate: Estimate
    private var provider: Provider
    
    private var kalmanFilters: [Context: KalmanFilter] = [:]
    
    public init(
        dimensions: Dimensions,
        estimate: Estimate,
        provider: @escaping Provider
    ) {
        assert(estimate.state.dimensions == dimensions.state)
        assert(estimate.covariance.columns == dimensions.state)
        assert(estimate.covariance.rows == dimensions.state)
        
        self.dimensions = dimensions
        self.estimate = estimate
        self.provider = provider
    }
    
    public func predict(
        control: Control
    ) -> Estimate {
        let kalmanFilter = self.kalmanFilter(
            for: control.context,
            dimensions: self.dimensions
        )
        return kalmanFilter.predict(control: control.payload)
    }
    
    public func update(
        prediction: Estimate,
        observation: Observation,
        control: Control
    ) -> Estimate {
        let kalmanFilter = self.kalmanFilter(
            for: control.context,
            dimensions: self.dimensions
        )
        let estimate = kalmanFilter.update(
            prediction: prediction,
            observation: observation,
            control: control.payload
        )
        self.estimate = estimate
        return estimate
    }
    
    private func kalmanFilter(
        for context: Context,
        dimensions: Dimensions
    ) -> KalmanFilter {
        let kalmanFilter = self.kalmanFilters[context] ?? self.provider(context, dimensions, self.estimate)
        assert(kalmanFilter.model.dimensions == dimensions)

        kalmanFilter.estimate = self.estimate

        self.kalmanFilters[context] = kalmanFilter
        return kalmanFilter
    }
}

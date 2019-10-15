import Foundation

import Surge
import BayesFilter

// swiftlint:disable all identifier_name

public class KalmanFilter: BayesFilter {
    public typealias Observation = Vector<Double>
    public typealias Control = Vector<Double>
    public typealias Estimate = (
        /// State vector (aka `x` in the literature)
        state: Vector<Double>,
        /// Estimate covariance matrix (aka `P`, or sometimes `Σ` in the literature)
        covariance: Matrix<Double>
    )
    
    public var estimate: Estimate
    
    public var model: Model
    
    private let identity: Matrix<Double>
    
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
    ///   - model: The process model.
    public init(estimate: Estimate, model: Model) {
        assert(estimate.state.dimensions == model.dimensions.state)
        assert(estimate.covariance.columns == model.dimensions.state)
        assert(estimate.covariance.rows == model.dimensions.state)
        
        self.estimate = estimate
        self.model = model
        
        let dimensions = model.dimensions
        
        self.identity = Matrix.identity(size: dimensions.state)
    }
    
    /// Predicts next state using current state and control and calculates probability estimate.
    ///
    /// Implements the following literature formulas:
    ///
    /// ```
    /// x'(k) = A * x(k-1) + B * u(k).
    /// P'(k) = A * P(k-1) * At + Q
    /// ```
    ///
    /// - Parameters:
    ///   - control: The control used for prediction step.
    public func predict(
        control: Control
    ) -> Estimate {
        let estimate = self.estimate
        let model = self.model
        
        let x = estimate.state
        let p = estimate.covariance
        let u = control
        
        let q = model.noiseModel.process
        
        // Calculate x prediction and A: x'(k), A
        let xP = model.motionModel.apply(state: x, control: u)
        let a = model.motionModel.jacobian(state: x, control: u)
        
        let aT = a.transposed()
        
        // Calculate predicted probability estimate:
        // P'(k) = A * P(k-1) * At + Q
        let pP = (a * p * aT) + q
        
        return Estimate(state: xP, covariance: pP)
    }
    
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
    public func update(
        prediction: Estimate,
        observation: Observation,
        control: Control
    ) -> Estimate {
        let model = self.model
        
        let x = prediction.state
        let p = prediction.covariance
        let z = observation
        
        let r = model.noiseModel.observation
        let i = self.identity
        
        // Calculate z prediction and H: z'(k), H
        let zP = model.observationModel.apply(state: x)
        let h = model.observationModel.jacobian(state: x)
        
        // Calculate transposed H:
        let hT = h.transposed()
        
        // Calculate innovation covariance matrix and its inverse:
        // S(k) = H * P'(k) * Ht + R
        let s = (h * p * hT) + r
        
        // Calculate inverse of S:
        // Si = S(k)^(-1)
        let sI = s.inversed()
        
        // Update kalman gain:
        // K(k) = P'(k) * Ht * Si
        let k = p * hT * sI
        
        // Calculate innovation:
        // y(k) = z(k) - z'(k)
        let y = z - zP
        
        // Correct state using Kalman gain:
        // x(k) = x'(k) + K(k) * y(k)
        let state = x + (k * y)
        
        // P(k) = (I - K(k) * H) * P'(k)
        let covariance = (i - (k * h)) * p
        
        let estimate = Estimate(
            state: state,
            covariance: covariance
        )
        
        self.estimate = estimate
        
        return estimate
    }
}

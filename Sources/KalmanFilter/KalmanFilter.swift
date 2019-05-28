import Foundation
import Surge

// swiftlint:disable all identifier_name

public class KalmanFilter {
    public var estimate: Estimate
    public var model: Model
    
    private let identity: Matrix<Double>
    
    public init(estimate: Estimate, model: Model) {
        self.estimate = estimate
        self.model = model
        
        let dimesions = model.dimensions
        
        self.identity = Matrix(identity: dimesions.state)
    }
    
    public func filter(output z: Vector<Double>, input u: Vector<Double>) -> Estimate {
        let (prediction: x, probability: p) = self.predict(input: u)
        return self.update(prediction: x, probability: p, output: z, input: u)
    }

    /// Predicts next state using current state and input and calculates probability estimate.
    /// x'(k) = A * x(k-1) + B * u(k).
    /// P'(k) = A * P(k-1) * At + Q
    public func predict(input u: Vector<Double>) -> (prediction: Vector<Double>, probability: Matrix<Double>) {
        let estimate = self.estimate
        let model = self.model
        
        let x = estimate.state
        let p = estimate.covariance
        
        let q = model.noiseModel.process
        
        // Calculate x prediction and A: x'(k), A
        let xP = model.motionModel.apply(state: x, input: u)
        let a = model.motionModel.jacobian(state: x, input: u)
        
        let aT = a.transposed()
        
        // Calculate predicted probability estimate:
        // P'(k) = A * P(k-1) * At + Q
        let pP = (a * p * aT) + q
        
        return (prediction: xP, probability: pP)
    }

    /// Corrects the state error covariance based on innovation vector and Kalman update.
    /// P'(k) = A * P(k-1) * At + Q
    /// K(k) = P'(k) * Ht * (H * P'(k) * Ht + R)^(-1)
    /// x(k) = x'(k) + K(k) * (z(k) - H * x'(k))
    public func update(
        prediction x: Vector<Double>,
        probability p: Matrix<Double>,
        output z: Vector<Double>,
        input u: Vector<Double>
    ) -> Estimate {
        let model = self.model
                
        let r = model.noiseModel.output
        let i = self.identity
        
        // Calculate z prediction and H: z'(k), H
        let zP = model.observationModel.apply(state: x)
        let h = model.observationModel.jacobian(state: x)
        
        let hT = h.transposed()
        
        // Calculate innovation covariance matrix and its inverse:
        // S(k) = H * P'(k) * Ht + R
        let s = (h * p * hT) + r
        
        let sI = s.inversed()
        
        // Update kalman gain:
        // K(k) = P'(k) * Ht * S(k)^(-1)
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

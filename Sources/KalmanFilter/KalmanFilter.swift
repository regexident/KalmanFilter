import Foundation
import Surge

// swiftlint:disable all identifier_name

public class KalmanFilter {
    public struct Shape {
        // state
        public let n: Int
        // input
        public let p: Int
        // output
        public let m: Int
    }
    
    public let shape: Shape
    
    public private(set) var x: Vector<Double>
    public private(set) var p: Matrix<Double>
    
    public let motion: MotionModel
    public let observation: ObservationModel
    
    public let q: Matrix<Double>
    public let r: Matrix<Double>
    
    private let i: Matrix<Double>
    private let u: Vector<Double>
    
    public init(_ configuration: Configuration) {
        self.shape = Shape(
            n: configuration.dimensions.state,
            p: configuration.dimensions.input,
            m: configuration.dimensions.output
        )
        
        self.x = configuration.state
        
        self.motion = configuration.motionModel
        self.observation = configuration.observationModel
        
        self.p = configuration.estimateCovariance
        self.q = configuration.processNoiseCovariance
        self.r = configuration.outputNoiseCovariance
        
        self.i = configuration.identity
        self.u = configuration.zeroInput
    }
    
    public func filter(output z: Vector<Double>, input u: Vector<Double>? = nil) -> Vector<Double> {
        let u = u ?? self.u
        
        func validateInput(output z: Vector<Double>, input u: Vector<Double>) {
            assert(
                z.rows == self.shape.m,
                "Expected \(self.shape.m)-dimensional vector"
            )
            assert(
                u.rows == self.shape.p,
                "Expected \(self.shape.p)-dimensional vector"
            )
        }
        validateInput(output: z, input: u)
        
        let (prediction: x, probability: p) = self.predict(input: u)
        return self.update(prediction: x, probability: p, output: z, input: u)
    }

    /// Predicts next state using current state and input and calculates probability estimate.
    /// x'(k) = A * x(k-1) + B * u(k).
    /// P'(k) = A * P(k-1) * At + Q
    public func predict(input u: Vector<Double>? = nil) -> (prediction: Vector<Double>, probability: Matrix<Double>) {
        let u = u ?? self.u
        
        let x = self.x
        
        func validateInput(input u: Vector<Double>) {
            assert(
                u.rows == self.shape.p,
                "Expected \(self.shape.p)-dimensional vector"
            )
        }
        validateInput(input: u)
        
        let (x: xP, a: a) = self.motion.apply(state: x, input: u)
        
        let aT = a.transposed()
        
        // Calculate predicted probability estimate:
        // P'(k) = A * P(k-1) * At + Q
        let p = (a * self.p * aT) + self.q
        
//        print("x:", x.scalars)
//        print("xP:", xP.scalars)
        
        return (prediction: xP, probability: p)
    }

    /// Corrects the state error covariance based on innovation vector and Kalman update.
    /// P'(k) = A * P(k-1) * At + Q
    /// K(k) = P'(k) * Ht * (H * P'(k) * Ht + R)^(-1)
    /// x(k) = x'(k) + K(k) * (z(k) - H * x'(k))
    public func update(
        prediction x: Vector<Double>,
        probability p: Matrix<Double>,
        output z: Vector<Double>,
        input u: Vector<Double>? = nil
    ) -> Vector<Double> {
        let u = u ?? self.u
        
        func validateInput(
            prediction x: Vector<Double>,
            input u: Vector<Double>,
            probability p: Matrix<Double>,
            output z: Vector<Double>
        ) {
            assert(
                x.rows == self.shape.n,
                "Expected \(self.shape.n)-dimensional vector"
            )
            assert(
                u.rows == self.shape.p,
                "Expected \(self.shape.p)-dimensional vector"
            )
            assert(
                p.rows == self.shape.n,
                "Expected \(self.shape.n)-dimensional vector"
            )
            assert(
                p.columns == self.shape.n,
                "Expected \(self.shape.n)-dimensional vector"
            )
            assert(
                z.rows == self.shape.m,
                "Expected \(self.shape.m)-dimensional vector"
            )
        }
        validateInput(prediction: x, input: u, probability: p, output: z)
        
        // Calculate z prediction and H: z'(k), H
        let (z: zP, h: h) = self.observation.apply(state: x)
        
//        print(h)
//        print("z:", z.scalars)
//        print("zP:", zP.scalars)
//        print()

        let hT = h.transposed()
        
        let (r, i) = (self.r, self.i)
        
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
        self.x = x + (k * y)
        
        // P(k) = (I - K(k) * H) * P'(k)
        self.p = (i - (k * h)) * p
        
        return self.x
    }
}

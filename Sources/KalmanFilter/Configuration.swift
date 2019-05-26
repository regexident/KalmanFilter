import Foundation

//typealias LinearConfiguration = Configuration<S: ProjectionFunction, TransitionFunction>

public class Configuration {
    public let dimensions: Dimensions
    
    /// State vector (aka `x`)
    ///
    /// This vector implies the current state.
    ///
    /// Default: zero vector.
    public var state: Vector<Double> {
        didSet {
            let m = self.dimensions.state
            assert(
                self.state.rows == m,
                "Expected \(m)-dimensional vector"
            )
        }
    }
    
    /// The kalman filters's motion model (used for prediction).
    public var motionModel: MotionModel {
        didSet {
//            let dummyState: Vector<Double> = .init(rows: self.dimensions.state)
//            let dummyInput: Vector<Double> = .init(rows: self.dimensions.input)
//            
//            let (x: x, a: a) = self.motionModel.apply(state: dummyState, input: dummyInput)
//            
//            let (m, n) = (self.dimensions.state, self.dimensions.state)
//            assert(
//                x.rows == m,
//                "Expected model to produce \(m)-dimensional vector"
//            )
//            assert(
//                (a.rows == m) && (a.columns == n),
//                "Expected model to produce \(m) × \(n) matrix"
//            )
        }
    }
    
    /// The kalman filters's observation model (used for correction).
    public var observationModel: ObservationModel {
        didSet {
//            let dummyState: Vector<Double> = .init(rows: self.dimensions.state)
//
//            let (z: z, h: h) = self.observationModel.apply(state: dummyState)
//
//            let (m, n) = (self.dimensions.output, self.dimensions.state)
//            assert(
//                z.rows == m,
//                "Expected model to produce \(m)-dimensional vector"
//            )
//            assert(
//                (h.rows == m) && (h.columns == n),
//                "Expected model to produce \(m) × \(n) matrix"
//            )
        }
    }
    
    /// Estimate covariance matrix (aka `P`, or sometimes `Σ`)
    ///
    /// This matrix implies the estimation noise covariance.
    ///
    /// Default: identity matrix.
    ///
    /// Initialization: unless more detailed domain-specific knowledge is available
    /// a good starting-point is: `P = variance * I` where `I` is the identity matrix.
    public var estimateCovariance: Matrix<Double> {
        didSet {
            let p = self.estimateCovariance
            
            let (m, n) = (self.dimensions.state, self.dimensions.state)
            assert(
                (p.rows == m) && (p.columns == n),
                "Expected \(m) × \(n) matrix"
            )
        }
    }
    
    /// Process noise matrix (aka `Q`)
    ///
    /// This matrix implies the process noise covariance.
    ///
    /// Default: zero matrix.
    public var processNoiseCovariance: Matrix<Double> {
        didSet {
            let q = self.processNoiseCovariance
            
            let (m, n) = (self.dimensions.state, self.dimensions.state)
            assert(
                (q.rows == m) && (q.columns == n),
                "Expected \(m) × \(n) matrix"
            )
        }
    }
    
    /// Output noise matrix (aka `R`)
    ///
    /// This matrix implies the output error covariance,
    /// based on the amount of sensor noise.
    ///
    /// Default: zero matrix.
    public var outputNoiseCovariance: Matrix<Double> {
        didSet {
            let r = self.outputNoiseCovariance
            
            let (m, n) = (self.dimensions.output, self.dimensions.output)
            assert(
                (r.rows == m) && (r.columns == n),
                "Expected \(m) × \(n) matrix"
            )
        }
    }
    
    /// Identity matrix (aka `I`)
    ///
    /// Used for internal calculation efficiency.
    ///
    /// Value: n × n identity matrix.
    internal let identity: Matrix<Double>
    
    /// Zero input vector (aka `u`)
    ///
    /// Used for internal calculation efficiency.
    ///
    /// Value: n-element zero vector.
    internal let zeroInput: Vector<Double>
    
    public init(dimensions: Dimensions, builder: (Configuration) -> ()) {
        assert(dimensions.state >= 1)
        assert(dimensions.input >= 1)
        assert(dimensions.output >= 1)
        
        self.dimensions = dimensions
        
        self.state = Vector(rows: dimensions.state)
        
        self.motionModel = StaticMatrixMotionModel(
            a: Matrix(
                diagonal: 1.0,
                size: dimensions.state
            ),
            b: Matrix(
                diagonal: 1.0,
                size: dimensions.state
            )
        )
        self.observationModel = StaticMatrixObservationModel(
            h: Matrix(
                diagonal: 1.0,
                size: dimensions.state
            )
        )
        
        self.estimateCovariance = Matrix(identity: dimensions.state)
        self.processNoiseCovariance = Matrix(identity: dimensions.state)
        self.outputNoiseCovariance = Matrix(identity: dimensions.output)
        
        self.identity = Matrix(identity: dimensions.state)
        self.zeroInput = Vector(rows: dimensions.input, repeatedValue: 0.0)
        
        builder(self)
    }
}

extension Configuration: CustomStringConvertible {
    public var description: String {
        var string = ""
        
        string += "dimensions: \(self.dimensions)\n"
        string += "state:\n\(self.state)\n"
        
        string += "motionModel:\n\(self.motionModel)\n"
        string += "observationModel:\n\(self.observationModel)\n"
        
        string += "estimateCovariance:\n\(self.estimateCovariance)\n"
        string += "processNoiseCovariance:\n\(self.processNoiseCovariance)\n"
        string += "outputNoiseCovariance:\n\(self.outputNoiseCovariance)\n"
        
        return string
    }
}

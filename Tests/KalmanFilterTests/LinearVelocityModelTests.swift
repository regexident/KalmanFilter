import XCTest

@testable import KalmanFilter

final class LinearVelocityModelTests: XCTestCase {
    let velocity: (x: Double, y: Double) = (x: 20.0, y: 10.0) // in m/s
    
    let configuration: Configuration = {
        // Modelled after:
        // https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb
        
        let dimensions = Dimensions(
            state: 4, // [position x, position y, velocity x, velocity y]
            input: 2, // [velocity x, velocity y]
            output: 2 // [position x, position y]
        )
        
        let time = 0.1 // time delta
        
        let initialState: Vector = [
            0.0, // Position X
            0.0, // Position Y
            0.0, // Velocity X
            0.0, // Velocity Y
        ]
        
        let motionModel = StaticMatrixMotionModel(
            a: [
                [1.0, 0.0, time, 0.0],
                [0.0, 1.0, 0.0, time],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ], b: [
                [0.0, 0.0],
                [0.0, 0.0],
                [time, 0.0],
                [0.0, time],
            ]
        )
        
        let observationModel = StaticMatrixObservationModel(
            h: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        
        let processNoiseCovariance: Matrix<Double> = {
            let acc = 1.0 // max expected acceleration in m/sec^2
            let qs: Matrix = [
                [acc * 0.5 * time * time], // translation in m (double-integrated acceleration)
                [acc * 0.5 * time * time], // translation in m (double-integrated acceleration)
                [acc * time], // velocity in m/s (integrated acceleration)
                [acc * time], // velocity in m/s (integrated acceleration)
            ]
            return (qs * qs.transposed()).squared()
        }()
        
        let outputNoiseVariance = 2.0
        let outputNoiseCovariance = Matrix(diagonal: outputNoiseVariance, size: dimensions.output).squared()
        
        let estimateCovariance: Matrix<Double> = Matrix(diagonal: 1.0, size: dimensions.state)
        
        return Configuration(dimensions: dimensions) { config in
            config.state = initialState
            config.motionModel = motionModel
            config.observationModel = observationModel
            config.estimateCovariance = estimateCovariance
            config.processNoiseCovariance = processNoiseCovariance
            config.outputNoiseCovariance = outputNoiseCovariance
        }
    }()
    
    func testConstantModel() {
        let configuration = self.configuration
        
        let sampleCount = 200
        let inputs: [Vector<Double>] = (0..<sampleCount).map { i in
            let x = self.velocity.x
            let y = self.velocity.y
            return Vector(column: [x, y])
        }
        
        let states = self.makeSignal(
            initial: configuration.state,
            inputs: inputs,
            model: configuration.motionModel,
            processNoise: configuration.processNoiseCovariance
        )
        
        let outputs: [Vector<Double>] = states.map { state in
            let output: Vector<Double> = configuration.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: configuration.dimensions.output)
            let noise: Vector<Double> = configuration.outputNoiseCovariance * standardNoise
            return output + noise
        }
        
        let kalmanFilter = KalmanFilter(configuration)

        let filteredStates: [Vector<Double>] = Swift.zip(inputs, outputs).map { input, output in
            let filteredState = kalmanFilter.filter(output: output, input: input)
            return filteredState
        }
        
//        self.printSheet(unfiltered: states, filtered: filteredStates, measured: outputs)
        
        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }

        XCTAssertLessThan(similarity, 0.5)
    }
    
    func testVariableModel() {
        let configuration = self.configuration
        
        let sampleCount = 200
        let inputs: [Vector<Double>] = (0..<sampleCount).map { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let x = self.velocity.x * sine
            let y = self.velocity.y * cosine
            return Vector(column: [x, y])
        }
        
        let states = self.makeSignal(
            initial: configuration.state,
            inputs: inputs,
            model: configuration.motionModel,
            processNoise: configuration.processNoiseCovariance
        )
        
        let outputs: [Vector<Double>] = states.map { state in
            let output: Vector<Double> = configuration.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: configuration.dimensions.output)
            let noise: Vector<Double> = configuration.outputNoiseCovariance * standardNoise
            return output + noise
        }
        
        let kalmanFilter = KalmanFilter(configuration)

        let filteredStates: [Vector<Double>] = Swift.zip(inputs, outputs).map { input, output in
            let filteredState = kalmanFilter.filter(output: output, input: input)
            return filteredState
        }
        
//        self.printSheet(unfiltered: states, filtered: filteredStates, measured: outputs)

        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }

        XCTAssertLessThan(similarity, 0.5)
    }
    
    private func printSheet(
        unfiltered: [Vector<Double>],
        filtered: [Vector<Double>],
        measured: [Vector<Double>]
    ) {
        self.printSheet(unfiltered2D: unfiltered, filtered2D: filtered, measured2D: measured)
        
        XCTFail("Printing found in test")
    }
    
    static var allTests = [
        ("testConstantModel", testConstantModel),
        ("testVariableModel", testVariableModel),
    ]
}

import XCTest

@testable import KalmanFilter

private func deg2rad(_ degree: Double) -> Double {
    return (degree / 180.0) * .pi
}

final class NonlinearYawAccelerationModelTests: XCTestCase {
    let acceleration: Double = 1.0 // in m/s^2
    let yaw: Double = deg2rad(30.0) // yaw rate in radians/s^2
    
    let configuration: Configuration = {
        // Modelled after:
        // https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRA.ipynb
        
        let dimensions = Dimensions(
            state: 6, // [position x, position y, h, velocity, yaw rate, acceleration]
            input: 2, // [yaw rate, acceleration]
            output: 2 // [position x, position y]
        )
        
        let time = 0.1 // time delta
        
        let initialState: Vector = [
            0.0, // Position X
            0.0, // Position Y
            0.0, // Heading
            0.0, // Velocity
            0.0, // Yaw Rate
            0.0, // Acceleration
        ]
        
        let motionModel = FunctionMotionModel { state, input in
            let (x, y, h, v) = (state[0], state[1], state[2], state[3]) // pos-x, pos-y, heading, velocity
            let (w, a) = (input[0], input[1]) // yaw-rate, acceleration
            let t = time // delta time
            return [
                x + (v / w) * (sin(h + w * t) - sin(h)),
                y + (v / w) * (-cos(h + w * t) + cos(h)),
                h + w * t,
                v + a * t,
                w,
                a,
            ]
        }
        
        let observationModel = StaticMatrixObservationModel(
            h: [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        
        let processNoiseCovariance: Matrix<Double> = {
            let acc = 1.0 // max expected acceleration in m/sec^2
            let yaw = 0.1 // max expected yaw in radians/s^2
            let qs: Matrix = [
                [acc * (0.5 * time * time)], // translation in m (double-integrated acceleration)
                [acc * (0.5 * time * time)], // translation in m (double-integrated acceleration)
                [yaw * time], // heading in radians/s (integrated of yaw)
                [acc * time], // velocity in m/s (integrated acceleration)
                [yaw * 1.0], // yaw in radians/s^2
                [acc * 1.0], // acceleration in m/s^2
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
            let yaw = self.yaw
            let acceleration = self.acceleration
            return Vector(column: [yaw, acceleration])
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

        XCTAssertLessThan(similarity, 5.0)
    }
    
    func testVariableModel() {
        let configuration = self.configuration
        
        let sampleCount = 200
        let inputs: [Vector<Double>] = (0..<sampleCount).map { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let yaw = self.yaw * sine
            let acceleration = self.acceleration * cosine
            return Vector(column: [yaw, acceleration])
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

        XCTAssertLessThan(similarity, 5.0)
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

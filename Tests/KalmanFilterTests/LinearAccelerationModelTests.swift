import XCTest

@testable import KalmanFilter

final class LinearAccelerationModelTests: XCTestCase {
    let acceleration: (x: Double, y: Double) = (x: 2.0, y: 1.0) // in m/s^2
    
    let model: Model = {
        // Modelled after:
        // https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA.ipynb
        
        let dimensions = Dimensions(
            state: 6, // [position x, position y, velocity x, velocity y, acceleration x, acceleration y]
            control: 2, // [acceleration x, acceleration y]
            output: 2 // [position x, position y]
        )
        
        let time = 0.1 // time delta
        
        let motionModel = LinearMotionModel(
            state: [
                [1.0, 0.0, time, 0.0, 0.5 * time * time, 0.0],
                [0.0, 1.0, 0.0, time, 0.0, 0.5 * time * time],
                [0.0, 0.0, 1.0, 0.0, time, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, time],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            control: [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.5 * time * time, 0.0],
                [0.0, 0.5 * time * time],
                [time, 0.0],
                [0.0, time],
            ]
        )
        
        let observationModel = LinearObservationModel(
            state: [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        
        let noiseModel = NoiseModel(
            process: {
                let acceleration = 1.0 // max expected acceleration in m/sec^2
                let qs: Matrix = [
                    [acceleration * 0.5 * time * time], // translation in m (double-integrated acceleration)
                    [acceleration * 0.5 * time * time], // translation in m (double-integrated acceleration)
                    [acceleration * time], // velocity in m/s (integrated acceleration)
                    [acceleration * time], // velocity in m/s (integrated acceleration)
                    [acceleration * 1.0], // acceleration in m/s^2
                    [acceleration * 1.0], // acceleration in m/s^2
                ]
                return (qs * qs.transposed()).squared()
            }(),
            output: Matrix(
                diagonal: 2.0,
                size: dimensions.output
            ).squared()
        )
        
        return Model(
            dimensions: dimensions,
            motionModel: motionModel,
            observationModel: observationModel,
            noiseModel: noiseModel
        )
    }()
    
    let initialState: Vector<Double> = [
        0.0, // Position X
        0.0, // Position Y
        0.0, // Velocity X
        0.0, // Velocity Y
        0.0, // Acceleration X
        0.0, // Acceleration Y
    ]
    
    func estimate() -> Estimate {
        return Estimate(
            state: self.initialState,
            covariance: Matrix(diagonal: 1.0, size: 6)
        )
    }
    
    func filter(control: (Int) -> Vector<Double>) -> Double {
        let model = self.model
        let estimate = self.estimate()
        let initialState = self.initialState
        
        let sampleCount = 200
        let controls: [Vector<Double>] = (0..<sampleCount).map(control)
        
        let states = self.makeSignal(
            initial: initialState,
            controls: controls,
            model: model.motionModel,
            processNoise: model.noiseModel.process
        )
        
        let outputs: [Vector<Double>] = states.map { state in
            let output: Vector<Double> = model.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: model.dimensions.output)
            let noise: Vector<Double> = model.noiseModel.output * standardNoise
            return output + noise
        }
        
        let kalmanFilter = KalmanFilter(estimate: estimate, model: model)
        
        let filteredStates: [Vector<Double>] = Swift.zip(controls, outputs).map { control, output in
            return kalmanFilter.filter(output: output, control: control).state
        }
        
//        self.printSheet(unfiltered: states, filtered: filteredStates, measured: outputs)
        
        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }
        
        return similarity
    }
    
    func testConstantModel() {
        let similarity = self.filter { i in
            let x = self.acceleration.x
            let y = self.acceleration.y
            return Vector(column: [x, y])
        }
        
        XCTAssertLessThan(similarity, 5.0)
    }
    
    func testVariableModel() {
        let similarity = self.filter { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let x = self.acceleration.x * sine
            let y = self.acceleration.y * cosine
            return Vector(column: [x, y])
        }
        
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

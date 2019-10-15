import XCTest

import Surge

@testable import KalmanFilter

final class LinearVelocityModelTests: XCTestCase {
    let velocity: (x: Double, y: Double) = (x: 20.0, y: 10.0) // in m/s
    
    let model: Model = {
        // Modelled after:
        // https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb
        
        let dimensions = Dimensions(
            state: 4, // [position x, position y, velocity x, velocity y]
            control: 2, // [velocity x, velocity y]
            observation: 2 // [position x, position y]
        )
        
        let time = 0.1 // time delta
        
        let motionModel = LinearMotionModel(
            state: [
                [1.0, 0.0, time, 0.0],
                [0.0, 1.0, 0.0, time],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            control: [
                [0.0, 0.0],
                [0.0, 0.0],
                [time, 0.0],
                [0.0, time],
            ]
        )
        
        let observationModel = LinearObservationModel(
            state: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        
        let noiseModel = NoiseModel(
            process: {
                let accel = 1.0 // max expected acceleration in m/sec^2
                let qs: Matrix = [
                    [accel * 0.5 * time * time], // translation in m (double-integrated acceleration)
                    [accel * 0.5 * time * time], // translation in m (double-integrated acceleration)
                    [accel * time], // velocity in m/s (integrated acceleration)
                    [accel * time], // velocity in m/s (integrated acceleration)
                ]
                return (qs * qs.transposed()).squared()
            }(),
            observation: Matrix.diagonal(
                rows: dimensions.observation,
                columns: dimensions.observation,
                repeatedValue: 2.0
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
    ]
    
    func estimate() -> (state: Vector<Double>, covariance: Matrix<Double>) {
        return (
            state: self.initialState,
            covariance: Matrix.diagonal(
                rows: 4,
                columns: 4,
                repeatedValue: 1.0
            )
        )
    }
    
    func filter(control: (Int) -> Vector<Double>) -> Double {
        let model = self.model
        let estimate = self.estimate()
        
        let sampleCount = 200
        let controls: [Vector<Double>] = (0..<sampleCount).map(control)
        
        let states = self.makeSignal(
            initial: self.initialState,
            controls: controls,
            model: model.motionModel,
            processNoise: model.noiseModel.process
        )
        
        let observations: [Vector<Double>] = states.map { state in
            let observation: Vector<Double> = model.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: model.dimensions.observation)
            let noise: Vector<Double> = model.noiseModel.observation * standardNoise
            return observation + noise
        }
        
        var kalmanFilter = KalmanFilter(estimate: estimate, model: model)
        
        let filteredStates: [Vector<Double>] = Swift.zip(controls, observations).map { argument in
            let (control, observation) = argument
            return kalmanFilter.filter(observation: observation, control: control).state
        }
        
//        self.printSheetAndFail(
//            trueStates: states,
//            estimatedStates: filteredStates,
//            observations: observations
//        )
        
        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }
        
        return similarity
    }
    
    func testConstantModel() {
        let similarity = self.filter { i in
            let x = self.velocity.x
            let y = self.velocity.y
            return Vector([x, y])
        }
        
        XCTAssertLessThan(similarity, 0.5)
    }
    
    func testVariableModel() {
        let similarity = self.filter { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let x = self.velocity.x * sine
            let y = self.velocity.y * cosine
            return Vector([x, y])
        }
        
        XCTAssertLessThan(similarity, 0.5)
    }
    
    private func printSheetAndFail(
        trueStates: [Vector<Double>],
        estimatedStates: [Vector<Double>],
        observations: [Vector<Double>]
    ) {
        self.printSheet(trueStates: trueStates, estimatedStates: estimatedStates, observations: observations)
        
        XCTFail("Printing found in test")
    }
    
    static var allTests = [
        ("testConstantModel", testConstantModel),
        ("testVariableModel", testVariableModel),
    ]
}

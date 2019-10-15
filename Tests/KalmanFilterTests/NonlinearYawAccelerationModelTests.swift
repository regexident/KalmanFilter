import XCTest

import Surge

@testable import KalmanFilter

private func deg2rad(_ degree: Double) -> Double {
    return (degree / 180.0) * .pi
}

final class NonlinearYawAccelerationModelTests: XCTestCase {
    let acceleration: Double = 1.0 // in m/s^2
    let yaw: Double = deg2rad(30.0) // yaw rate in radians/s^2
    
    let model: Model = {
        // Modelled after:
        // https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRA.ipynb
        
        let dimensions = Dimensions(
            state: 6, // [position x, position y, h, velocity, yaw rate, acceleration]
            control: 2, // [yaw rate, acceleration]
            observation: 2 // [position x, position y]
        )
        
        let time = 0.1 // time delta
        
        let motionModel = NonlinearMotionModel(dimensions: dimensions) { state, control in
            let (x, y, h, v) = (state[0], state[1], state[2], state[3]) // pos-x, pos-y, heading, velocity
            let (w, a) = (control[0], control[1]) // yaw-rate, acceleration
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
        
        let observationModel = LinearObservationModel(
            state: [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        
        let noiseModel = NoiseModel(
            process: {
                let acceleration = 1.0 // max expected acceleration in m/sec^2
                let yaw = 0.1 // max expected yaw in radians/s^2
                let qs: Matrix = [
                    [acceleration * (0.5 * time * time)], // translation in m (double-integrated acceleration)
                    [acceleration * (0.5 * time * time)], // translation in m (double-integrated acceleration)
                    [yaw * time], // heading in radians/s (integrated of yaw)
                    [acceleration * time], // velocity in m/s (integrated acceleration)
                    [yaw * 1.0], // yaw in radians/s^2
                    [acceleration * 1.0], // acceleration in m/s^2
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
        0.0, // Heading
        0.0, // Velocity
        0.0, // Yaw Rate
        0.0, // Acceleration
    ]
    
    func estimate() -> (state: Vector<Double>, covariance: Matrix<Double>) {
        return (
            state: self.initialState,
            covariance: Matrix.diagonal(
                rows: 6,
                columns: 6,
                repeatedValue: 1.0
            )
        )
    }
    
    func filter(control: (Int) -> Vector<Double>) -> Double {
        let model = self.model
        let estimate = self.estimate()
        let initialState = self.initialState
        
        let sampleCount = 200
        let controls: [Vector<Double>] = (0..<sampleCount).map { i in
            let yaw = self.yaw
            let acceleration = self.acceleration
            return Vector([yaw, acceleration])
        }
        
        let states = self.makeSignal(
            initial: initialState,
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
            let yaw = self.yaw
            let acceleration = self.acceleration
            return Vector([yaw, acceleration])
        }
        
        XCTAssertLessThan(similarity, 2.0)
    }
    
    func testVariableModel() {
        let similarity = self.filter { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let yaw = self.yaw * sine
            let acceleration = self.acceleration * cosine
            return Vector([yaw, acceleration])
        }
        
        XCTAssertLessThan(similarity, 5.0)
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

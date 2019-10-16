import XCTest

import Surge

@testable import KalmanFilter

final class LocalizationTests: XCTestCase {
    let model: Model = {
        let dimensions = Dimensions(
            state: 4, // [target position x, target position y, self position x, self position y]
            control: 2, // [self position x, self position y]
            observation: 1 // [distance]
        )
        
        let motionModel = LinearMotionModel(
            state: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            control: [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        
        let observationModel = NonlinearObservationModel(dimensions: dimensions) { state in
            let targetPosition: Vector<Double> = [state[0], state[1]]
            let selfPosition: Vector<Double> = [state[2], state[3]]
            let delta = targetPosition - selfPosition
            let dist = delta.magnitude()
            return [dist]
        }
        
        let noiseModel = NoiseModel(
            process: Matrix.diagonal(
                rows: dimensions.state,
                columns: dimensions.state,
                repeatedValue: 0.0
            ),
            observation: {
                return Matrix.diagonal(
                    rows: dimensions.observation,
                    columns: dimensions.observation,
                    repeatedValue: 0.5
                ).squared()
            }()
        )
        
        return Model(
            dimensions: dimensions,
            motion: motionModel,
            observation: observationModel,
            noise: noiseModel
        )
    }()
    
    func testModel() {
        let model = self.model
        
        let initialState: Vector<Double> = [
            5.0, // target position x
            10.0, // target position y
            0.0, // self position x
            0.0, // self position y
        ]
        
        let estimate: (state: Vector<Double>, covariance: Matrix<Double>) = (
            state: initialState,
            covariance: Matrix.diagonal(
                rows: model.dimensions.state,
                columns: model.dimensions.state,
                repeatedValue: 1.0
            )
        )
        
        let interval = 0.05
        let sampleCount = 500
        let controls: [Vector<Double>] = (0..<sampleCount).map { i in
            let a = Double(i) * interval
            let r = 7.5
            let x = r * sin(a)
            let y = r * cos(a)
            return Vector([x, y])
        }
        
        let states = self.makeSignal(
            initial: estimate.state,
            controls: controls,
            model: model.motion,
            processNoise: model.noise.process
        )
        
        let observations: [Vector<Double>] = states.map { state in
            let observation: Vector<Double> = model.observation.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: model.dimensions.observation)
            let noise: Vector<Double> = model.noise.observation * standardNoise
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
        ("testModel", testModel),
    ]
}

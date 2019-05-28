import XCTest

@testable import KalmanFilter

final class LocalizationTests: XCTestCase {
    let model: Model = {
        let dimensions = Dimensions(
            state: 4, // [target position x, target position y, self position x, self position y]
            input: 2, // [self position x, self position y]
            output: 1 // [distance]
        )
        
        let motionModel = LinearMotionModel(
            state: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            input: [
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
            let dist = delta.magnitude
            return [dist]
        }
        
        let noiseModel = NoiseModel(
            process: Matrix(diagonal: 0.0, size: 4),
            output: {
                return Matrix(
                    diagonal: 2.0,
                    size: dimensions.output
                ).squared()
            }()
        )
        
        return Model(
            dimensions: dimensions,
            motionModel: motionModel,
            observationModel: observationModel,
            noiseModel: noiseModel
        )
    }()
    
    func testModel() {
        let model = self.model
        
        let estimate = Estimate(
            state: [
                5.0, // Target Position X
                10.0, // Target Position Y
                0.0, // Self Position X
                0.0, // Self Position Y
            ],
            covariance: Matrix(diagonal: 1.0, size: model.dimensions.state)
        )
        
        let interval = 0.1
        let sampleCount = 200
        let inputs: [Vector<Double>] = (0..<sampleCount).map { i in
            let a = Double(i) * interval
            let r = 7.5
            let x = r * sin(a)
            let y = r * cos(a)
            return Vector(column: [x, y])
        }
        
        let states = self.makeSignal(
            initial: estimate.state,
            inputs: inputs,
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

        let filteredStates: [Vector<Double>] = Swift.zip(inputs, outputs).map { input, output in
            return kalmanFilter.filter(output: output, input: input).state
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
        self.printSheet(unfiltered2D: unfiltered, filtered2D: filtered, measured1D: measured)
        
        XCTFail("Printing found in test")
    }
    
    static var allTests = [
        ("testModel", testModel),
    ]
}

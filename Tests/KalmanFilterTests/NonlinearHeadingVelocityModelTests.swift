import XCTest

@testable import KalmanFilter

private func deg2rad(_ degree: Double) -> Double {
    return (degree / 180.0) * .pi
}

final class NonlinearHeadingVelocityModelTests: XCTestCase {
    let heading: Double = deg2rad(20.0) // heading in radians/s
    let velocity: Double = 5.0 // in m/s
    
    let model: Model = {
        // Modelled after:
        // https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRV.ipynb
        
        let dimensions = Dimensions(
            state: 4, // [position x, position y, h, velocity]
            control: 2, // [heading, velocity]
            output: 2 // [position x, position y]
        )
        
        let time = 0.1 // time delta
        
        let motionModel = NonlinearMotionModel(dimensions: dimensions) { state, control in
            let (x, y) = (state[0], state[1]) // pos-x, pos-y
            let (h, v) = (control[0], control[1]) // heading, velocity
            let t = time // delta time
            return [
                x + v * t * cos(h),
                y + v * t * sin(h),
                h,
                v,
            ]
        }
        
        let observationModel = LinearObservationModel(
            state: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
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
        0.0, // Heading
        0.0, // Velocity
    ]
    
    func estimate() -> Estimate {
        return Estimate(
            state: self.initialState,
            covariance: Matrix(diagonal: 1.0, size: 4)
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
            let heading = self.heading
            let velocity = self.velocity
            return Vector(column: [heading, velocity])
        }
        
        XCTAssertLessThan(similarity, 1.0)
    }
    
    func testVariableModel() {
        let similarity = self.filter { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let heading = self.heading * sine
            let velocity = self.velocity * cosine
            return Vector(column: [heading, velocity])
        }
        
        XCTAssertLessThan(similarity, 1.0)
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

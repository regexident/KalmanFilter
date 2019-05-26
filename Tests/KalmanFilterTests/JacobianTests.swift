import XCTest

@testable import KalmanFilter

class JacobianTests: XCTestCase {
    func testExample() {
        let jacobian = Jacobian(shape: (columns: 3, rows: 3))
        
        let state: Vector<Double> = [1.0, 2.0, 3.0]
        
        let function: (Vector<Double>) -> Vector<Double> = { state in
            return [state[0], state[1] * state[1], state[2]]
        }
        
        let matrix: Matrix<Double> = jacobian.numeric(state: state, delta: 1.0, function: function)
        
        XCTAssertEqual(matrix, Matrix(diagonal: [1.0, 4.0, 1.0]))
    }
}

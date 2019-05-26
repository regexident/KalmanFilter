import Foundation

public struct Jacobian {
    let shape: Matrix<Double>.Shape
    
    init(shape: Matrix<Double>.Shape) {
        self.shape = shape
    }
    
    public func numeric(
        state x: Vector<Double>,
        delta t: Double = 0.000001,
        function f: (Vector<Double>) -> Vector<Double>
    ) -> Matrix<Double> {
        assert(self.shape.columns == x.rows)
        
        var jacobian: Matrix<Double> = .init(shape: self.shape)
        var dx: Vector<Double> = .init(rows: x.rows)
        for i in 0..<x.rows {
            dx[i] = t
            let column = (f(x + dx) - f(x - dx)) / (t * 2.0)
            jacobian[i] = column
            dx[i] = 0.0
        }
        return jacobian
    }
}

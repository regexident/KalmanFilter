import Foundation

import Surge

public struct Jacobian {
    let rows: Int
    let columns: Int
    
    init(rows: Int, columns: Int) {
        self.rows = rows
        self.columns = columns
    }
    
    public func numeric(
        state x: Vector<Double>,
        delta t: Double = 0.000001,
        function f: (Vector<Double>) -> Vector<Double>
    ) -> Matrix<Double> {
        assert(self.columns == x.dimensions)
        
        var jacobian: Matrix<Double> = .init(rows: self.rows, columns: self.columns, repeatedValue: 0.0)
        var dx: Vector<Double> = .init(dimensions: self.columns, repeatedValue: 0.0)
        for i in 0..<x.dimensions {
            dx[i] = t
            let column = (f(x + dx) - f(x - dx)) / (t * 2.0)
            jacobian[column: i] = column.scalars
            dx[i] = 0.0
        }
        return jacobian
    }
}

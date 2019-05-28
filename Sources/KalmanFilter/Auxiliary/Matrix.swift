import Foundation
import Surge

public struct Matrix<Scalar> where Scalar: FloatingPoint, Scalar: ExpressibleByFloatLiteral {
    public typealias Shape = (rows: Int, columns: Int)
    
    public var rows: Int {
        return self.matrix.rows
    }

    public var columns: Int {
        return self.matrix.columns
    }
    
    public var shape: Shape {
        return (columns: self.columns, rows: self.rows)
    }

    internal var matrix: Surge.Matrix<Scalar>
    
    public init(shape: Shape, repeatedValue: Scalar = 0.0) {
        self.init(
            rows: shape.rows,
            columns: shape.columns,
            repeatedValue: repeatedValue
        )
    }
    
    public init(rows: Int, columns: Int, repeatedValue: Scalar = 0.0) {
        self.matrix = Surge.Matrix<Scalar>(
            rows: rows,
            columns: columns,
            repeatedValue: repeatedValue
        )
    }
    
    public init(identity size: Int) {
        self.init(diagonal: 1.0, size: size)
    }
    
    public init(diagonal repeatedValue: Scalar, size: Int) {
        let scalars = repeatElement(repeatedValue, count: size)
        self.init(diagonal: scalars)
    }
    
    public init(diagonal repeatedValue: Scalar, rows: Int, columns: Int) {
        let size = min(rows, columns)
        let scalars = repeatElement(repeatedValue, count: size)
        self.init(diagonal: scalars, rows: rows, columns: columns)
    }
    
    public init<C: Collection>(diagonal scalars: C) where C.Element == Scalar {
        let size = scalars.count
        self.init(diagonal: scalars, rows: size, columns: size)
    }
    
    public init<C: Collection>(diagonal scalars: C, rows: Int, columns: Int) where C.Element == Scalar {
        assert(scalars.count <= rows)
        assert(scalars.count <= columns)
        
        self.init(rows: rows, columns: columns)
        
        for (i, scalar) in scalars.enumerated() {
            self.matrix[i, i] = scalar
        }
    }
    
    internal init(matrix: Surge.Matrix<Scalar>) {
        self.matrix = matrix
    }
    
    public subscript(column: Int) -> Vector<Scalar> {
        get {
            return Vector(column: self.matrix[column: column])
        }
        set {
            self.matrix[column: column] = newValue.scalars
        }
    }
    
    public subscript(_ i: Int, _ j: Int) -> Scalar {
        get {
            return self.matrix[i, j]
        }
        set {
            self.matrix[i, j] = newValue
        }
    }
}

extension Matrix where Scalar == Double {
    public var determinant: Scalar? {
        return Surge.det(self.matrix)
    }
}

extension Matrix where Scalar == Float {
    public var determinant: Scalar? {
        return Surge.det(self.matrix)
    }
}

extension Matrix: CustomStringConvertible {
    public var description: String {
        return MatrixDescription(matrix: self.matrix).description
    }
}

extension Matrix where Scalar == Float {
    public func transposed() -> Matrix {
        return Matrix(matrix: Surge.transpose(self.matrix))
    }

    public func inversed() -> Matrix {
        return Matrix(matrix: Surge.inv(self.matrix))
    }
    
    public func squared() -> Matrix {
        return Matrix(matrix: Surge.pow(self.matrix, 2.0))
    }

    public static func + (left: Matrix, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.add(left.matrix, right.matrix))
    }

    public static func - (left: Matrix, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.sub(left.matrix, right.matrix))
    }

    public static func * (left: Matrix, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.mul(left.matrix, right.matrix))
    }
    
    public static func * (left: Matrix, right: Vector<Scalar>) -> Vector<Scalar> {
        return Vector(matrix: Surge.mul(left.matrix, right.matrix))
    }
}

extension Matrix where Scalar == Double {
    public func transposed() -> Matrix {
        return Matrix(matrix: Surge.transpose(self.matrix))
    }
    
    public func inversed() -> Matrix {
        return Matrix(matrix: Surge.inv(self.matrix))
    }
    
    public func squared() -> Matrix {
        return Matrix(matrix: Surge.pow(self.matrix, 2.0))
    }
    
    public static func + (left: Matrix, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.add(left.matrix, right.matrix))
    }
    
    public static func - (left: Matrix, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.sub(left.matrix, right.matrix))
    }
    
    public static func * (left: Matrix, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.mul(left.matrix, right.matrix))
    }
    
    public static func * (left: Matrix, right: Vector<Scalar>) -> Vector<Scalar> {
        return Vector(matrix: Surge.mul(left.matrix, right.matrix))
    }
    
    public static func * (left: Scalar, right: Matrix) -> Matrix {
        return Matrix(matrix: Surge.mul(left, right.matrix))
    }
}

extension Matrix: ExpressibleByArrayLiteral {
    public init(arrayLiteral: [Scalar]...) {
        self.init(matrix: Surge.Matrix(arrayLiteral))
    }
}

extension Matrix: Equatable where Scalar: Equatable {}

internal struct MatrixDescription<T: FloatingPoint & ExpressibleByFloatLiteral> {
    let matrix: Surge.Matrix<T>
}

extension MatrixDescription: CustomStringConvertible {
    public var description: String {
        let rows = (0..<self.matrix.rows).map { i in
            (0..<self.matrix.columns).map { j in
                "\(self.matrix[i, j])"
            }
        }
        
        let columnWidths: [Int] = (0..<self.matrix.columns).map { j in
            let column = rows.map { $0[j] }
            let width = column.max { $0.count < $1.count }?.count ?? 0
            return width
        }
        
        func leftPad(_ string: String, to length: Int) -> String {
            let stringLength = string.count
            guard stringLength < length else {
                return String(string.suffix(length))
            }
            let paddingLength = length - stringLength
            let padding = String(Swift.repeatElement(" ", count: paddingLength))
            return padding + string
        }
        
        let top = "┌" + columnWidths.map {
            String(Swift.repeatElement("─", count: $0))
        }.joined(separator: "┬") + "┐"
        
        let line = "├" + columnWidths.map {
            String(Swift.repeatElement("─", count: $0))
        }.joined(separator: "┼") + "┤"
        
        let bottom = "└" + columnWidths.map {
            String(Swift.repeatElement("─", count: $0))
        }.joined(separator: "┴") + "┘"
        
        let body = rows.map { row in
            let row: String = Swift.zip(row, columnWidths).map { value, width in
                leftPad(value, to: width)
            }.joined(separator: "│")
            return "│\(row)│"
        }.joined(separator: "\n\(line)\n")
        
        return "\(top)\n\(body)\n\(bottom)"
    }
}

import Foundation
import Surge

public struct Vector<Scalar> where Scalar: FloatingPoint, Scalar: ExpressibleByFloatLiteral {
    public var rows: Int {
        return self.matrix.rows
    }
    
    public var scalars: [Scalar] {
        return self.matrix[column: 0]
    }
    
    internal var matrix: Surge.Matrix<Scalar>
    
    public init(rows: Int, repeatedValue: Scalar = 0.0) {
        let column = Array(repeating: repeatedValue, count: rows)
        self.init(column: column)
    }
    
    public init(column scalars: [Scalar]) {
        self.matrix = Surge.Matrix(column: scalars)
    }
    
    internal init(matrix: Surge.Matrix<Scalar>) {
        assert(matrix.columns == 1)
        self.matrix = matrix
    }
    
    public subscript(_ i: Int) -> Scalar {
        get {
            return self.matrix[i, 0]
        }
        set {
            self.matrix[i, 0] = newValue
        }
    }
}

extension Vector where Scalar == Float {
    public var magnitude: Scalar {
        return sqrt(self * self)
    }
    
    public func distance(to other: Vector<Scalar>) -> Scalar {
        let delta = self - other
        
        return sqrt((delta * delta))
    }
}

extension Vector where Scalar == Double {
    public var magnitude: Scalar {
        return sqrt(self * self)
    }
    
    public func distance(to other: Vector<Scalar>) -> Scalar {
        let delta = self - other
        let dotProduct = delta * delta
        return sqrt(dotProduct)
    }
}

extension Vector: ExpressibleByArrayLiteral {
    public init(arrayLiteral: Scalar...) {
        self.init(matrix: Surge.Matrix(column: arrayLiteral))
    }
}

extension Vector: Equatable where Scalar: Equatable {}

extension Vector: CustomStringConvertible {
    public var description: String {
        return self.scalars.description
    }
}

extension Vector where Scalar == Float {
    public static func * (left: Vector, right: Vector) -> Scalar {
        return Surge.dot(left.matrix[column: 0], right.matrix[column: 0])
    }
    
    public static func + (left: Vector, right: Vector) -> Vector {
        return Vector(matrix: Surge.add(left.matrix, right.matrix))
    }
    
    public static func - (left: Vector, right: Vector) -> Vector {
        return Vector(matrix: Surge.sub(left.matrix, right.matrix))
    }
    
    public static func * (left: Scalar, right: Vector) -> Vector {
        return Vector(matrix: left * right.matrix)
    }
    
    public static func / (left: Vector, right: Scalar) -> Vector {
        return Vector(matrix: left.matrix / right)
    }
}

extension Vector where Scalar == Double {
    public static func * (left: Vector, right: Vector) -> Scalar {
        return Surge.dot(left.matrix[column: 0], right.matrix[column: 0])
    }
    
    public static func + (left: Vector, right: Vector) -> Vector {
        return Vector(matrix: Surge.add(left.matrix, right.matrix))
    }
    
    public static func - (left: Vector, right: Vector) -> Vector {
        return Vector(matrix: Surge.sub(left.matrix, right.matrix))
    }
    
    public static func * (left: Scalar, right: Vector) -> Vector {
        return Vector(matrix: left * right.matrix)
    }
    
    public static func / (left: Vector, right: Scalar) -> Vector {
        return Vector(matrix: left.matrix / right)
    }
}

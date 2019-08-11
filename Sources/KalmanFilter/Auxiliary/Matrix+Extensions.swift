import Surge

extension Matrix where Scalar == Float {
    public func transposed() -> Matrix {
        return Surge.transpose(self)
    }

    public func inversed() -> Matrix {
        return Surge.inv(self)
    }

    public func squared() -> Matrix {
        return Surge.pow(self, 2.0)
    }
}

extension Matrix where Scalar == Double {
    public func transposed() -> Matrix {
        return Surge.transpose(self)
    }
    
    public func inversed() -> Matrix {
        return Surge.inv(self)
    }
    
    public func squared() -> Matrix {
        return Surge.pow(self, 2.0)
    }
}

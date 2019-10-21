import Surge

public struct KalmanEstimate {
    /// State vector (aka `x` in the literature)
    public var state: Vector<Double>

    /// Estimate covariance matrix (aka `P`, or sometimes `Σ` in the literature)
    public var covariance: Matrix<Double>

    public init(
        state: Vector<Double>,
        covariance: Matrix<Double>
    ) {
        self.state = state
        self.covariance = covariance
    }
}

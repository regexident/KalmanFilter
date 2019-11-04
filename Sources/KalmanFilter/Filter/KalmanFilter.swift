import BayesFilter

public protocol KalmanFilterProtocol: BayesFilterProtocol
where
    Estimate == KalmanEstimate
{}

public typealias KalmanFilter<Predictor, Updater> = BayesFilter<Predictor, Updater, KalmanEstimate>

extension KalmanFilter: KalmanFilterProtocol
where
    Predictor: KalmanPredictorProtocol,
    Updater: KalmanUpdaterProtocol,
    Predictor.Estimate == KalmanEstimate,
    Updater.Estimate == KalmanEstimate,
    Estimate == KalmanEstimate
{}

extension KalmanFilter: KalmanPredictorProtocol
where
    Predictor: KalmanPredictorProtocol,
    Predictor.Estimate == KalmanEstimate,
    Estimate == KalmanEstimate
{
    public typealias MotionModel = Predictor.MotionModel
}

extension KalmanFilter: KalmanUpdaterProtocol
where
    Updater: KalmanUpdaterProtocol,
    Updater.Estimate == KalmanEstimate,
    Estimate == KalmanEstimate
{
    public typealias ObservationModel = Updater.ObservationModel
}

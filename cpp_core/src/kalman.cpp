#include "kalman.hpp"
#include <cmath>
#include <algorithm>

namespace linucast {

KalmanFilter::KalmanFilter(int state_size, int measurement_size) 
    : initialized_(false)
    , process_noise_(1e-4f)
    , measurement_noise_(1e-1f) {
    
    kalman_.init(state_size, measurement_size, 0, CV_32F);
    
    // State transition matrix (constant velocity model)
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,   // x' = x + vx
        0, 1, 0, 1,   // y' = y + vy
        0, 0, 1, 0,   // vx' = vx
        0, 0, 0, 1);  // vy' = vy
    
    // Measurement matrix (we only observe position)
    kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,   // measure x
        0, 1, 0, 0);  // measure y
    
    // Process noise covariance
    kalman_.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        process_noise_, 0, 0, 0,
        0, process_noise_, 0, 0,
        0, 0, process_noise_, 0,
        0, 0, 0, process_noise_);
    
    // Measurement noise covariance
    kalman_.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
        measurement_noise_, 0,
        0, measurement_noise_);
    
    // Error covariance matrix
    kalman_.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
}

void KalmanFilter::initialize(const cv::Point2f& initial_position) {
    kalman_.statePre = (cv::Mat_<float>(4, 1) << 
        initial_position.x, initial_position.y, 0, 0);
    kalman_.statePost = kalman_.statePre.clone();
    initialized_ = true;
}

cv::Point2f KalmanFilter::predict() {
    if (!initialized_) {
        return cv::Point2f(0, 0);
    }
    
    cv::Mat prediction = kalman_.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

cv::Point2f KalmanFilter::update(const cv::Point2f& measurement) {
    if (!initialized_) {
        initialize(measurement);
        return measurement;
    }
    
    cv::Mat measurement_mat = (cv::Mat_<float>(2, 1) << measurement.x, measurement.y);
    cv::Mat corrected = kalman_.correct(measurement_mat);
    
    return cv::Point2f(corrected.at<float>(0), corrected.at<float>(1));
}

void KalmanFilter::set_process_noise(float noise) {
    process_noise_ = noise;
    kalman_.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * process_noise_;
}

void KalmanFilter::set_measurement_noise(float noise) {
    measurement_noise_ = noise;
    kalman_.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * measurement_noise_;
}

// FaceTracker implementation
FaceTracker::FaceTracker(int max_faces, int max_frames_without_detection)
    : next_face_id_(0)
    , max_faces_(max_faces)
    , max_frames_without_detection_(max_frames_without_detection)
    , max_distance_threshold_(100.0f)
    , smoothing_factor_(0.7f) {
}

void FaceTracker::update(const std::vector<cv::Rect>& detections, 
                        const std::vector<std::vector<cv::Point2f>>& landmarks) {
    // Update existing tracks
    for (auto& face : tracked_faces_) {
        if (face.is_active) {
            face.frames_since_detection++;
            if (face.frames_since_detection > max_frames_without_detection_) {
                face.is_active = false;
            }
        }
    }
    
    // Match detections to existing tracks
    std::vector<bool> detection_matched(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        int best_match = find_best_match(detections[i], tracked_faces_);
        
        if (best_match >= 0) {
            // Update existing track
            auto& face = tracked_faces_[best_match];
            face.bbox = detections[i];
            if (i < landmarks.size()) {
                face.landmarks = landmarks[i];
            }
            face.frames_since_detection = 0;
            face.is_active = true;
            
            // Update Kalman filter
            cv::Point2f center(detections[i].x + detections[i].width / 2.0f,
                              detections[i].y + detections[i].height / 2.0f);
            face.position_filter.update(center);
            
            detection_matched[i] = true;
        }
    }
    
    // Create new tracks for unmatched detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_matched[i] && tracked_faces_.size() < static_cast<size_t>(max_faces_)) {
            TrackedFace new_face;
            new_face.id = next_face_id_++;
            new_face.bbox = detections[i];
            if (i < landmarks.size()) {
                new_face.landmarks = landmarks[i];
            }
            new_face.frames_since_detection = 0;
            new_face.confidence = 1.0f;
            new_face.is_active = true;
            
            // Initialize Kalman filter
            cv::Point2f center(detections[i].x + detections[i].width / 2.0f,
                              detections[i].y + detections[i].height / 2.0f);
            new_face.position_filter.initialize(center);
            
            tracked_faces_.push_back(new_face);
        }
    }
    
    // Remove inactive tracks
    tracked_faces_.erase(
        std::remove_if(tracked_faces_.begin(), tracked_faces_.end(),
                      [](const TrackedFace& face) { return !face.is_active; }),
        tracked_faces_.end());
}

std::vector<FaceTracker::TrackedFace> FaceTracker::get_active_faces() const {
    std::vector<TrackedFace> active_faces;
    std::copy_if(tracked_faces_.begin(), tracked_faces_.end(),
                std::back_inserter(active_faces),
                [](const TrackedFace& face) { return face.is_active; });
    return active_faces;
}

FaceTracker::TrackedFace* FaceTracker::get_face_by_id(int id) {
    auto it = std::find_if(tracked_faces_.begin(), tracked_faces_.end(),
                          [id](const TrackedFace& face) { return face.id == id && face.is_active; });
    return (it != tracked_faces_.end()) ? &(*it) : nullptr;
}

void FaceTracker::set_max_distance_threshold(float threshold) {
    max_distance_threshold_ = threshold;
}

void FaceTracker::set_smoothing_factor(float factor) {
    smoothing_factor_ = std::clamp(factor, 0.0f, 1.0f);
}

float FaceTracker::calculate_distance(const cv::Rect& rect1, const cv::Rect& rect2) {
    cv::Point2f center1(rect1.x + rect1.width / 2.0f, rect1.y + rect1.height / 2.0f);
    cv::Point2f center2(rect2.x + rect2.width / 2.0f, rect2.y + rect2.height / 2.0f);
    
    float dx = center1.x - center2.x;
    float dy = center1.y - center2.y;
    return std::sqrt(dx * dx + dy * dy);
}

int FaceTracker::find_best_match(const cv::Rect& detection, 
                                const std::vector<TrackedFace>& active_faces) {
    int best_match = -1;
    float min_distance = max_distance_threshold_;
    
    for (size_t i = 0; i < active_faces.size(); ++i) {
        if (!active_faces[i].is_active) continue;
        
        float distance = calculate_distance(detection, active_faces[i].bbox);
        if (distance < min_distance) {
            min_distance = distance;
            best_match = static_cast<int>(i);
        }
    }
    
    return best_match;
}

} // namespace linucast

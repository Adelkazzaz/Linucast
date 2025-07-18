#pragma once

// Include individual OpenCV headers to reduce dependencies
#include <opencv2/core.hpp>        // Basic OpenCV structures
#include <opencv2/video/tracking.hpp>  // KalmanFilter implementation
#include <vector>

namespace linucast {

class KalmanFilter {
public:
    KalmanFilter(int state_size = 4, int measurement_size = 2);
    ~KalmanFilter() = default;
    
    void initialize(const cv::Point2f& initial_position);
    cv::Point2f predict();
    cv::Point2f update(const cv::Point2f& measurement);
    
    void set_process_noise(float noise);
    void set_measurement_noise(float noise);
    
    bool is_initialized() const { return initialized_; }

private:
    cv::KalmanFilter kalman_;
    bool initialized_;
    float process_noise_;
    float measurement_noise_;
};

class FaceTracker {
public:
    struct TrackedFace {
        int id;
        cv::Rect bbox;
        std::vector<cv::Point2f> landmarks;
        KalmanFilter position_filter;
        int frames_since_detection;
        float confidence;
        bool is_active;
    };
    
    FaceTracker(int max_faces = 10, int max_frames_without_detection = 30);
    ~FaceTracker() = default;
    
    void update(const std::vector<cv::Rect>& detections, 
                const std::vector<std::vector<cv::Point2f>>& landmarks);
    
    std::vector<TrackedFace> get_active_faces() const;
    TrackedFace* get_face_by_id(int id);
    
    void set_max_distance_threshold(float threshold);
    void set_smoothing_factor(float factor);

private:
    float calculate_distance(const cv::Rect& rect1, const cv::Rect& rect2);
    int find_best_match(const cv::Rect& detection, 
                       const std::vector<TrackedFace>& active_faces);
    
    std::vector<TrackedFace> tracked_faces_;
    int next_face_id_;
    int max_faces_;
    int max_frames_without_detection_;
    float max_distance_threshold_;
    float smoothing_factor_;
};

} // namespace linucast

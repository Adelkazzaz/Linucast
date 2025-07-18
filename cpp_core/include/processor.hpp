#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>

namespace linucast {

struct Face {
    cv::Rect bbox;
    std::vector<cv::Point2f> landmarks;
    float confidence;
    int id;
    std::vector<float> embedding;
};

struct ProcessingConfig {
    bool enable_face_tracking = true;
    bool enable_background_removal = true;
    bool enable_smoothing = true;
    float smoothing_factor = 0.7f;
    int target_fps = 30;
    cv::Size output_resolution{1280, 720};
    std::string background_mode = "blur"; // "blur", "replace", "none"
    std::string background_image_path;
};

class FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor();

    bool initialize(const ProcessingConfig& config);
    void shutdown();
    
    // Main processing function
    cv::Mat process_frame(const cv::Mat& input_frame);
    
    // Face processing
    void update_faces(const std::vector<Face>& faces);
    void set_selected_face_id(int face_id);
    
    // Background processing
    void update_background_mask(const cv::Mat& mask);
    void set_background_image(const cv::Mat& background);
    
    // Configuration
    void update_config(const ProcessingConfig& config);
    ProcessingConfig get_config() const;
    
    // Performance metrics
    double get_fps() const;
    double get_processing_time_ms() const;

private:
    void smooth_frame(cv::Mat& frame);
    void apply_background_effect(cv::Mat& frame, const cv::Mat& mask);
    void track_faces();
    
    ProcessingConfig config_;
    std::vector<Face> current_faces_;
    int selected_face_id_;
    cv::Mat background_image_;
    cv::Mat last_background_mask_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    double current_fps_;
    double last_processing_time_;
    std::chrono::steady_clock::time_point last_frame_time_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread processing_thread_;
    
    // Frame smoothing
    cv::Mat previous_frame_;
    bool has_previous_frame_;
};

// Virtual camera interface
class VirtualCamera {
public:
    VirtualCamera();
    ~VirtualCamera();
    
    bool initialize(const std::string& device_path, cv::Size resolution, int fps);
    void shutdown();
    
    bool write_frame(const cv::Mat& frame);
    bool is_open() const;
    
    std::string get_device_path() const;

private:
    cv::VideoWriter writer_;
    std::string device_path_;
    cv::Size resolution_;
    int fps_;
    bool is_initialized_;
};

// Main application class
class linucastCore {
public:
    linucastCore();
    ~linucastCore();
    
    bool initialize(const ProcessingConfig& config, 
                   const std::string& input_device = "/dev/video0",
                   const std::string& output_device = "/dev/video10");
    void run();
    void shutdown();
    
    // External interfaces for Python binding
    void update_faces_from_python(const std::vector<Face>& faces);
    void update_background_mask_from_python(const cv::Mat& mask);
    void set_config_from_python(const ProcessingConfig& config);
    
    // Status
    bool is_running() const;
    double get_fps() const;

private:
    void capture_loop();
    
    std::unique_ptr<FrameProcessor> processor_;
    std::unique_ptr<VirtualCamera> virtual_cam_;
    cv::VideoCapture capture_;
    
    std::atomic<bool> running_;
    std::thread capture_thread_;
    
    std::string input_device_;
    std::string output_device_;
    ProcessingConfig config_;
};

} // namespace linucast

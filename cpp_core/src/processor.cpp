#include "processor.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace linucast {

FrameProcessor::FrameProcessor() 
    : selected_face_id_(-1)
    , current_fps_(0.0)
    , last_processing_time_(0.0)
    , running_(false)
    , has_previous_frame_(false) {
}

FrameProcessor::~FrameProcessor() {
    shutdown();
}

bool FrameProcessor::initialize(const ProcessingConfig& config) {
    config_ = config;
    running_ = true;
    
    std::cout << "FrameProcessor initialized with resolution: " 
              << config.output_resolution.width << "x" 
              << config.output_resolution.height << std::endl;
    
    return true;
}

void FrameProcessor::shutdown() {
    running_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

cv::Mat FrameProcessor::process_frame(const cv::Mat& input_frame) {
    auto start_time = std::chrono::steady_clock::now();
    
    if (input_frame.empty()) {
        return cv::Mat();
    }
    
    cv::Mat output_frame = input_frame.clone();
    
    // Resize to target resolution if needed
    if (output_frame.size() != config_.output_resolution) {
        cv::resize(output_frame, output_frame, config_.output_resolution);
    }
    
    // Apply smoothing if enabled
    if (config_.enable_smoothing) {
        smooth_frame(output_frame);
    }
    
    // Apply background effects if mask is available
    if (config_.enable_background_removal && !last_background_mask_.empty()) {
        apply_background_effect(output_frame, last_background_mask_);
    }
    
    // Update performance metrics
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        last_processing_time_ = duration.count();
        
        if (last_frame_time_ != std::chrono::steady_clock::time_point{}) {
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - last_frame_time_);
            if (frame_duration.count() > 0) {
                current_fps_ = 1000.0 / frame_duration.count();
            }
        }
        last_frame_time_ = end_time;
    }
    
    return output_frame;
}

void FrameProcessor::smooth_frame(cv::Mat& frame) {
    if (!has_previous_frame_) {
        previous_frame_ = frame.clone();
        has_previous_frame_ = true;
        return;
    }
    
    // Temporal smoothing using weighted average
    cv::addWeighted(previous_frame_, config_.smoothing_factor, 
                   frame, 1.0 - config_.smoothing_factor, 0, frame);
    
    previous_frame_ = frame.clone();
}

void FrameProcessor::apply_background_effect(cv::Mat& frame, const cv::Mat& mask) {
    cv::Mat mask_resized;
    if (mask.size() != frame.size()) {
        cv::resize(mask, mask_resized, frame.size());
    } else {
        mask_resized = mask;
    }
    
    // Ensure mask is single channel
    if (mask_resized.channels() > 1) {
        cv::cvtColor(mask_resized, mask_resized, cv::COLOR_BGR2GRAY);
    }
    
    // Normalize mask to 0-1 range
    mask_resized.convertTo(mask_resized, CV_32F, 1.0/255.0);
    
    if (config_.background_mode == "blur") {
        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(51, 51), 0);
        
        // Blend based on mask
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                float alpha = mask_resized.at<float>(y, x);
                cv::Vec3b original = frame.at<cv::Vec3b>(y, x);
                cv::Vec3b blur = blurred.at<cv::Vec3b>(y, x);
                
                frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    alpha * original[0] + (1 - alpha) * blur[0],
                    alpha * original[1] + (1 - alpha) * blur[1],
                    alpha * original[2] + (1 - alpha) * blur[2]
                );
            }
        }
    } else if (config_.background_mode == "replace" && !background_image_.empty()) {
        cv::Mat bg_resized;
        cv::resize(background_image_, bg_resized, frame.size());
        
        // Blend with background image
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                float alpha = mask_resized.at<float>(y, x);
                cv::Vec3b original = frame.at<cv::Vec3b>(y, x);
                cv::Vec3b background = bg_resized.at<cv::Vec3b>(y, x);
                
                frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    alpha * original[0] + (1 - alpha) * background[0],
                    alpha * original[1] + (1 - alpha) * background[1],
                    alpha * original[2] + (1 - alpha) * background[2]
                );
            }
        }
    }
}

void FrameProcessor::update_faces(const std::vector<Face>& faces) {
    current_faces_ = faces;
}

void FrameProcessor::set_selected_face_id(int face_id) {
    selected_face_id_ = face_id;
}

void FrameProcessor::update_background_mask(const cv::Mat& mask) {
    last_background_mask_ = mask.clone();
}

void FrameProcessor::set_background_image(const cv::Mat& background) {
    background_image_ = background.clone();
}

void FrameProcessor::update_config(const ProcessingConfig& config) {
    config_ = config;
}

ProcessingConfig FrameProcessor::get_config() const {
    return config_;
}

double FrameProcessor::get_fps() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_fps_;
}

double FrameProcessor::get_processing_time_ms() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return last_processing_time_;
}

// VirtualCamera implementation
VirtualCamera::VirtualCamera() 
    : fps_(30)
    , is_initialized_(false) {
}

VirtualCamera::~VirtualCamera() {
    shutdown();
}

bool VirtualCamera::initialize(const std::string& device_path, cv::Size resolution, int fps) {
    device_path_ = device_path;
    resolution_ = resolution;
    fps_ = fps;
    
    // Try to open the v4l2loopback device
    writer_.open(device_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, resolution);
    
    if (!writer_.isOpened()) {
        std::cerr << "Failed to open virtual camera device: " << device_path << std::endl;
        return false;
    }
    
    is_initialized_ = true;
    std::cout << "Virtual camera initialized: " << device_path 
              << " (" << resolution.width << "x" << resolution.height 
              << " @ " << fps << " fps)" << std::endl;
    
    return true;
}

void VirtualCamera::shutdown() {
    if (writer_.isOpened()) {
        writer_.release();
    }
    is_initialized_ = false;
}

bool VirtualCamera::write_frame(const cv::Mat& frame) {
    if (!is_open() || frame.empty()) {
        return false;
    }
    
    cv::Mat output_frame = frame;
    
    // Ensure frame is the correct size
    if (frame.size() != resolution_) {
        cv::resize(frame, output_frame, resolution_);
    }
    
    writer_.write(output_frame);
    return true;
}

bool VirtualCamera::is_open() const {
    return is_initialized_ && writer_.isOpened();
}

std::string VirtualCamera::get_device_path() const {
    return device_path_;
}

// linucastCore implementation
linucastCore::linucastCore() 
    : running_(false) {
}

linucastCore::~linucastCore() {
    shutdown();
}

bool linucastCore::initialize(const ProcessingConfig& config,
                              const std::string& input_device,
                              const std::string& output_device) {
    config_ = config;
    input_device_ = input_device;
    output_device_ = output_device;
    
    // Initialize frame processor
    processor_ = std::make_unique<FrameProcessor>();
    if (!processor_->initialize(config)) {
        std::cerr << "Failed to initialize frame processor" << std::endl;
        return false;
    }
    
    // Initialize virtual camera
    virtual_cam_ = std::make_unique<VirtualCamera>();
    if (!virtual_cam_->initialize(output_device, config.output_resolution, config.target_fps)) {
        std::cerr << "Failed to initialize virtual camera" << std::endl;
        return false;
    }
    
    // Initialize input capture
    capture_.open(input_device);
    if (!capture_.isOpened()) {
        std::cerr << "Failed to open input device: " << input_device << std::endl;
        return false;
    }
    
    // Set capture properties
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, config.output_resolution.width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config.output_resolution.height);
    capture_.set(cv::CAP_PROP_FPS, config.target_fps);
    
    std::cout << "linucastCore initialized successfully" << std::endl;
    return true;
}

void linucastCore::run() {
    if (!processor_ || !virtual_cam_ || !capture_.isOpened()) {
        std::cerr << "System not properly initialized" << std::endl;
        return;
    }
    
    running_ = true;
    capture_thread_ = std::thread(&linucastCore::capture_loop, this);
    
    std::cout << "linucast started. Press Ctrl+C to stop." << std::endl;
}

void linucastCore::capture_loop() {
    cv::Mat frame;
    
    while (running_) {
        if (!capture_.read(frame) || frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps fallback
            continue;
        }
        
        // Process the frame
        cv::Mat processed_frame = processor_->process_frame(frame);
        
        // Write to virtual camera
        if (!processed_frame.empty()) {
            virtual_cam_->write_frame(processed_frame);
        }
        
        // Frame rate limiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / config_.target_fps));
    }
}

void linucastCore::shutdown() {
    running_ = false;
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    if (capture_.isOpened()) {
        capture_.release();
    }
    
    if (virtual_cam_) {
        virtual_cam_->shutdown();
    }
    
    if (processor_) {
        processor_->shutdown();
    }
    
    std::cout << "linucastCore shutdown complete" << std::endl;
}

void linucastCore::update_faces_from_python(const std::vector<Face>& faces) {
    if (processor_) {
        processor_->update_faces(faces);
    }
}

void linucastCore::update_background_mask_from_python(const cv::Mat& mask) {
    if (processor_) {
        processor_->update_background_mask(mask);
    }
}

void linucastCore::set_config_from_python(const ProcessingConfig& config) {
    config_ = config;
    if (processor_) {
        processor_->update_config(config);
    }
}

bool linucastCore::is_running() const {
    return running_;
}

double linucastCore::get_fps() const {
    if (processor_) {
        return processor_->get_fps();
    }
    return 0.0;
}

} // namespace linucast

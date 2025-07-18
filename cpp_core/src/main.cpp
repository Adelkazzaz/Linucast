#include "processor.hpp"
#include <iostream>
#include <signal.h>
#include <unistd.h>

using namespace linucast;

// Global flag for signal handling
volatile sig_atomic_t g_shutdown = 0;

void signal_handler(int signal) {
    g_shutdown = 1;
    std::cout << "\nShutdown signal received..." << std::endl;
}

int main(int argc, char* argv[]) {
    // Setup signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "linucast - AI Virtual Camera for Linux" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Parse command line arguments
    std::string input_device = "/dev/video0";
    std::string output_device = "/dev/video10";
    bool debug = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_device = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_device = argv[++i];
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --input <device>   Input camera device (default: /dev/video0)" << std::endl;
            std::cout << "  --output <device>  Output virtual camera device (default: /dev/video10)" << std::endl;
            std::cout << "  --debug           Enable debug output" << std::endl;
            std::cout << "  --help, -h        Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create configuration
    ProcessingConfig config;
    config.enable_face_tracking = true;
    config.enable_background_removal = false; // Start with basic functionality
    config.enable_smoothing = true;
    config.smoothing_factor = 0.7f;
    config.target_fps = 30;
    config.output_resolution = cv::Size(1280, 720);
    config.background_mode = "none";
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input device: " << input_device << std::endl;
    std::cout << "  Output device: " << output_device << std::endl;
    std::cout << "  Resolution: " << config.output_resolution.width 
              << "x" << config.output_resolution.height << std::endl;
    std::cout << "  Target FPS: " << config.target_fps << std::endl;
    std::cout << "  Debug mode: " << (debug ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;
    
    // Initialize and run the core system
    linucastCore core;
    
    if (!core.initialize(config, input_device, output_device)) {
        std::cerr << "Failed to initialize linucast core" << std::endl;
        return 1;
    }
    
    // Start processing
    core.run();
    
    // Main loop
    while (!g_shutdown && core.is_running()) {
        sleep(1);
        
        if (debug) {
            std::cout << "FPS: " << core.get_fps() << std::endl;
        }
    }
    
    // Shutdown
    std::cout << "Shutting down..." << std::endl;
    core.shutdown();
    
    std::cout << "linucast stopped." << std::endl;
    return 0;
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "processor.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf_info = input.request();
    
    int rows = buf_info.shape[0];
    int cols = buf_info.shape[1];
    int channels = (buf_info.ndim == 3) ? buf_info.shape[2] : 1;
    
    cv::Mat mat(rows, cols, (channels == 3) ? CV_8UC3 : CV_8UC1, 
                (unsigned char*)buf_info.ptr);
    
    return mat.clone(); // Make a copy to ensure memory safety
}

// Helper function to convert cv::Mat to numpy array
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    if (mat.channels() == 3) {
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols, mat.channels()},
            {sizeof(uint8_t) * mat.cols * mat.channels(), 
             sizeof(uint8_t) * mat.channels(), 
             sizeof(uint8_t)},
            mat.data
        );
    } else {
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols},
            {sizeof(uint8_t) * mat.cols, sizeof(uint8_t)},
            mat.data
        );
    }
}

PYBIND11_MODULE(linucast_cpp, m) {
    m.doc() = "Linucast C++ backend bindings";
    
    // Face structure
    py::class_<linucast::Face>(m, "Face")
        .def(py::init<>())
        .def_readwrite("bbox", &linucast::Face::bbox)
        .def_readwrite("landmarks", &linucast::Face::landmarks)
        .def_readwrite("confidence", &linucast::Face::confidence)
        .def_readwrite("id", &linucast::Face::id)
        .def_readwrite("embedding", &linucast::Face::embedding);
    
    // ProcessingConfig structure
    py::class_<linucast::ProcessingConfig>(m, "ProcessingConfig")
        .def(py::init<>())
        .def_readwrite("enable_face_tracking", &linucast::ProcessingConfig::enable_face_tracking)
        .def_readwrite("enable_background_removal", &linucast::ProcessingConfig::enable_background_removal)
        .def_readwrite("enable_smoothing", &linucast::ProcessingConfig::enable_smoothing)
        .def_readwrite("smoothing_factor", &linucast::ProcessingConfig::smoothing_factor)
        .def_readwrite("target_fps", &linucast::ProcessingConfig::target_fps)
        .def_readwrite("background_mode", &linucast::ProcessingConfig::background_mode)
        .def_readwrite("background_image_path", &linucast::ProcessingConfig::background_image_path);
    
    // OpenCV Rect binding
    py::class_<cv::Rect>(m, "Rect")
        .def(py::init<int, int, int, int>())
        .def_readwrite("x", &cv::Rect::x)
        .def_readwrite("y", &cv::Rect::y)
        .def_readwrite("width", &cv::Rect::width)
        .def_readwrite("height", &cv::Rect::height);
    
    // OpenCV Point2f binding
    py::class_<cv::Point2f>(m, "Point2f")
        .def(py::init<float, float>())
        .def_readwrite("x", &cv::Point2f::x)
        .def_readwrite("y", &cv::Point2f::y);
    
    // OpenCV Size binding
    py::class_<cv::Size>(m, "Size")
        .def(py::init<int, int>())
        .def_readwrite("width", &cv::Size::width)
        .def_readwrite("height", &cv::Size::height);
    
    // FrameProcessor class
    py::class_<linucast::FrameProcessor>(m, "FrameProcessor")
        .def(py::init<>())
        .def("initialize", &linucast::FrameProcessor::initialize)
        .def("shutdown", &linucast::FrameProcessor::shutdown)
        .def("process_frame", [](linucast::FrameProcessor& self, py::array_t<uint8_t> input) {
            cv::Mat input_mat = numpy_to_mat(input);
            cv::Mat result = self.process_frame(input_mat);
            return mat_to_numpy(result);
        })
        .def("update_faces", &linucast::FrameProcessor::update_faces)
        .def("set_selected_face_id", &linucast::FrameProcessor::set_selected_face_id)
        .def("update_background_mask", [](linucast::FrameProcessor& self, py::array_t<uint8_t> mask) {
            cv::Mat mask_mat = numpy_to_mat(mask);
            self.update_background_mask(mask_mat);
        })
        .def("set_background_image", [](linucast::FrameProcessor& self, py::array_t<uint8_t> bg) {
            cv::Mat bg_mat = numpy_to_mat(bg);
            self.set_background_image(bg_mat);
        })
        .def("update_config", &linucast::FrameProcessor::update_config)
        .def("get_config", &linucast::FrameProcessor::get_config)
        .def("get_fps", &linucast::FrameProcessor::get_fps)
        .def("get_processing_time_ms", &linucast::FrameProcessor::get_processing_time_ms);
    
    // VirtualCamera class
    py::class_<linucast::VirtualCamera>(m, "VirtualCamera")
        .def(py::init<>())
        .def("initialize", &linucast::VirtualCamera::initialize)
        .def("shutdown", &linucast::VirtualCamera::shutdown)
        .def("write_frame", [](linucast::VirtualCamera& self, py::array_t<uint8_t> frame) {
            cv::Mat frame_mat = numpy_to_mat(frame);
            return self.write_frame(frame_mat);
        })
        .def("is_open", &linucast::VirtualCamera::is_open)
        .def("get_device_path", &linucast::VirtualCamera::get_device_path);
    
    // LinucastCore class
    py::class_<linucast::linucastCore>(m, "LinucastCore")
        .def(py::init<>())
        .def("initialize", &linucast::linucastCore::initialize)
        .def("run", &linucast::linucastCore::run)
        .def("shutdown", &linucast::linucastCore::shutdown)
        .def("update_faces_from_python", &linucast::linucastCore::update_faces_from_python)
        .def("update_background_mask_from_python", [](linucast::linucastCore& self, py::array_t<uint8_t> mask) {
            cv::Mat mask_mat = numpy_to_mat(mask);
            self.update_background_mask_from_python(mask_mat);
        })
        .def("set_config_from_python", &linucast::linucastCore::set_config_from_python)
        .def("is_running", &linucast::linucastCore::is_running)
        .def("get_fps", &linucast::linucastCore::get_fps);
    
    // Utility functions
    m.def("numpy_to_mat", &numpy_to_mat, "Convert numpy array to OpenCV Mat");
    m.def("mat_to_numpy", &mat_to_numpy, "Convert OpenCV Mat to numpy array");
}

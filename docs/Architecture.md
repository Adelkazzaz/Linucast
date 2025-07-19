# Linucast Architecture

This document describes the architecture of Linucast, an AI-powered virtual camera system for Linux.

## Overview

Linucast is designed with a hybrid architecture that combines Python for AI components and C++ for high-performance video processing. This hybrid approach leverages the strengths of each language:

- **Python**: Used for AI models, GUI, and overall application management
- **C++**: Used for high-performance video processing and virtual camera integration

## System Architecture

```mermaid
flowchart TB
    subgraph "Python Layer"
        pyApp["Main Application"]
        pyAI["AI Components"]
        pyGUI["GUI Interface"]
        pyBridge["Python-C++ Bridge"]
    end
    
    subgraph "C++ Layer"
        cppCore["LinucastCore"]
        frameProc["Frame Processor"]
        virtCam["Virtual Camera"]
    end
    
    subgraph "External"
        inCam["Input Camera"]
        outCam["V4L2 Virtual Device"]
        vidApps["Video Applications"]
    end
    
    inCam --> cppCore
    pyApp --> pyAI
    pyApp --> pyGUI
    pyApp --> pyBridge
    
    pyBridge <--> cppCore
    
    cppCore --> frameProc
    cppCore --> virtCam
    
    frameProc <-- "AI Results" --> pyAI
    
    virtCam --> outCam
    outCam --> vidApps
    
    pyAI -- "Face Detection\nBackground Segmentation" --> pyBridge
```

## Processing Pipeline

```mermaid
flowchart LR
    input["Input Camera"] --> capture["Frame Capture"]
    capture --> resize["Resize Frame"]
    
    subgraph "AI Processing"
        resize --> faceDetect["Face Detection"]
        resize --> bgSegment["Background Segmentation"]
        faceDetect --> faceID["Face Identification"]
    end
    
    subgraph "C++ Processing"
        faceID --> tracking["Face Tracking"]
        bgSegment --> bgEffect["Background Effects"]
        tracking --> smooth["Frame Smoothing"]
        bgEffect --> smooth
    end
    
    smooth --> output["Virtual Camera Output"]
```

## Component Details

### Python Components

#### AI Module

The AI module contains three main components:

```mermaid
classDiagram
    class FaceDetector {
        +initialize()
        +detect_faces(frame)
        -mp_face_detection
        -mp_face_mesh
    }
    
    class FaceIdentifier {
        +initialize()
        +identify_faces(faces, frame)
        -model
    }
    
    class BackgroundSegmenter {
        +initialize()
        +segment_background(frame)
        -model
    }
    
    FaceDetector -- FaceIdentifier
    FaceIdentifier -- BackgroundSegmenter
```

#### Bridge Architecture

The bridge module manages the communication between Python and C++:

```mermaid
classDiagram
    class CppBridge {
        +initialize(input_device, output_device)
        +start_processing()
        +stop_processing()
        +update_faces(faces)
        +update_background_mask(mask)
        +get_latest_frame()
        -_processing_loop()
    }
    
    class ProcessingConfig {
        +enable_face_tracking
        +enable_background_removal
        +enable_smoothing
        +smoothing_factor
        +target_fps
        +background_mode
        +to_cpp_config()
    }
    
    CppBridge --> ProcessingConfig
```

### C++ Components

#### Core Processing Architecture

```mermaid
classDiagram
    class LinucastCore {
        +initialize(config, input_device, output_device)
        +run()
        +shutdown()
        +update_faces_from_python(faces)
        +update_background_mask_from_python(mask)
        -capture_loop()
    }
    
    class FrameProcessor {
        +initialize(config)
        +process_frame(input_frame)
        +update_faces(faces)
        +update_background_mask(mask)
        -smooth_frame(frame)
        -apply_background_effect(frame, mask)
    }
    
    class VirtualCamera {
        +initialize(device_path, resolution, fps)
        +write_frame(frame)
        +shutdown()
    }
    
    class Face {
        +bbox
        +landmarks
        +confidence
        +id
        +embedding
    }
    
    class ProcessingConfig {
        +enable_face_tracking
        +enable_background_removal
        +enable_smoothing
        +smoothing_factor
        +target_fps
        +background_mode
    }
    
    LinucastCore --> FrameProcessor
    LinucastCore --> VirtualCamera
    FrameProcessor --> Face
    LinucastCore --> ProcessingConfig
    FrameProcessor --> ProcessingConfig
```

## Data Flow

```mermaid
sequenceDiagram
    participant IC as Input Camera
    participant CC as C++ Core
    participant FP as Frame Processor
    participant PY as Python AI
    participant VC as Virtual Camera
    
    IC->>CC: Raw video frames
    CC->>PY: Frame for AI processing
    PY->>PY: Face detection
    PY->>PY: Background segmentation
    PY->>CC: AI results (faces, masks)
    CC->>FP: Frame & AI data for processing
    FP->>FP: Apply effects & transformations
    FP->>CC: Processed frame
    CC->>VC: Final frame
    VC->>Applications: Virtual video device
```

## Initialization Sequence

```mermaid
sequenceDiagram
    participant APP as Main App
    participant PY as Python Components
    participant CPP as C++ Core
    
    APP->>PY: Initialize AI components
    PY-->>APP: AI initialized
    APP->>CPP: Initialize C++ bridge
    CPP->>CPP: Setup frame processor
    CPP->>CPP: Setup virtual camera
    CPP-->>APP: C++ bridge initialized
    APP->>CPP: Start processing
    CPP->>CPP: Start capture loop
    APP->>PY: Start GUI/headless mode
```

## Fallback Mechanism

```mermaid
flowchart TD
    init["Initialize C++ Bridge"]
    check{"C++ Available?"}
    cpp["Use C++ Processing"]
    py["Use Python Fallback"]
    process["Process Video"]
    
    init --> check
    check -->|Yes| cpp
    check -->|No| py
    cpp --> process
    py --> process
```

## Project Structure

```mermaid
flowchart TB
    linucast["Linucast"]
    
    linucast --> python["python_core/"]
    linucast --> cpp["cpp_core/"]
    linucast --> docs["docs/"]
    linucast --> install["install/"]
    linucast --> tools["tools/"]
    
    python --> py_main["linucast/"]
    python --> py_assets["assets/"]
    python --> py_config["config.yaml"]
    
    py_main --> py_ai["ai/"]
    py_main --> py_gui["gui/"]
    py_main --> py_ipc["ipc/"]
    
    cpp --> cpp_include["include/"]
    cpp --> cpp_src["src/"]
    cpp --> cpp_build["build/"]
    cpp --> cpp_cmake["CMakeLists.txt"]
    
    install --> install_scripts["Installation Scripts"]
    tools --> tools_scripts["Utility Scripts"]
    docs --> documentation["Documentation Files"]
```

## Integration with External Applications

Linucast integrates with external applications through the V4L2 loopback virtual camera module, allowing processed video to be used with any application that supports standard camera devices.

```mermaid
flowchart LR
    linucast["Linucast"]
    v4l2["V4L2 Loopback"]
    apps["Applications"]
    
    linucast --> v4l2
    v4l2 --> apps
    
    apps --> browser["Web Browsers"]
    apps --> meet["Video Conferencing"]
    apps --> stream["Streaming Tools"]
    apps --> record["Recording Software"]
```

## Conclusion

The hybrid architecture of Linucast combines the flexibility and AI capabilities of Python with the performance benefits of C++. This design enables high-performance video processing while leveraging advanced AI models for features like face detection and background segmentation.

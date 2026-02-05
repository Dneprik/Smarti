    #include <iostream>
    #include <vector>
    #include <string>
    #include <opencv2/opencv.hpp>
    #include <opencv2/dnn.hpp>

    // Settings 
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0; 
    const float SCORE_THRESHOLD = 0.15;
    const float NMS_THRESHOLD = 0.4; 

    // Camera step size
    const int STEP_SIZE = 320; 

    struct Defect {
        int id;
        float confidence;
        cv::Rect box;                   
        std::vector<cv::Point> polygon; 
        int tileIndex;          

        // Output as JSON 
        void print(bool isLast = false) const {
            std::cout << "  {" << std::endl;
            std::cout << "    \"id\": " << id << "," << std::endl;
            std::cout << "    \"tile_index\": " << tileIndex << "," << std::endl;
            std::cout << "    \"bbox\": [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]," << std::endl;
            
            std::cout << "    \"polygon\": [";
            for (size_t i = 0; i < polygon.size(); ++i) {
                std::cout << "[" << polygon[i].x << ", " << polygon[i].y << "]";
                if (i < polygon.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  }" << (isLast ? "" : ",") << std::endl;
        }
    };

    // Find count of picture 
    int extractTileIndex(std::string filename) {
        size_t underscorePos = filename.find('_');
        size_t dotPos = filename.find('.');
        if (underscorePos != std::string::npos && dotPos != std::string::npos) {
            std::string numberStr = filename.substr(underscorePos + 1, dotPos - underscorePos - 1);
            try { return std::stoi(numberStr); } catch (...) { return 0; }
        }
        return 0;
    }

    // Get knot polygon
    std::vector<cv::Point> getKnotContour(const cv::Mat& wholeImage, cv::Rect box) {
        box = box & cv::Rect(0, 0, wholeImage.cols, wholeImage.rows);
        if (box.area() <= 0) return {};
        
        cv::Mat roi = wholeImage(box);
        cv::Mat gray, binary;
        
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) return {};
        
        // Find the largest contour
        size_t maxIdx = 0;
        double maxArea = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) { maxArea = area; maxIdx = i; }
        }
        return contours[maxIdx]; 
    }


    int main() {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
        
        cv::dnn::Net net;
        try {
            net = cv::dnn::readNetFromONNX("best.onnx");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } catch (const cv::Exception& e) { 
            std::cerr << "Error: " << e.what() << std::endl;
            return -1; 
        }

        std::vector<std::string> tileFiles = { "0_0.png", "0_1.png", "0_5.png", "0_6.png", "0_7.png", "0_9.png", "0_10.png", "0_14.png", "0_15.png"};
        
        std::vector<cv::Rect> globalBoxes;
        std::vector<float> globalConfidences;
        std::vector<std::vector<cv::Point>> globalPolygons;
        std::vector<int> sourceIndices;

        std::string folderPath = "images/";

        for (size_t i = 0; i < tileFiles.size(); ++i) {
            std::string filename = tileFiles[i];
            std::string fullPath = folderPath + filename;
            
            cv::Mat tile = cv::imread(fullPath);

            if (tile.empty()) {
                std::cerr << "Skipping file: " << fullPath << std::endl;
                continue;  
            }

            int tileRealIndex = extractTileIndex(fullPath); 
            int currentGlobalOffset = tileRealIndex * STEP_SIZE; 
            
            cv::Mat blob;
            
            cv::dnn::blobFromImage(tile, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            
            cv::Mat output_t = outputs[0].reshape(1, outputs[0].size[1]).t();
            float* data = (float*)output_t.data;
            
            float x_factor = (float)tile.cols / INPUT_WIDTH;
            float y_factor = (float)tile.rows / INPUT_HEIGHT;

            for (int r = 0; r < 8400; ++r) {
                float* row = data + (r * 5);
                float score = row[4];
                if (score > SCORE_THRESHOLD) {
                    float cx = row[0], cy = row[1], w = row[2], h = row[3];
                    
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    
                    cv::Rect localBox(left, top, width, height);

                    std::vector<cv::Point> localContour = getKnotContour(tile, localBox);

                    cv::Rect globalBox = localBox;
                    globalBox.x += currentGlobalOffset; 

                    std::vector<cv::Point> globalContour;
                    for(auto& pt : localContour) {
                        globalContour.push_back(cv::Point(pt.x + localBox.x + currentGlobalOffset, pt.y + localBox.y));
                    }

                    globalBoxes.push_back(globalBox);
                    globalConfidences.push_back(score);
                    globalPolygons.push_back(globalContour);
                    sourceIndices.push_back(tileRealIndex);
                }
            }
        }

        std::cout << "Merging overlapping defects..." << std::endl;
        std::vector<int> indices;
        cv::dnn::NMSBoxes(globalBoxes, globalConfidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        std::vector<Defect> resultDefects;
        for (int i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Defect d;
            d.id = i;
            d.confidence = globalConfidences[idx];
            d.box = globalBoxes[idx];
            d.polygon = globalPolygons[idx];
            d.tileIndex = sourceIndices[idx];
            resultDefects.push_back(d);
        }

        std::cout << "\n--- FINAL OUTPUT JSON ---" << std::endl;
        std::cout << "[" << std::endl;
        for (size_t i = 0; i < resultDefects.size(); ++i) {
            resultDefects[i].print(i == resultDefects.size() - 1);
        }
        std::cout << "]" << std::endl;

        return 0;
    }
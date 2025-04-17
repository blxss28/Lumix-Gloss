#include <opencv2/opencv.hpp>
#include <vector>

// Apply Canny edge detection to refine depth buffer
cv::Mat applyEdgeDetection(const cv::Mat& frame) {
    cv::Mat gray, edges;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200); // Canny edge detection parameters
    return edges;
}

// Combine depth map with edges for refinement
cv::Mat combineDepthAndEdges(const cv::Mat& depthMap, const cv::Mat& edges) {
    cv::Mat refinedDepth;
    cv::addWeighted(depthMap, 0.8, edges, 0.2, 0, refinedDepth);
    return refinedDepth;
}
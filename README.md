# auto-traffic-camera-calib
Code implementation of “AutoCalib: Automatic Traffic Camera Calibration at Scale”

## Pipeline

| Step | Desc                         | Input              | Output                | Method              |
| ---- | ---------------------------- | ------------------ | --------------------- | ------------------- |
| 1    | Vehicle Detection            | Images             | Bounding Boxes        | EfficientDet        |
| 2    | Vehicle Keypoint Detection   | Cropped Images     | Keypoints             | KP_Orientation_Net  |
| 3    | Vehicle Model Classification | Cropped Images     | Model Index           | VGG-16              |
| 4    | Single Vehicle Calibration   | KPs, Real Coords   | Calibration Candidate | SolvePnP            |
| 5    | Calibration Filtering        | Calibrations       | Valid Calibrations    | Statistical Filters |
| 6    | Calibration Averaging        | Valid Calibrations | Final Calibration     | Median Filter       |


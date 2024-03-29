# RANSAC Plane Fitting

This repository contains a Python implementation of the Random Sample Consensus (RANSAC) algorithm for fitting a plane to a 3D point cloud. The goal is to robustly estimate the parameters of a plane in the presence of outliers.

## Task Overview
The task involves implementing a custom RANSAC algorithm to fit a plane to a given 3D point cloud. Using the provided code snippet, a demo point cloud from Open3D is loaded and visualized. The implemented RANSAC function iteratively fits planes to subsets of points, identifying inliers based on a specified distance threshold.

### Code Explanation
#### RANSAC Function
- The `RANSAC` function performs the RANSAC algorithm to fit a plane to a set of points.
- Randomly selects subsets of points, fits a plane, and identifies inliers based on a distance threshold.
- Iterates multiple times to find the best-fitted plane.

#### create_plane_mesh Function
- The `create_plane_mesh` function takes plane parameters and meshgrid ranges as input.
- Generates a meshgrid for the plane, creating a set of points in the x-y plane.
- Forms a mesh by connecting vertices to create triangles, representing the fitted plane.
- Sets colors for vertices (green in this case).

#### Main Part
- Loads a demo point cloud from Open3D.
- Applies the custom RANSAC function to find the plane that fits the point cloud.
- Creates a plane mesh using the fitted plane parameters.
- Visualizes the point cloud and the fitted plane mesh using Open3D.

### Usage
To run the code, make sure to have Open3D and NumPy installed. You can install them using the following command:
```bash
pip install open3d numpy
```
After installing the dependencies, execute the provided script.

### Note
Using RANSAC APIs from existing libraries instead of the custom implementation will result in a reduced score (60% of the total score), as mentioned in the task requirements.

Feel free to explore, modify, and enhance the code for your specific use case!

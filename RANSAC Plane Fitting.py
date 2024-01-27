# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:43:35 2023

@author: techv
"""
import open3d as o3d
import numpy as np

import random


def RANSAC(points):
    # Define the RANSAC parameters
    num_iterations = 1000
    min_inliers = 50
    distance_threshold = 0.01  # Tolerance for inlier classification
    
    # Define a function to fit a plane to a subset of points
    def fit_plane(sample):
        # Calculate the plane parameters (normal and distance to the origin)
        p1, p2, p3 = sample
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        
        return normal, d
    
    best_plane = None
    best_inliers = []
    
    for _ in range(num_iterations):
        # Randomly sample a minimal subset of points to fit a plane
        sample_indices = random.sample(range(len(points)), 3)
        sample_points = points[sample_indices]
        # print("\n",np.shape(sample_points),sample_points)
        # Fit a plane to the sample
        plane_normal, plane_d = fit_plane(sample_points)
        
        # Compute the distances from all points to the plane
        distances = np.abs(np.dot(points, plane_normal) + plane_d) / np.linalg.norm(plane_normal)
        
        # Count the number of inliers
        inlier_indices = np.where(distances < distance_threshold)
        num_inliers = len(inlier_indices[0])
        
        # Update the best model if this model has more inliers
        if num_inliers > min_inliers and num_inliers > len(best_inliers):
            best_inliers = inlier_indices
            best_plane = (plane_normal, plane_d)
    
    # Extract the best plane and inliers
    best_plane_normal, best_plane_d = best_plane
    inlier_points = points[best_inliers]
    return best_plane

def create_plane_mesh(plane_parameters, x_range, y_range):
    a = plane_parameters[0][0]
    b = plane_parameters[0][1]
    c = plane_parameters[0][2]
    d = plane_parameters[1]

    xx, yy = np.meshgrid(x_range, y_range)
    zz = (-a * xx - b * yy - d) / c
    vertices = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    triangles = []
    colors = np.zeros_like(vertices)  # You can set the color here
    for i in range(zz.shape[0] - 1):
        for j in range(zz.shape[1] - 1):
            v1 = i * zz.shape[1] + j
            v2 = v1 + 1
            v3 = (i + 1) * zz.shape[1] + j
            v4 = v3 + 1
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])
            # Set color for each vertex
            colors[v1] = [0, 1, 0]  # RGB color values (red)
            colors[v2] = [0, 1, 0]  # RGB color values (green)
            colors[v3] = [0, 1, 0]  # RGB color values (blue)
            colors[v4] = [0, 1, 0]  # RGB color values (yellow)

    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    plane_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return plane_mesh


# read demo point cloud provided by Open3D
pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)
inlier_points = RANSAC(np.asarray(pcd.points))
print(inlier_points)

points = np.asarray(pcd.points)

x_range = np.linspace(points[:, 0].min(), points[:, 0].max(), 10)
y_range = np.linspace(points[:, 1].min(), points[:, 1].max(), 10)

plane_mesh = create_plane_mesh(inlier_points, x_range, y_range)

o3d.visualization.draw_geometries([pcd, plane_mesh],
                                zoom=1,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])

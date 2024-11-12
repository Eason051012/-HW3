# Import necessary libraries
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Step 2: Define an elliptical distribution function
def elliptical_function(x1, x2, semi_major, semi_minor):
    # Z values decrease as we move further from the center, forming a pyramid-like shape with elliptical base.
    return np.maximum(0, 10 - ((x1 / semi_major)**2 + (x2 / semi_minor)**2))

# Step 5: Visualization function using Plotly
def plot_3d_elliptical_with_plane(X_new, z_threshold):
    # Separate data points based on the plane
    above_plane = X_new[X_new[:, 2] > z_threshold]
    below_plane = X_new[X_new[:, 2] <= z_threshold]

    # Create a Plotly figure
    fig = go.Figure()

    # Add scatter plots for points above and below the plane
    fig.add_trace(go.Scatter3d(
        x=above_plane[:, 0], y=above_plane[:, 1], z=above_plane[:, 2],
        mode='markers', marker=dict(color='blue', size=3),
        name='Above Plane'
    ))

    fig.add_trace(go.Scatter3d(
        x=below_plane[:, 0], y=below_plane[:, 1], z=below_plane[:, 2],
        mode='markers', marker=dict(color='red', size=3),
        name='Below Plane'
    ))

    # Generate a plane at z = z_threshold
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X_mesh, Y_mesh = np.meshgrid(x, y)
    Z_plane = np.full(X_mesh.shape, z_threshold)

    # Add the plane as a surface
    fig.add_trace(go.Surface(x=X_mesh, y=Y_mesh, z=Z_plane, colorscale='gray', opacity=0.5, name='Plane'))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        title="3D Scatter Plot with Adjustable Plane and Elliptical Distribution",
    )

    st.plotly_chart(fig)  # Display the interactive 3D plot in Streamlit

# Streamlit App
def main():
    st.title("3D Scatter Plot with Adjustable Plane and Elliptical Distribution")
    st.write("This application shows a 3D scatter plot with an adjustable plane to divide data into two regions, based on an elliptical distribution.")

    # Sliders for elliptical parameters and plane height
    z_threshold = st.slider("Plane Height (Z-axis Threshold)", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
    semi_major = st.slider("Semi-Major Axis", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
    semi_minor = st.slider("Semi-Minor Axis", min_value=0.1, max_value=10.0, value=3.0, step=0.1)

    # Step 1: Generate Data in an elliptical-based shape
    np.random.seed(0)
    X = np.random.uniform(-10, 10, (600, 2))  # Generate 600 points uniformly distributed in a square
    Z = elliptical_function(X[:, 0], X[:, 1], semi_major, semi_minor)  # Use the elliptical function to create Z values
    X_new = np.c_[X, Z]  # Combine x1, x2 with elliptical-based Z values to form 3D data

    # Display the 3D plot with the adjustable plane
    plot_3d_elliptical_with_plane(X_new, z_threshold)  # Pass X_new and z_threshold as arguments

if __name__ == "__main__":
    main()

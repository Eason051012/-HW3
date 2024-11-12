# Import necessary libraries
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Step 2: Define Gaussian function to generate a cone-like distribution
def gaussian_function(x1, x2):
    return np.exp(-(x1**2 + x2**2) / 20)  # This function is a 2D Gaussian

# Step 5: Visualization function using Plotly
def plot_3d_cone_with_plane(X_new, z_threshold):
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
        title="3D Scatter Plot with Adjustable Plane",
    )

    st.plotly_chart(fig)  # Display the interactive 3D plot in Streamlit

# Streamlit App
def main():
    st.title("3D Scatter Plot with Adjustable Plane")
    st.write("This application shows a 3D scatter plot with an adjustable plane to divide data into two regions.")

    # Slider for z-axis threshold of the plane
    z_threshold = st.slider("Plane Height (Z-axis Threshold)", min_value=0.1, max_value=10.0, value=5.0, step=0.1)

    # Step 1: Generate Data in a cone-like shape
    np.random.seed(0)
    X = np.random.randn(600, 2) * 10  # Generate 600 points with normal distribution centered around (0,0)
    Z = gaussian_function(X[:, 0], X[:, 1]) * 10  # Scale the Gaussian to create a cone shape
    X_new = np.c_[X, Z]  # Combine x1, x2 with Gaussian result to form 3D data (cone shape)

    # Display the 3D plot with the adjustable plane
    plot_3d_cone_with_plane(X_new, z_threshold)  # Pass X_new and z_threshold as arguments

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform as tf

def sample_orientations(quat, num_samples=100, max_angle=np.radians(10)):
    """
    Sample orientations around the given quaternion by rotating around the local x-axis.
    
    :param quat: Original orientation as a (4,) numpy array (w, x, y, z).
    :param num_samples: Number of orientations to sample.
    :param max_angle: Maximum rotation angle in radians.
    :return: List of new quaternions.
    """
    sampled_quaternions = []
    for _ in range(num_samples):
        alpha = np.random.uniform(0, max_angle)  # Random angle within the max range
        beta = np.random.uniform(0, 2 * np.pi)  # Random direction in the circular plane

        # Compute local rotation axis (staying in the x-plane)
        axis = np.array([1, np.cos(beta), np.sin(beta)])
        axis /= np.linalg.norm(axis)  # Normalize the axis
        
        # Create a quaternion for the local rotation
        rot_quat = tf.Rotation.from_rotvec(alpha * axis).as_quat()

        # Apply rotation to original quaternion
        new_quat = tf.Rotation.from_quat(quat) * tf.Rotation.from_quat(rot_quat)
        print(np.linalg.norm(new_quat.as_quat()))
        sampled_quaternions.append(new_quat.as_quat())

    return sampled_quaternions

def visualize_orientations(quat, samples):
    """
    Visualizes sampled orientations by plotting the rotated x-axis in 3D space.
    
    :param quat: Original orientation quaternion (w, x, y, z).
    :param samples: List of sampled quaternions.
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # Original x-axis direction (unit vector along x)
    x_axis = np.array([1, 0, 0])

    # Apply sampled quaternions to the x-axis
    transformed_axes = [tf.Rotation.from_quat(q).apply(x_axis) for q in samples]

    # Convert to arrays for plotting
    transformed_axes = np.array(transformed_axes)
    ax.scatter(transformed_axes[:, 0], transformed_axes[:, 1], transformed_axes[:, 2], c='b', alpha=0.6, label="Sampled X-axes")

    # Plot the original x-axis
    original_x = tf.Rotation.from_quat(quat).apply(x_axis)
    ax.quiver(0, 0, 0, original_x[0], original_x[1], original_x[2], color='r', linewidth=2, label="Original X-axis")

    # Labels and view adjustments
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Sampled Orientations Around X-Axis")
    ax.legend()
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    plt.show()

# Example usage:
quat = np.array([1, 0, 0, 0])  # Identity quaternion (w, x, y, z)
samples = sample_orientations(quat, num_samples=200)
visualize_orientations(quat, samples)

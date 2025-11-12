# import json
# import matplotlib.pyplot as plt

# def plot_slam_results(filepath="slam_results.json"):
#     """
#     Reads the saved SLAM data and plots the trajectories and landmarks.
#     """
#     try:
#         with open(filepath, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Could not find '{filepath}'.")
#         print("Please run the simulation first to generate the file.")
#         return
#     except Exception as e:
#         print(f"Error loading JSON file: {e}")
#         return

#     # Extract data
#     true_path = data.get('true_robot_path', [])
#     ekf_path = data.get('ekf_robot_path', [])
#     true_lms = data.get('true_landmarks', [])
#     ekf_lms = data.get('ekf_landmarks', [])

#     if not true_path or not ekf_path:
#         print("Error: Data file is missing trajectory data.")
#         return

#     # --- Process data for plotting ---
#     # Unzip coordinate lists into separate x and y lists
#     true_x, true_y = zip(*true_path)
#     ekf_x, ekf_y = zip(*ekf_path)
    
#     # --- Create the Plot ---
#     plt.figure(figsize=(12, 9))
    
#     # Plot trajectories
#     plt.plot(true_x, true_y, color='red', linestyle='-', label='True Robot Path', alpha=0.8)
#     plt.plot(ekf_x, ekf_y, color='cyan', linestyle='--', label='EKF Estimated Path', linewidth=2)
    
#     # Plot landmarks (if they exist)
#     if true_lms:
#         true_lm_x, true_lm_y = zip(*true_lms)
#         plt.scatter(true_lm_x, true_lm_y, c='green', marker='o', s=150, 
#                     label='True Landmarks', edgecolors='black', zorder=5)
    
#     if ekf_lms:
#         ekf_lm_x, ekf_lm_y = zip(*ekf_lms)
#         plt.scatter(ekf_lm_x, ekf_lm_y, c='purple', marker='x', s=150, 
#                     label='EKF Estimated Landmarks', linewidth=3, zorder=5)

#     # --- Style the Plot ---
#     plt.title('EKF-SLAM Results: Trajectory and Landmarks')
#     plt.xlabel('X Position (pixels)')
#     plt.ylabel('Y Position (pixels)')
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.6)
    
#     # Use 'equal' axis scaling for a correct spatial representation
#     plt.axis('equal') 
    
#     # Invert Y-axis to match Pygame's coordinate system (0,0 at top-left)
#     plt.gca().invert_yaxis() 
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_slam_results()

import json
import matplotlib.pyplot as plt

def plot_slam_results(filepath="slam_results.json"):
    """
    Reads the saved SLAM data and plots the trajectories and landmarks
    on a single graph.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        print("Please run the simulation first to generate the file.")
        return
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Extract data
    true_path = data.get('true_robot_path', [])
    ekf_path = data.get('ekf_robot_path', [])
    true_lms = data.get('true_landmarks', [])
    ekf_lms = data.get('ekf_landmarks', [])

    if not true_path or not ekf_path:
        print("Error: Data file is missing trajectory data.")
        return

    # --- Process data for plotting ---
    true_x, true_y = zip(*true_path)
    ekf_x, ekf_y = zip(*ekf_path)
    
    # --- Create the Plot ---
    plt.figure(figsize=(12, 9))
    
    # Plot trajectories
    # !!! NOTE: The 'true_robot_path' will be invisible on this graph
    # because its data is clustered in a tiny 1x1 area.
    plt.plot(true_x, true_y, color='red', linestyle='-', label='True Robot Path (Not Visible)', alpha=0.8)
    plt.plot(ekf_x, ekf_y, color='cyan', linestyle='--', label='EKF Estimated Path', linewidth=2)
    
    # Plot landmarks (if they exist)
    if true_lms:
        true_lm_x, true_lm_y = zip(*true_lms)
        plt.scatter(true_lm_x, true_lm_y, c='green', marker='o', s=150, 
                    label='True Landmarks', edgecolors='black', zorder=5)
    
    if ekf_lms:
        ekf_lm_x, ekf_lm_y = zip(*ekf_lms)
        plt.scatter(ekf_lm_x, ekf_lm_y, c='purple', marker='x', s=150, 
                    label='EKF Estimated Landmarks', linewidth=3, zorder=5)

    # --- Style the Plot ---
    plt.title('EKF-SLAM Results: Trajectory and Landmarks')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Use 'equal' axis scaling for a correct spatial representation
    plt.axis('equal') 
    
    # Invert Y-axis to match Pygame's coordinate system (0,0 at top-left)
    plt.gca().invert_yaxis() 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_slam_results()
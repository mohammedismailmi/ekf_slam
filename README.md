# Camera-Based EKF-SLAM Implementation

## 🎯 Project Overview

This project implements a complete Camera-Based SLAM (Simultaneous Localization and Mapping) system using Extended Kalman Filter (EKF) in Python. The system demonstrates:

1. **Feature-Based SLAM**: Landmark detection and tracking
2. **Grid-Based Occupancy Mapping**: Environmental mapping
3. **Camera Sensor Simulation**: Monocular camera with FOV
4. **Real-time Visualization**: Pygame rendering with Flask web controls

## 📋 Requirements

```txt
pygame==2.5.2
numpy==1.24.3
flask==3.0.0
flask-cors==4.0.0
```

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install pygame numpy flask flask-cors
```

### Step 2: Create Directory Structure

```
ekf_slam_project/
│
├── slam_main.py          # Main application file
├── templates/
│   └── index.html        # Flask web interface
└── README.md             # This file
```

### Step 3: Save Files

1. Save the main Python code as `slam_main.py`
2. Create a `templates` folder
3. Save the HTML code as `templates/index.html`

### Step 4: Run the Application

```bash
python slam_main.py
```

This will:
- Start the Flask server on `http://localhost:5000`
- Open a Pygame window showing the simulation
- Display real-time SLAM visualization

## 🎮 Usage

### Web Interface Controls

Open your browser and navigate to: `http://localhost:5000`

**Movement Controls:**
- **Forward** (↑): Move robot forward
- **Backward** (↓): Move robot backward
- **Left** (←): Rotate counter-clockwise
- **Right** (→): Rotate clockwise
- **STOP**: Stop all movement

**Keyboard Controls:**
- Arrow keys or WASD for movement
- Spacebar for pause/resume

**View Options:**
- **Camera FOV**: Toggle field of view visualization
- **Occupancy Grid**: Toggle grid-based mapping
- **Feature Landmarks**: Toggle landmark visibility
- **Pause/Resume**: Pause or resume simulation

### Pygame Window

The main visualization window shows:
- **Red Circle**: Actual robot position
- **Cyan Circle**: EKF estimated robot position
- **Green Circles**: True landmark positions
- **Purple Circles**: EKF estimated landmarks
- **Yellow Cone**: Camera field of view
- **Blue Trail**: Robot's path history
- **Gray Grid**: Occupancy map (dark = obstacles, light = free)
- **Dark Gray Rectangles**: Walls and obstacles

## 🔬 Technical Components

### 1. Camera Sensor (`Camera` class)
- **Field of View**: 90° cone
- **Max Range**: 300 pixels
- **Ray Casting**: 15 sampling rays
- Simulates landmark detection with measurement noise

### 2. EKF-SLAM (`EKF_SLAM` class)

**State Vector:**
```
[robot_x, robot_y, robot_theta, landmark1_x, landmark1_y, ...]
```

**Prediction Step:**
- Motion model: Differential drive kinematics
- Process noise covariance: Q matrix
- State propagation with Jacobian

**Update Step:**
- Measurement model: Distance and bearing to landmarks
- Measurement noise covariance: R matrix
- Kalman gain computation
- State and covariance update

### 3. Occupancy Grid (`OccupancyGrid` class)
- **Cell Size**: 20x20 pixels
- **Probability Updates**: Bayesian inference
- **Ray Casting**: Updates free space and obstacles
- **Visualization**: Grayscale probability map

### 4. Robot (`Robot` class)
- **Kinematics**: Differential drive model
- **Control**: Linear (v) and angular (w) velocities
- **Trail Visualization**: Path history tracking

## 📊 Key Algorithms

### Motion Model
```
x_new = x + (v/w) * (sin(θ + w*dt) - sin(θ))
y_new = y + (v/w) * (-cos(θ + w*dt) + cos(θ))
θ_new = θ + w*dt
```

### Measurement Model
```
distance = sqrt((landmark_x - robot_x)² + (landmark_y - robot_y)²)
bearing = atan2(landmark_y - robot_y, landmark_x - robot_x) - robot_θ
```

### EKF Update Equations
```
Innovation: y = z - h(x)
Kalman Gain: K = P * H^T * (H * P * H^T + R)^(-1)
State Update: x = x + K * y
Covariance Update: P = (I - K * H) * P
```

## 🎓 Educational Features

### 1. Feature-Based SLAM
- Landmarks represented as point features
- Data association by landmark ID
- Uncertainty ellipses showing covariance

### 2. Occupancy Grid Mapping
- Probabilistic grid representation
- Real-time updates with camera rays
- Free space vs. obstacle classification

### 3. Sensor Modeling
- Camera field of view constraints
- Range limitations
- Gaussian measurement noise

### 4. State Estimation
- Extended Kalman Filter
- Uncertainty propagation
- Covariance visualization

## 🛠️ Customization

### Adjust Camera Parameters
```python
self.camera = Camera(fov=90, max_range=300)
```

### Modify Process Noise
```python
self.Q = np.diag([0.1, 0.1, 0.05])  # x, y, theta noise
```

### Change Measurement Noise
```python
self.R = np.diag([4.0, 0.04])  # distance, bearing noise
```

### Add More Landmarks
```python
self.true_landmarks = {
    0: (150, 150),
    1: (650, 150),
    # Add more...
}
```

### Modify Room Layout
```python
self.walls = [
    pygame.Rect(50, 50, 700, 10),  # Top wall
    # Add more obstacles...
]
```

## 📈 Performance Metrics

The simulation tracks:
- Robot position error (true vs. estimated)
- Landmark position uncertainty
- Number of landmarks detected
- Occupancy grid accuracy

## 🐛 Troubleshooting

### Pygame Window Doesn't Open
- Ensure pygame is installed: `pip install pygame`
- Check for display driver issues

### Flask Server Won't Start
- Port 5000 might be in use
- Modify port in `app.run(port=5000)`

### Robot Moves Too Fast/Slow
- Adjust velocity in control commands
- Modify `v = 50` to desired speed

### Landmarks Not Detected
- Check camera FOV and range
- Verify landmark positions are within range
- Ensure robot is oriented towards landmarks

## 📚 Theory Background

### Extended Kalman Filter (EKF)
The EKF linearizes the nonlinear motion and measurement models using first-order Taylor expansion (Jacobian matrices).

### SLAM Problem
Estimate robot pose **and** landmark positions simultaneously from sensor measurements, accounting for:
- Process noise (motion uncertainty)
- Measurement noise (sensor uncertainty)
- Data association (which landmark is observed)

### Camera-Based SLAM
Unlike LiDAR SLAM, camera-based systems:
- Extract visual features (landmarks)
- Have limited field of view
- Provide bearing and distance measurements
- Require feature tracking across frames

## 🎯 Learning Objectives

1. Understand EKF prediction and update steps
2. Implement motion models for mobile robots
3. Handle sensor measurements with uncertainty
4. Visualize covariance and uncertainty
5. Build occupancy grids from sensor data
6. Integrate multiple mapping approaches

## 📝 Assignment Tips

1. **Documentation**: Add comments explaining EKF math
2. **Experiments**: Try different noise parameters
3. **Analysis**: Compare true vs. estimated positions
4. **Extensions**: Add loop closure detection
5. **Visualization**: Plot error metrics over time

## 🌟 Advanced Extensions

- Multi-robot SLAM
- Loop closure detection
- Graph-based SLAM backend
- Visual odometry integration
- 3D SLAM with depth camera
- Real robot hardware integration

## 📄 License

Educational project for robotics course.

## 👥 Contributors

- USN School of Computer Science and Engineering
- Minor in Robotics and Industrial Automation
- Course: CS3112 - Perception & Localization

## 📞 Support

For issues or questions about the implementation:
1. Review the code comments
2. Check the troubleshooting section
3. Refer to course materials on EKF-SLAM
4. Consult with instructors

---

**Happy Mapping! 🗺️🤖**
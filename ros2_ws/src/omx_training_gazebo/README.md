# OMX training Gazebo arena

This package starts a Gazebo Sim 6 world whose clear interior measures
1.75 m by 1.13 m. Four 0.30 m walls surround the floor, and the TurtleBot3
Waffle Pi with OpenMANIPULATOR-X is spawned at the centre by default.

The robot description is not duplicated here. Source the overlays in this
order so `turtlebot3_manipulation_description` resolves to the checked source
in `/home/ktj/omx_turtle_ws/src`:

```bash
source /opt/ros/humble/setup.bash
source /home/ktj/turtlebot3_ws/install/setup.bash
source /home/ktj/omx_turtle_ws/install/setup.bash
source /home/ktj/omx_train_ws/ros2_ws/install/setup.bash
ros2 launch omx_training_gazebo walled_arena.launch.py
```

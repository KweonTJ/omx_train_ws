"""Spawn the OMX TurtleBot in the 1.75 m by 1.13 m walled arena."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Start Gazebo Sim and spawn the source-overlay robot at arena center."""
    world = PathJoinSubstitution([
        FindPackageShare('omx_training_gazebo'),
        'worlds',
        'walled_arena_1_75x1_13.sdf',
    ])
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw = LaunchConfiguration('yaw')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_manipulation_gazebo'),
                'launch',
                'gazebo.launch.py',
            ])
        ]),
        launch_arguments={
            'world': world,
            'gz_args': ['-r -v 3 ', world],
            'start_rviz': 'false',
            'start_depth_camera': 'false',
            'use_sim': 'true',
            'move_to_stay_pose': 'false',
            'x_pose': x_pose,
            'y_pose': y_pose,
            'z_pose': '0.01',
            'roll': '0.0',
            'pitch': '0.0',
            'yaw': yaw,
        }.items(),
    )

    frame_camera = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ign',
                    'service',
                    '-s',
                    '/gui/move_to/pose',
                    '--reqtype',
                    'ignition.msgs.GUICamera',
                    '--reptype',
                    'ignition.msgs.Boolean',
                    '--timeout',
                    '3000',
                    '--req',
                    (
                        'pose: {position: {x: -1.6, y: -1.6, z: 1.6}, '
                        'orientation: {x: -0.1039, y: 0.2509, '
                        'z: 0.3683, w: 0.8890}}, '
                        'view_controller: "orbit", '
                        'projection_type: "perspective"'
                    ),
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'x_pose',
            default_value='0.0',
            description='Robot X position inside the arena in metres.',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.0',
            description='Robot Y position inside the arena in metres.',
        ),
        DeclareLaunchArgument(
            'yaw',
            default_value='0.0',
            description='Robot yaw at spawn in radians.',
        ),
        gazebo,
        frame_camera,
    ])

#!/usr/bin/env python

# wave.py: "Wave" the fetch gripper
import rospy
from moveit_msgs.msg import MoveItErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import actionlib
import control_msgs.msg
from tf.transformations import *

# Note: fetch_moveit_config move_group.launch must be running
# Safety!: Do NOT run this script near people or objects.
# Safety!: There is NO perception.
#          The ONLY objects the collision detection software is aware
#          of are itself & the floor.
CLOSED_POS = 0.0  # The position for a fully-closed gripper (meters).
OPENED_POS = 0.10  # The position for a fully-open gripper (meters).
ACTION_SERVER = 'gripper_controller/gripper_action'
MIN_EFFORT = 35  # Min grasp force, in Newtons
MAX_EFFORT = 200  # Max grasp force, in Newtons

latest_pose = None

def wait_for_keypress():
    print("Press any key to continue...")
    raw_input()  # Waits here for the user to press a key
    print("Keypress detected. Continuing...")

def check_result(result):
    if result:
        # Checking the MoveItErrorCode
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            rospy.loginfo("Hello there!")
        else:
            # If you get to this point please search for:
            # moveit_msgs/MoveItErrorCodes.msg
            rospy.logerr("Arm goal in state: %s",
                            move_group.get_move_action().get_state())
    else:
        rospy.logerr("MoveIt! failure no result returned.")

def callback(data):
    global latest_pose 
    latest_pose = data
    #print(latest_pose)



def gripper_open():
        """Opens the gripper.
        """
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = OPENED_POS
        _client.send_goal_and_wait(goal, rospy.Duration(10))

def gripper_close(width=0.0, max_effort=MAX_EFFORT):
    """Closes the gripper.

    Args:
        width: The target gripper width, in meters. (Might need to tune to
            make sure the gripper won't damage itself or whatever it's
            gripping.)
        max_effort: The maximum effort, in Newtons, to use. Note that this
            should not be less than 35N, or else the gripper may not close.
    """
    assert CLOSED_POS <= width <= OPENED_POS
    goal = control_msgs.msg.GripperCommandGoal()
    goal.command.position = width
    goal.command.max_effort = max_effort
    _client.send_goal_and_wait(goal, rospy.Duration(10))

if __name__ == '__main__':
    

    rospy.init_node('listener_node', anonymous=True)
    rospy.Subscriber("pose_tra", PoseStamped, callback)
    #connect to action server
    _client = actionlib.SimpleActionClient(ACTION_SERVER, control_msgs.msg.GripperCommandAction)
    _client.wait_for_server(rospy.Duration(10))
    # Create move group interface for a fetch robot
    
    move_group = MoveGroupInterface("arm", "base_link")

    # Define ground plane
    # This creates objects in the planning scene that mimic the ground
    # If these were not in place gripper could hit the ground
    planning_scene = PlanningSceneInterface("base_link")
    planning_scene.removeCollisionObject("my_front_ground")
    planning_scene.removeCollisionObject("my_table")

    planning_scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
    # planning_scene.addCube("my_table", 1, 0.8, 0.0, 0.35)
    planning_scene.addCube("my_rod", 1, 1.50, 0.0, 1.9)



    # This is the wrist link not the gripper itself
    gripper_frame = 'gripper_link'
    # Position and rotation of two "wave end poses"
    # home_pose = Pose(Point(0.049, 0.730, 1.0),
    #                       Quaternion(-0.707, -0.0, 0.707, 0.0))
    # prep_pose = Pose(Point(0.523, -0.108, 1.0),
    #                       Quaternion(-0.707, -0.0, 0.707, 0.0))

    #camera frame home and prep
    home_pose = Pose(Point(0.0, 0.63, -0.5),
                          Quaternion(-0.707, -0.0, 0.707, 0.0))
    prep_pose = Pose(Point(0.63, 0.0, -0.15),
                          Quaternion(-0.707, -0.1, 0.0, 0.0))
    
                        #   Quaternion(0.614, 0.409, -0.545, 0.396))
    z_rot = 0
    q_rot = quaternion_from_euler(0, 0, z_rot)

    # Construct a "pose_stamped" message as required by moveToPose
    gripper_pose_stamped = PoseStamped()
    gripper_pose_stamped.header.frame_id = 'head_camera_rgb_frame'    

    # Main loop
    #head on grip Quaternion(-0.707, 0, 0, 0))
    #sideways from right grip Quaternion(-0.707, -0.707, 0, 0))
    #sideways from left grip Quaternion(-0.707, 0.707, 0, 0))
    test_latest_poses = [Pose(Point(0.8, 0.41, -0.21), Quaternion(-0.707, 0.0, 0.0, 0.0))]
    
    ctr = 0
    while not rospy.is_shutdown():
        wait_for_keypress()
        # Maybe do some more work here or exit


        
        # Finish building the Pose_stamped message
        # If the message stamp is not current it could be ignored
        # gripper_pose_stamped.header.stamp = rospy.Time.now()
        # start move
        gripper_pose_stamped.pose = home_pose
        move_group.moveToPose(gripper_pose_stamped, gripper_frame)
        # gripper_pose_stamped.pose = rot_test
        move_group.moveToPose(gripper_pose_stamped, gripper_frame)

        result = move_group.get_move_action().get_result()
        print("at start")
        # wait_for_keypress()

        gripper_open()
        rospy.sleep(1)
        gripper_close()
        rospy.sleep(1)
        gripper_open()
        rospy.sleep(1)
        print("init gripper")
        wait_for_keypress()


        gripper_pose_stamped.pose = prep_pose
        move_group.moveToPose(gripper_pose_stamped, gripper_frame)
        result = move_group.get_move_action().get_result()
        print("at prep")
        
        test_pose_stamped = PoseStamped()
        test_pose_stamped.header.frame_id = 'base_link'
        test_pose_stamped.header.stamp = rospy.Time.now()
        test_pose_stamped.pose = test_latest_poses[ctr]

        pre_pick_pose = test_pose_stamped
        pre_pick_pose.pose.position.z = pre_pick_pose.pose.position.z + .10
        move_group.moveToPose(pre_pick_pose, gripper_frame,0.05)
        result = move_group.get_move_action().get_result()
        print("at prepick")
        # wait_for_keypress()

        pick_pose = test_pose_stamped
        move_group.moveToPose(pick_pose, gripper_frame)
        result = move_group.get_move_action().get_result()
        print("at pick")
        ctr += 1
        # wait_for_keypress()


        gripper_close()
        print("close gripper")
        rospy.sleep(1)
        gripper_open()
        # wait_for_keypress()

        post_pick_pose = test_pose_stamped
        pre_pick_pose.pose.position.x = pre_pick_pose.pose.position.z - .45
        move_group.moveToPose(pre_pick_pose, gripper_frame,0.05)
        result = move_group.get_move_action().get_result()
        print("at postpick")


        # move_group.moveToPose(pre_pick_pose, gripper_frame)
        # result = move_group.get_move_action().get_result()
        # print("at prepick")
        # wait_for_keypress()

        gripper_pose_stamped.pose = home_pose
        move_group.moveToPose(gripper_pose_stamped, gripper_frame)
        result = move_group.get_move_action().get_result()
        print("at home")
        gripper_open()
        print("open gripper")
        rospy.sleep(1)

        # wait_for_keypress()
        
        
            

    # This stops all arm movement goals
    # It should be called when a program is exiting so movement stops
    move_group.get_move_action().cancel_all_goals()


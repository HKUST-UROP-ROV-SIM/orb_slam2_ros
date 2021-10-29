# Time when SLAM processing completed
std_msgs/Header header

# Image time(s)
builtin_interfaces/Time left_stamp
builtin_interfaces/Time right_stamp

int32 STATE_SYSTEM_NOT_READY=-1     # LOADING ORB VOCABULARY. PLEASE WAIT...
int32 STATE_NO_IMAGES_YET=0         # WAITING FOR IMAGES
int32 STATE_NOT_INITIALIZED=1       # TRYING TO INITIALIZE
int32 STATE_OK=2                    # if !mbOnlyTracking SLAM MODE else LOCALIZATION
int32 STATE_LOST=3                  # TRACK LOST. TRYING TO RELOCALIZE
int32 state                         # mState
bool only_tracking                  # mbOnlyTracking

int32 keyframes                     # mpMap->KeyFramesInMap()
int32 map_points                    # mpMap->MapPointsInMap()
int32 tracked                       # mnTracked
int32 tracked_v0                    # mnTrackedV0

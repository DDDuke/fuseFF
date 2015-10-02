PROJECT(daneFace)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
SET(CMAKE_PREFIX_PATH "/home/vorse/bin/opencv2.4.11")
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE(../FingerFace/saveMaster lbp.cpp ../FingerFace/saveMaster.cpp)
ADD_EXECUTABLE(../FingerFace/testMaster lbp.cpp ../FingerFace/testMaster.cpp)
# ADD_EXECUTABLE(../FingerFace/lbp2 lbp.cpp main.cpp)
# ADD_EXECUTABLE(kak alignLBPHVisual.cpp)

TARGET_LINK_LIBRARIES(../FingerFace/saveMaster ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(../FingerFace/testMaster ${OpenCV_LIBS})

PROJECT(danelbp)
SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
SET(CMAKE_PREFIX_PATH "/home/vorse/bin/opencv2.4.11")
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE(lbp lbp.cpp main.cpp)

TARGET_LINK_LIBRARIES(lbp ${OpenCV_LIBS})


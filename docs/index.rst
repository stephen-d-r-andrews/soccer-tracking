.. soccer-tracking documentation master file, created by
   sphinx-quickstart on Sun Oct 17 09:49:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to soccer-tracking's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Usage
=====

Place the video to be analyzed in a folder called videos. Create a folder called output for the output video. The yolov3 weights and configuration are assumed to be in a folder called yolo-coco. 

To run:

python tracking.py --input videos/inputvideo.mp4 --output output/outputvideo.avi --anchors videos/inputvideo.json --yolo yolo-coco

The output is placed in the file outputvideo.avi in the output folder. As the annotater marks the video the anchors are stored in vidoes/inputvideo.json. This file is created if it doesn't already exist. 

Annotations
===========

The job of the annotater is to mark 4 anchors on a subset of the frames that correspond to known points on the field. The rest of the script then uses these anchors to calculate the postions of the players. In the current script 1 out of every 6 frames is presented to the annotator who has to choose a rectangle on the field. Many of the example videos are taken on a football field and so these rectangles are easy to find based on the yard lines. There are 3 ways to specify the rectangle.

* Press the 's' key (for start). This tells the script that the annotater wants a new rectangle. The annotater then clicks 4 times on the screen to rotate through the corner points. After each click the annotater is prompted in the terminal to write the corresponding coordinates on the field. After specifying the 4th point the annotater presses the 'd' key (for done).
* Press the 'm' key (for modify). This tells the script that the annotater wants to use the same rectangle on the field as in the last annotated. However, this time the points on the screen on the field have moved. (This is visible because the previous screen points will be still marked.) All the annotater needs to do is click the new points on the screen that correspond to the rectangle. When done the annotater presses 'd'.
* Press the 'd' key (for done). This tells the script that both the screen points and field points have not changed since the last annotated frame (i.e. the camera has not moved). The script can then progress to the next frame.

Correcting errors
=================

The script stores the anchor points (both the screen coordinates as well as the field coordinates) in the file specified by the --anchors flag. Sometimes the annotater makes a mistake (e.g. accidentally clicking in the wrong place). If this happens the annotater just needs to kill the script and edit the anchors file manually to remove the information for the most recent frame. When the script is restarted it creates the output video for the frames that are already annotated, and only prompts for input when it reaches the new point in the input video.

Annotating on a football field
==============================

All the example videos were taken on a high school football field with the goals placed behind each endzone and the soccer sidelines placed 20feet outside the football sidelines. The length of the field is 360feet (including the 2 endzones) and the width is 200feet. The two lines of hashmarks are at 73.33feet and 126.67feet across the field. For the example videos it was most convienent to use anchor points on one of the 10-yard lines, together with the near sideline and the near line of hashmarks. For example, to use anchors on the 30 and 40 yard lines, the field anchor coordinates in feet are (120, 180), (120, 126.67), (150, 126.67) and (150, 180).

Output video
============

The script uses YOLO to detect players in the video and then uses the anchors to map the coordinates on the screen to coordinates on the field. Each player is associated with the nearest play in the previous annotated frame. From this the script calculates the speed of the player (in feet/sec) and marks the speed above the player. The color of the text is based on the speed so fast players are visible in red. In the top right of the output frame there is a representation of the field with all the detected players marked.

YOLO is not perfect and so we sometimes miss players. We only calculate the speed for a player if they have been continuously detected for at least second. If the script detects a player but not for long enough to calculate the speed, the player is marked in black on the field representation. If the speed is calculated then the color of the marking corresponds to that speed. 

.. raw:: html

  <iframe width="560" height="315" src="https://www.youtube.com/embed/hmcKE-3u0LE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



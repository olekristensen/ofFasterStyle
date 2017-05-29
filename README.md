# ofFasterStyle

This is an openFrameworks realtime c++ implementation of [Logan Engstrom's fast-style-transfer](https://github.com/lengstrom/fast-style-transfer), using the ir image from a kinect as input to the style transfer net. The models are loaded as binary protobuffers using an ugly varhack to preserve the variables in the graph, as explained in [ofxMSATensorFlow](https://github.com/memo/ofxMSATensorFlow).

## The trained models are in the release
https://github.com/olekristensen/ofFasterStyle/releases/tag/v0.8

## Training you own models
Look at [my fork of lengstrom's project](https://github.com/olekristensen/fast-style-transfer) and follow the original instructions, then style an image with *exactly the same pixel dimensions* as the input pixels you want to feed from your ofFasterStyle project and our fork of fast-style-transfer will have saved a of.pb file for you in the model checkpoint dir. 

## Thanks to
Lilia Amundsen, the master student who initiated this project

DEIC, the super computing guys who let us grind our teeth with this little project on Abacus 2.0 

Logan Engstrom, whose tensorflow implementation of a fast style transfer network provided a good starting point for learning.

Memo, who always is ahead of the curve in the openFrameworks community - super awsome addon - really!

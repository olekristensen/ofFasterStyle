#pragma once

#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp{

	public:

        void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
        bool loadGraph(string path);

        int currentModel;
        void loadNextGraph();

        msa::tf::Session_ptr session;                   // shared pointer to tensorflow session
        msa::tf::GraphDef_ptr graph_def;                // shared pointer to tensorflow graph definition
        vector<tensorflow::Tensor> output_tensors;      // stores all output tensors
        ofVideoGrabber grab;

        ofImage cam_img;
        ofFloatImage cam_float_img;
        ofFloatImage styled_float_img;
        ofImage styled_img;

        const int img_width = 960; //1280*3/4;
        const int img_height = 540; //720*3/4;

        std::vector<tensorflow::int64> image_dims;      // dimensions of input image. must have 3 values: { w, h, c (number of channels) }
        std::vector<tensorflow::int64> itensor_dims;    // dimensions of tensor to pass to tensorflow (could be same as image, could be shaped different, depends on network

        tensorflow::Tensor  image_tensor;               // stores input image as tensor

        int num_elements = 1;

        bool drawGui = false;

        const string modelDir = "final_models";
        ofDirectory dir;
        vector<pair<string,int>> models;

        ofxPanel gui;
        ofxIntSlider noiseAmount;
        ofxFloatSlider secondsInterval;
        double lastTimeLoadModel;
        ofxLabel fps;

};

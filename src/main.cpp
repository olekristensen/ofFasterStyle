#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main( ){
    /*
    ofGLFWWindowSettings settings;
       settings.monitor = 1;
       settings.decorated = false;
       settings.setPosition(ofVec2f(1920,0));
       settings.width = 1280;
       settings.height = 800;
       settings.windowMode = OF_WINDOW;
       auto win1 = ofCreateWindow(settings);

       ofRunApp(win1, std::make_shared<ofApp>());

       ofRunMainLoop();*/

    ofSetupOpenGL(1280,800, OF_WINDOW);			// <-------- setup the GL context

        // this kicks off the running of my app
        // can be OF_WINDOW or OF_FULLSCREEN
        // pass in width and height too:
        ofRunApp( new ofApp());

}

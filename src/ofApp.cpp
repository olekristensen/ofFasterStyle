 #include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){

    dir.listDir(modelDir);
    dir.sort(); // in linux the file system doesn't return file lists ordered in alphabetical order

    // you can now iterate through the files and load them into the ofImage vector
    for(ofFile file : dir.getFiles()){
        if(file.isDirectory()){
            ofDirectory subDir;
            subDir.listDir(file.getAbsolutePath());
            for(ofFile f : subDir.getFiles()){
                if(f.getFileName() == "of.pb"){
                    models.push_back({f.getAbsolutePath(), 0});
                }

            }
        }
    }

    if(models.size() == 0) ofExit(-1);

    for(auto m : models){
        if(loadGraph(m.first)) break;
    }

    gui.setup(); //GUI  most of the time you don't need a name
    gui.add(noiseAmount.setup("noise", 0, 0, 255));
    gui.add(secondsInterval.setup("interval (sec)",1*60, 3, 20*60));
    gui.add(fps.setup("FPS", ""));

    // set up camera
    grab.initGrabber(img_width, img_height);

    cam_float_img.allocate(img_height,img_width,OF_IMAGE_COLOR);
    styled_float_img.allocate(img_height,img_width,OF_IMAGE_COLOR);

    image_dims = { img_width, img_height, 3 };
    itensor_dims = { 1, img_width , img_height, 3 };

    for(auto i : image_dims) num_elements *= i;

    image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(itensor_dims));

    ofSetLogLevel(OF_LOG_ERROR);
}

//--------------------------------------------------------------
void ofApp::update(){

    if(lastTimeLoadModel + secondsInterval < ofGetElapsedTimef()){
        lastTimeLoadModel = ofGetElapsedTimef();
        loadNextGraph();
    }

    fps = ofToString(ofGetFrameRate());

    grab.update();

    if (grab.isFrameNew()) {

        cam_img.setFromPixels(grab.getPixels());

        if(noiseAmount > 0.0){

            auto pix_data = cam_img.getPixels().getData();
            if(!pix_data) {
                ofLogError() << "Could not classify. pixel data is NULL";
            }else{
                for(int i=0; i<num_elements; i+=2){
                    if(pix_data[i] > 127){
                        pix_data[i] -= random()%noiseAmount;
                    }else {
                        pix_data[i] += random()%noiseAmount;
                    }
                }
            }
            cam_img.update();
        }

        cam_img.rotate90(3);

        // convert to float
        ofFloatPixels fpix = cam_img.getPixels();

        // set number of channels
        fpix.setNumChannels(3);

        cam_float_img.setFromPixels(fpix);

        if(session) {

            using namespace tensorflow;
            using namespace ops;

            ofLog() << num_elements;

            // copy data from image into tensorflow's Tensor class
            msa::tf::image_to_tensor(cam_float_img, image_tensor);

            ofLog() << image_tensor.DebugString();

            /*
            /// \brief Runs the graph with the provided input tensors and fills
            /// `outputs` for the endpoints specified in `output_tensor_names`.
            /// Runs to but does not return Tensors for the nodes in
            /// `target_node_names`.
            ///
            /// The order of tensors in `outputs` will match the order provided
            /// by `output_tensor_names`.
            ///
            /// If `Run` returns `OK()`, then `outputs->size()` will be equal to
            /// `output_tensor_names.size()`.  If `Run` does not return `OK()`, the
            /// state of `outputs` is undefined.
            ///
            /// REQUIRES: The name of each Tensor of the input or output must
            /// match a "Tensor endpoint" in the `GraphDef` passed to `Create()`.
            ///
            /// REQUIRES: At least one of `output_tensor_names` and
            /// `target_node_names` must be non-empty.
            ///
            /// REQUIRES: outputs is not nullptr if `output_tensor_names` is non-empty.
            virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
                               const std::vector<string>& output_tensor_names,
                               const std::vector<string>& target_node_names,
                               std::vector<Tensor>* outputs) = 0;

*/

            Status s = session->Run({ { "img_placeholder", image_tensor} } , { "add_37" }, {  }, &output_tensors);

            if(!s.ok()) {
                ofLogError() << "Error running network for styling";
                ofLogError() << s.error_message();
            }

            auto t_img_output = output_tensors[0];

            ofLog() << t_img_output.DebugString();

            msa::tf::tensor_to_image(t_img_output, styled_float_img);

            float* pix_data = styled_float_img.getPixels().getData();
            if(!pix_data) {
                ofLogError() << "Could not classify. pixel data is NULL";
            } else {
                for(int i=0; i<num_elements; i++) pix_data[i] = (pix_data[i] / 255.0f);
            }

            // convert to int
            ofPixels ipix = styled_float_img.getPixels();

            // set number of channels
            ipix.setNumChannels(3);

            styled_img.setFromPixels(ipix);
            styled_img.rotate90(3);

        }

    }

}

//--------------------------------------------------------------
void ofApp::draw(){
    styled_img.draw(-64,-64,ofGetWidth()+128, ofGetHeight()+128);
    if(drawGui){
        cam_float_img.draw(ofGetWidth()-(20+(cam_float_img.getWidth()/4)),20,cam_float_img.getWidth()/4,cam_float_img.getHeight()/4);
        gui.draw();
    }

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    if (key == 'f') ofToggleFullscreen();
    if (key == 'd') drawGui = !drawGui;
    if (key == 'l') loadNextGraph();
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}

//--------------------------------------------------------------
void ofApp::loadNextGraph(){
    loadGraph(models[currentModel++].first);
    currentModel%=models.size();
}

bool ofApp::loadGraph(string path){
    ofLogNotice() << "trying to load from: " << path;
    // Load graph (i.e. trained model) we exported from python, and initialize session
    tensorflow::ConfigProto soft_config;
    soft_config.set_allow_soft_placement(true);
    soft_config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::SessionOptions session_opts;
    session_opts.config = soft_config;

    // Load graph (i.e. trained model), return if error
    graph_def = msa::tf::load_graph_def(path);
    // ofLogNotice() << graph_def->DebugString();
    if(!graph_def) {
        ofLogError() << "Error loading graph " << path;
        return false;
    }
    // Initialize tensorflow session with graph, return if error
    session = msa::tf::create_session_with_graph(graph_def, "/gpu:0", session_opts);
    if(!session){
        ofLogError() << "Error creating session";
        return false;
    }

    // will store names of constant units
    std::vector<string> names;

    vector<tensorflow::Tensor> output_hack_tensors;      // stores all output tensors

    int node_count = graph_def->node_size();
    ofLogNotice() << "Classifier::hack_variables - " << node_count << " nodes in graph";

    // iterate all nodes
    for(int i=0; i<node_count; i++) {
        auto n = graph_def->node(i);
        ofLogNotice() << i << ":" << n.name(); // << n.DebugString();

        // if name contains var_hack, add to vector
        if(n.name().find("_VARHACK") != std::string::npos) {
            names.push_back(n.name());
            ofLogNotice() << "......bang";
        }
    }

    // run the network inputting the names of the constant variables we want to run
    if(!session->Run({}, names, {}, &output_hack_tensors).ok()) {
        ofLogError() << "Error running network for weights and biases variable hack";
        return false;
    }

    return true;

}

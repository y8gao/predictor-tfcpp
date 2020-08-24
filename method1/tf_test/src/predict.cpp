#include <iostream>
#include <string>
#include <chrono>
#include <utility>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

#define INPUT_LAYER_NAME    "input layer name"
#define OUTPUT_LAYER_NAME   "output layer name"

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path of model>" << std::endl;
        return 1;
    }
    auto model_path = argv[1];
    // load tensorflow model
    Session *session;
    std::cout << "[+] Start initalize session" << std::endl;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << "[-] Error: " << status.ToString() << std::endl;
        return 1;
    }
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
    if (!status.ok()) {
        std::cout << "[-] Error: " << status.ToString() << std::endl;
        return 1;
    }
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << "[-] Error: " << status.ToString() << std::endl;
        return 1;
    }
    std::cout << "[+] Loading tensorflow model, " << model_path << ", succeeded." << std::endl;


    /////////////////////////////
    // input data
    float data[] = {
        4, 2, 0, 0, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 1, 1, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 1, 1, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 1, 1, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 0, 0, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 0, 0, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 1, 1, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 1, 1, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 1, 1, 15, 2, 4, 2, 3, 3, 15, 2,
        4, 2, 0, 0, 15, 2, 4, 2, 3, 3, 15, 2,
    };
    std::vector<float> mydata(data, data+sizeof(data)/sizeof(float));


    // define tensor's type/shape, then copy data into it
    Tensor tensor(DT_FLOAT, TensorShape({1,10,12}));
    copy_n(mydata.begin(), 120, tensor.flat<float>().data());


    // define output for model
    std::vector<Tensor> outputs;

    // do inference
    status = session->Run({{INPUT_LAYER_NAME, tensor}}, {OUTPUT_LAYER_NAME}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "[-] Error: " << status.ToString() << std::endl;
        return 1;
    }
    //get the final label by max probablity
    Tensor t = outputs[0];                   // Fetch the first tensor
    std::cout << "[+] Output: " << t.DebugString() << std::endl;
    int ndim = t.shape().dims();             // Get the dimension of the tensor
    auto results =  t.tensor<float, 2>();
    std::cout << "[+] \tPredicted result: " << results(0,0) << std::endl;

    // bentchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i=0; i <= 1000; i++) {
        status = session->Run({{INPUT_LAYER_NAME, tensor}}, {OUTPUT_LAYER_NAME}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "[-] Error: " << status.ToString() << std::endl;
            return 1;
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    auto dura = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    std::cout << "[+] Bentchmark(1000):" << std::endl
        << "\tDuration: " << dura/1000 << " us" << std::endl
        << "\tAverage:"<< dura / 1000 / 1000 << " us"<< std::endl;
}

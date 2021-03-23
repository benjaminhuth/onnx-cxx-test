#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

#include <core/session/onnxruntime_cxx_api.h>

#include <Eigen/Core>

#include "OnnxModel.hpp"


int main(int argc, char **argv) 
{    
    if( argc != 2 )
    {
        std::cout << "Usage: " << argv[0] << " <path to onnx file>" << std::endl;
        return 1;
    }
    
    std::string path(argv[1]);
    
    if( !std::filesystem::exists(path) )
    {
        std::cout << "Path to onnx file '" << path << "' does not exist" << std::endl;
        return 1;
    }
    
    // Onnx Environment and options
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark_a");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    Ort::AllocatorWithDefaultOptions allocator;        
    
    std::vector batch_sizes = { 1ul, 2ul, 4ul, 8ul, 16ul, 32ul, 64ul, 128ul, 256ul };
    
    for(auto N : batch_sizes)
    {
        using MyModel = OnnxModel<OnnxInputs<5>, OnnxOutputs<5>>;
        MyModel model(env, opts, path);
        
        MyModel::InVectorTuple input_batch;
        std::get<0>(input_batch).resize(N);
        
        for(auto i=0ul; i<N; ++i)
            std::get<0>(input_batch)[i] = std::tuple_element_t<0, MyModel::InTuple>::Constant(1.f);
        
        const auto [output_data] = model.predict(input_batch);
        
        std::cout << "Batch size " << output_data.size() << " successful" << std::endl;
    }
    
    std::cout << "Done!" << std::endl;
}

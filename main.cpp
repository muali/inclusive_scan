#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.h>
#include "cl.hpp"

#include <cassert>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

using std::vector;

vector<float> read()
{
    std::ifstream in("input.txt");
    size_t n;
    in >> n;
    vector<float> res(n);
    for (size_t i = 0; i < n; ++i)
        in >> res[i];
    return res;
}

cl::Buffer gpu_inclusive_scan(
    cl::Context const &context,
    cl::Program const &program, 
    cl::Buffer const &input, 
    cl::CommandQueue const &queue,
    size_t size, 
    size_t block_size)
{
    size_t reduce_size = max(size / block_size, block_size);
    cl::Buffer reduce_buf(context, CL_MEM_READ_WRITE, reduce_size * sizeof(float));
    cl::Buffer result(context, CL_MEM_READ_WRITE, size * sizeof(float));

    size_t block_cnt = size / block_size + (size % block_size ? 1 : 0);
    size_t dev_size = block_cnt * block_size;

    cl::Kernel scan_stage(program, "scan_hillis_steele");
    cl::KernelFunctor scan_stage_f(scan_stage, queue, cl::NullRange, cl::NDRange(dev_size), cl::NDRange(block_size));
    scan_stage_f(size, input, result, cl::__local(block_size * sizeof(float)), 
        cl::__local(block_size * sizeof(float)), reduce_buf).wait();

    if (size > block_size)
    {
        cl::Buffer next_stage_res = gpu_inclusive_scan(context, program, reduce_buf, queue, reduce_size, block_size);

        cl::Kernel back_stage(program, "back_stage");
        cl::KernelFunctor back_stage_f(back_stage, queue, cl::NullRange, cl::NDRange(dev_size), cl::NDRange(block_size));
        back_stage_f(size, result, result, next_stage_res).wait();
    }

    return result;
}

vector<float> generate(size_t size, float /*step*/)
{
    vector<float> res(size);
    for (size_t i = 0; i < size; ++i)
        res[i] = 1.;
    return res;
}

void test(vector<float> const &data, vector<float> const &gpu)
{
    const double eps = 1e-2;
    float cumsum = 0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        cumsum += data[i];
        if (fabs(cumsum - gpu[i]) > eps)
        {
            std::cerr << i << " " << cumsum << " " << gpu[i] << std::endl;
            return;
        }
    }
}

int main()
{
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("inclusive_scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
            cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices, "-D BLOCK_SIZE=16");

        vector<float> data = read();
        //vector<float> data = generate(1048576, 1e-3);

        size_t block_size = 256;


        // allocate device buffer to hold message
        cl::Buffer dev_data(context, CL_MEM_READ_ONLY, data.size() * sizeof(float));

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_data, CL_TRUE, 0, data.size() * sizeof(float), &data[0]);

        cl::Buffer dev_result = gpu_inclusive_scan(context, program, dev_data, queue, data.size(), block_size);

        std::vector<float> result(data.size());
        queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, data.size() * sizeof(float), &result[0]);


        std::ofstream out("output.txt");
        out.setf(out.fixed);
        out.precision(3);
        for (size_t i = 0; i < result.size(); ++i)
        {
            out << result[i] << (i == result.size() - 1 ? '\n' : ' ');
        }
    }
    catch (cl::Error e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}

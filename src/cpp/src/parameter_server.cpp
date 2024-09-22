#include <grpcpp/grpcpp.h>
#include "parameter_server.grpc.pb.h"
#include <boost/lockfree/queue.hpp>
#include <vector>
#include <thread>
#include <mutex>

class ParameterServerImpl final : public ParameterServer::Service {
public:
    ParameterServerImpl() : model_weights(1000) {}

    grpc::Status UpdateModel(grpc::ServerContext* context, const ModelUpdate* request, UpdateResponse* response) override {
        std::lock_guard<std::mutex> lock(model_mutex);
        for (int i = 0; i < request->weights_size(); ++i) {
            model_weights[i] = request->weights(i);
        }
        response->set_success(true);
        return grpc::Status::OK;
    }

    grpc::Status GetModel(grpc::ServerContext* context, const GetModelRequest* request, Model* response) override {
        std::lock_guard<std::mutex> lock(model_mutex);
        for (float weight : model_weights) {
            response->add_weights(weight);
        }
        return grpc::Status::OK;
    }

private:
    std::vector<float> model_weights;
    std::mutex model_mutex;
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    ParameterServerImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}
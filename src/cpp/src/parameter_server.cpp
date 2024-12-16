#include <grpcpp/grpcpp.h>
#include "parameter_server.grpc.pb.h"
#include <boost/lockfree/queue.hpp>
#include <vector>
#include <thread>
#include <atomic>

class ParameterServerImpl final : public ParameterServer::Service {
public:
    ParameterServerImpl() : model_weights(1000), update_queue(1000) {
        // Start a background thread to process updates
        update_thread = std::thread(&ParameterServerImpl::ProcessUpdates, this);
    }

    ~ParameterServerImpl() {
        // Signal the update thread to stop and wait for it to finish
        stop_updates = true;
        if (update_thread.joinable()) {
            update_thread.join();
        }
    }

    grpc::Status UpdateModel(grpc::ServerContext* context, const ModelUpdate* request, UpdateResponse* response) override {
        std::vector<float> weights(request->weights().begin(), request->weights().end());
        if (update_queue.push(weights)) {
            response->set_success(true);
        } else {
            response->set_success(false);
        }
        return grpc::Status::OK;
    }

    grpc::Status GetModel(grpc::ServerContext* context, const GetModelRequest* request, Model* response) override {
        for (float weight : model_weights) {
            response->add_weights(weight);
        }
        return grpc::Status::OK;
    }

private:
    void ProcessUpdates() {
        while (!stop_updates) {
            std::vector<float> weights;
            while (update_queue.pop(weights)) {
                for (size_t i = 0; i < weights.size(); ++i) {
                    model_weights[i] = weights[i];
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Adjust sleep duration as needed
        }
    }

    std::vector<float> model_weights;
    boost::lockfree::queue<std::vector<float>> update_queue;
    std::thread update_thread;
    std::atomic<bool> stop_updates{false};
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
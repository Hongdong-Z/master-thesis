#pragma once
#include <fstream>
#include <filesystem>

#include "nlohmann/json.hpp"

namespace localization_lidar {

using json = nlohmann::json;

class ParameterServer {
public:
    static void initialize(std::filesystem::path filename) {
        ParameterServer::instance_ = new json(json::parse(std::ifstream(filename)));
    }

    static json& get() {
        if(ParameterServer::instance_ == nullptr) {
            throw std::runtime_error("Load a configuration file first with ParameterServer::initialize() before calling get().");
        }

        return *ParameterServer::instance_;
    }

private:
    inline static json* instance_;
};




}
#pragma once

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"
#include <string>
#include <stdexcept>
#include <cstdio>

namespace kodes 
{

class Config
{
private:
    rapidjson::Document document;
    FILE* file;

    const rapidjson::Value* getValue(const std::string& name) const;

public:
    explicit Config(const std::string& json_path);
    
    ~Config();
    
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    
    Config(Config&& other) noexcept;
    Config& operator=(Config&& other) noexcept;

    double getDouble(const std::string& name, double default_value);
    int getInt(const std::string& name, int default_value);
    std::string getString(const std::string& name, const std::string& default_value);
    bool getBool(const std::string& name, bool default_value);
    
    bool hasKey(const std::string& name) const;
};

} // namespace kodes
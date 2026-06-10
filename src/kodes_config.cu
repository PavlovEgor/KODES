#include "kodes_config.cuh"
#include <iostream> 

namespace kodes 
{

Config::Config(const std::string& json_path)
    : file(nullptr)
{
    file = fopen(json_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Cannot open config file: " + json_path);
    }
    
    char readBuffer[65536];
    rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));
    
    document.ParseStream(is);
    
    if (document.HasParseError()) {
        std::string error_msg = rapidjson::GetParseError_En(document.GetParseError());
        throw std::runtime_error(
            "JSON parse error in " + json_path + " at offset " + 
            std::to_string(document.GetErrorOffset()) + ": " + error_msg
        );
    }
    
    if (!document.IsObject()) {
        throw std::runtime_error("Config file root must be a JSON object");
    }
}

Config::~Config() {
    if (file) {
        fclose(file);
    }
}

Config::Config(Config&& other) noexcept
    : document(std::move(other.document))
    , file(other.file)
{
    other.file = nullptr;
}

Config& Config::operator=(Config&& other) noexcept {
    if (this != &other) {
        document = std::move(other.document);
        if (file) fclose(file);
        file = other.file;
        other.file = nullptr;
    }
    return *this;
}

const rapidjson::Value* Config::getValue(const std::string& name) const {
    auto it = document.FindMember(name.c_str());
    if (it == document.MemberEnd()) {
        return nullptr;
    }
    return &(it->value);
}

double Config::getDouble(const std::string& name, double default_value) const {
    const rapidjson::Value* val = getValue(name);
    if (val && val->IsDouble()) {
        return val->GetDouble();
    }
    return default_value;
}

int Config::getInt(const std::string& name, int default_value) const {
    const rapidjson::Value* val = getValue(name);
    if (val && val->IsInt()) {
        return val->GetInt();
    }
    return default_value;
}

std::string Config::getString(const std::string& name, const std::string& default_value) const {
    const rapidjson::Value* val = getValue(name);
    if (val && val->IsString()) {
        return val->GetString();
    }
    return default_value;
}

bool Config::getBool(const std::string& name, bool default_value) const {
    const rapidjson::Value* val = getValue(name);
    if (val && val->IsBool()) {
        return val->GetBool();
    }
    return default_value;
}

bool Config::hasKey(const std::string& name) const {
    return getValue(name) != nullptr;
}

} // namespace kodes
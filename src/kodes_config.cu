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

// "controls.absTol" walks into the nested object rather than looking for a
// member of that name, so a settings file can be grouped instead of flat.
const rapidjson::Value* Config::getValue(const std::string& name) const {
    const rapidjson::Value* value = &document;

    for (size_t start = 0; ; ) {
        const size_t dot = name.find('.', start);
        const std::string part =
            name.substr(start, dot == std::string::npos ? std::string::npos : dot - start);

        if (!value->IsObject()) {
            return nullptr;
        }

        auto it = value->FindMember(part.c_str());
        if (it == value->MemberEnd()) {
            return nullptr;
        }

        value = &(it->value);

        if (dot == std::string::npos) {
            return value;
        }

        start = dot + 1;
    }
}

// IsNumber rather than IsDouble: a tolerance written 1 or 10 parses as an
// integer, and silently handing back the default for it is the kind of thing
// that is only noticed in the results.
double Config::getDouble(const std::string& name, double default_value) const {
    const rapidjson::Value* val = getValue(name);
    if (val && val->IsNumber()) {
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
#ifndef IO_UTILS_HPP
#define IO_UTILS_HPP

#include <nlohmann/json.hpp>
#include <string>

namespace IOUtils {

    nlohmann::json readConfigFromJSON(const std::string& jsonFilePath);
    void writeJSONToCGNS(const std::string& cgnsFilePath, const std::string& jsonFilePath);

} // namespace IOUtils

#endif // IO_UTILS_HPP

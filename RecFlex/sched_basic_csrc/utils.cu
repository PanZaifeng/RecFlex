#include "utils.cuh"

#include <fstream>

std::string get_env(const std::string &key, const std::string &default_val) {
  char *env = getenv(key.c_str());
  return env ? env : default_val;
}

bool exist_file(const std::string &fname) {
  std::ifstream ist(fname);
  return ist.good();
}
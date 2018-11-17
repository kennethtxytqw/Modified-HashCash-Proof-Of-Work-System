#pragma once
#include <string>

#define HEX_SIZE 4 // bits
#define MAX_HEX_VALUE 16 
#define BYTE_SIZE 8

#define X_SIZE 416 // bits
#define ID_SIZE 64 // bits
#define EPOCH_SIZE 32 // bits
#define NONCE_SIZE 64 // bits
#define DIGEST_SIZE_IN_BITS 256
#define DIGEST_SIZE_IN_BYTES 32
#define TARGET_SIZE 64 // bits

typedef unsigned long long int ullong;

extern int hex_char_to_int(char c);
extern void hex_str_to_uint8_little_endian(uint8_t arr[], const std::string& hex);
extern void ascii_str_to_uint8_little_endian(uint8_t buf[], const std::string &str);
extern std::string uint8_to_hexstring(uint8_t arr[], const int len);
extern void ulong_to_uint8_little_endian(uint8_t buf[], const ulong a);
extern __host__ __device__ void ullong_to_uint8_big_endian(uint8_t buf[], const ullong a);
extern __host__ __device__ ulong uint8_little_endian_to_ulong(const uint8_t buf[]);
extern __host__ __device__ ulong uint8_big_endian_to_ulong(const uint8_t buf[]);
extern __host__ __device__ ullong uint8_little_endian_to_ullong(const uint8_t buf[]);
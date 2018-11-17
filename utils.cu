#include "utils.cuh"
#include <iostream>
#include <sstream>

using namespace std;

// Adapted from https://stackoverflow.com/a/18311086
int hex_char_to_int(char c)
{
    // TODO handle default / error
    switch(toupper(c))
    {
        case '0': return 0;
        case '1': return 1;
        case '2': return 2;
        case '3': return 3;
        case '4': return 4;
        case '5': return 5;
        case '6': return 6;
        case '7': return 7;
        case '8': return 8;
        case '9': return 9;
        case 'A': return 10;
        case 'B': return 11;
        case 'C': return 12;
        case 'D': return 13;
        case 'E': return 14;
        case 'F': return 15;
    }
    cerr << "Unknown Hex: " << c << endl;
    exit(1);
}


// Adapted from https://stackoverflow.com/a/18311086
void hex_str_to_uint8_little_endian(uint8_t arr[], const std::string& hex)
{
    // Stored in little endian
    int len = hex.length();
    for(int i = 0; i < hex.length()/2; i++)
    {
        uint8_t sum = hex_char_to_int(hex[len-2*i-1]);
        
        if (len-2*i-2 >= 0)
        {
            sum += hex_char_to_int(hex[len-2*i-2]) * (MAX_HEX_VALUE);
        }

        arr[i] = sum;
    }
}

void ascii_str_to_uint8_little_endian(uint8_t buf[], const std::string &str)
{
    // Store it in little endian
    for(int i = 0; i < str.length(); i++)
    {
        buf[i] = int(str[str.length() - i - 1]);
    }
}

std::string uint8_to_hexstring(uint8_t arr[], const int len)
{
    std::string hexstring;
    for(unsigned i = 0; i < len; i++)
    {
        std::stringstream sstream;
        sstream << hex << unsigned(arr[i]);

        
        if (sstream.str().length() < 2) 
        {
            hexstring = "0" + sstream.str() + hexstring;
        } else
        {
            hexstring = sstream.str() + hexstring;
        }
    }
    return hexstring;
}

void ulong_to_uint8_little_endian(uint8_t buf[], const ulong a)
{
    ulong x = a;
    for(unsigned i = 0; i < sizeof(ulong); i++)
    {
        buf[i] = x & 0xFF;
        x >>= BYTE_SIZE;
    }
}

__host__ __device__ void ullong_to_uint8_big_endian(uint8_t buf[], const ullong a)
{
    ullong x = a;
    for(unsigned i = 0; i < sizeof(ullong); i++)
    {
        buf[sizeof(ullong) - i - 1] = x & 0xFF;
        x >>= BYTE_SIZE;
    }
}

__host__ __device__ ulong uint8_little_endian_to_ulong(const uint8_t buf[])
{
    ulong x;
    for(unsigned i = 0; i < sizeof(ulong); i++)
    {        
        x <<= BYTE_SIZE;
        x += buf[sizeof(ulong) - i -1];
    }
    return x;
}

__host__ __device__ ulong uint8_big_endian_to_ulong(const uint8_t buf[])
{
    ulong x;
    for(unsigned i = 0; i < sizeof(ulong); i++)
    {        
        x <<= BYTE_SIZE;
        x += buf[i];
    }
    return x;
}

__host__ __device__ ullong uint8_little_endian_to_ullong(const uint8_t buf[])
{
    ullong x;
    for(unsigned i = 0; i < sizeof(ullong); i++)
    {
        x <<= BYTE_SIZE;
        x += buf[sizeof(ullong) - i -1];
    }
    return x;
}
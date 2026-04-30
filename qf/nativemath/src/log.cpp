#include "log.h"

using namespace std;

std::ostream & operator << (std::ostream & out, const std::vector<float> & values) {
    size_t size = values.size();

    out << '[';
    for (size_t i = 0; i < size; i++) {
        out << values[i];
        if (i < size - 1) {
            out << ", ";
        }
    }
    out << ']';
    return out;

}
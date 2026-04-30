#ifndef __LOG_H__
#define __LOG_H__

#include <iostream>
#include <vector>

std::ostream & operator << (std::ostream & out, const std::vector<float> & values);
#endif
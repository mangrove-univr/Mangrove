#include "Vcd2mangrove.hh"

#include <string>
#include <iostream>

// print a short guide for vcd2mangrove
void printUsage();

int main(int argc, char * argv[])
{
    if (argc != 3 && argc != 4)
    {
        printUsage();
        return 1;
    }

    std::string fileName(argv[1]);
    if (fileName.empty() || (fileName.find(".vcd") == std::string::npos))
    {
        printUsage();
        return 1;
    }

    unsigned int factor = (unsigned int) std::stoi(argv[2]);
    int mode = (argc == 4)? std::stoi(argv[3]) : 0;

    Vcd2Mangrove_t converter(fileName.c_str(), factor, mode);
    converter.run();

    return 0;
}

void printUsage()
{
    std::cout << "\n usage ./vcd2mangrove <FILE> <FACT> [<MODE>] \n"
      << "\n where:\n"
      << "\t <FILE>   the input trace in the VCD format that must be converted\n"
      << "\t <FACT>   the output trace will have a length multiple of this value\n"
      << "\t <MODE>   the output trace will be generated in one of the following ways:\n"
      << "\t   [0]    the output trace will contain the values with the original type [default]\n"
      << "\t   [1]    the output trace will contain the bitvectors splitted in each component\n"
      << "\t   [2]    the output trace will contain the bitvectors converted in numeric value\n"
      << "\n example: ./vcd2mangrove TRACE.vcd 1\n"
      << "\n output:\n"
      << "\t TRACE.vcd.mangrove   the output trace with the same length of the original trace\n"
      << "\t TRACE.vcd.variables  the list of variables with the same order of corresponding values\n\n";
}

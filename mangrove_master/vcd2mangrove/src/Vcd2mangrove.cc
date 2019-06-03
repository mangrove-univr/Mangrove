#include "../include/Vcd2mangrove.hh"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

// Input modes
#define MODE_ORIGINAL 0
#define MODE_SPLITTED 1
#define MODE_NUMERIC 2

using namespace std;

namespace {

// Utilities

// Note: return true if the string contains only spaces

const char * ws = " \t\n\r\f\v";

inline bool isEmpty(const std::string str)
{
  return str.find_first_not_of(ws) == string::npos;
}

// Note: trim from end of string (right)

inline std::string& rtrim(std::string& s, const char* t = ws)
{
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

// Note: trim from start of string (left)

inline std::string& ltrim(std::string& s, const char* t = ws)
{
  s.erase(0, s.find_first_not_of(t));
  return s;
}

// Note: trim from both ends of string (left & right)

inline std::string& trim(std::string& s, const char* t = ws)
{
  return ltrim(rtrim(s, t), t);
}

// Note: open an input stream for reading the specified file

void openInputStream(const string &fileName, ifstream& stream)
{
  stream.open(fileName.data());
  if (!(stream.is_open() && stream.good())) 
  {
    cerr << "FATAL ERROR: <FILE> " << fileName << " cannot be opened" << endl;
    exit(1); 
  }
}

// Note: open an output stream for writing the specified file

void openOutputStream(const string &fileName, ofstream& stream)
{
  stream.open(fileName.data());
  if (!(stream.is_open() && stream.good()))
  {
    cerr << "FATAL ERROR: <FILE> " << fileName << " cannot be opened" << endl;
    exit(1); 
  }
}

// Note: adjust the length of the value

void adjustLengthOfValue(std::string &value, unsigned int size)
{
  while (value.length() < size)
  {
    value.insert(0, "0");
  }
}

// Note: adjust the length of the list of values

void adjustLengthOfValues(std::list<std::string> &values, unsigned int size)
{
  for (list<string>::iterator valueIterator =  values.begin();
    valueIterator != values.end(); ++valueIterator)
  {
    adjustLengthOfValue(*valueIterator, size);
  }
}

// Note: remove the undefined value from the value

void removeUndefinedValue(std::string &value)
{
  static bool warning = true;
  for (unsigned int j = 0; j < value.length(); ++j)
  {
    if (value[j] == 'x' || (value[j] == 'X'))
    {
      if (warning)
      {
        std::cout << "WARNING: x/X has been replaced with 0 at least one time" << endl;
        warning = false;
      }
    
      value.replace(j, 1, "0");
    }
  }
}

// Note: remove the undefined value from the list of values

void removeUndefinedValues(std::list<std::string> &values)
{
  for (list<string>::iterator valueIterator =  values.begin();
    valueIterator != values.end(); ++valueIterator)
  {
    removeUndefinedValue(*valueIterator);
  }
}

// Note: convert the number in decimal from binary
// Note: up to 64 bit integers

long long int bin2dec(const char * bin)
{
    long long int sum = 0;
    long long int b, m, n;

    int len = static_cast<int>( strlen(bin) - 1 );
    for(int k = 0; k <= len; k++)
  {
    n = (bin[k] - '0');
    if ((n > 1) || (n < 0))
    {
      cerr << "Error: binary has only 1 and 0" << endl;
      return 0;
    }
    for (b = 1, m = len; m > k; m--) b *= 2;
    sum = sum + n * b;
  }
  return sum;
}

// Note: round up the number w.r.t. the multiple
unsigned int roundUp(unsigned int numToRound, unsigned int multiple)
{
  if (multiple == 0)
      return numToRound;

  unsigned int remainder = numToRound % multiple;
  if (remainder == 0)
      return numToRound;

  return numToRound + multiple - remainder;
}

} // namespace

// Constructor

Vcd2Mangrove_t::Vcd2Mangrove_t(const char *fileName, unsigned int factor, int mode)
  : _fileName(fileName),
    _factor(factor),
    _mode(mode),
    _associations(),
    _ranges(),
    _values()
{
    if (factor == 0)
    {
        cerr << "FATAL ERROR: <FACT> must be greater than 0" << endl;
        exit(1);
    }
}

// Destructor

Vcd2Mangrove_t::~Vcd2Mangrove_t()
{
  // Nothing to do
}

// Public methods

void Vcd2Mangrove_t::run()
{
    // 1. Open the input stream
    ifstream sourceStream;
    openInputStream(_fileName, sourceStream);

    // 2. Parse variables
    _parseDeclarations(sourceStream);

    // 3. Parse values
    _parseDumping(sourceStream);

    // 4. Close stream
    sourceStream.close();

    // 5. Dump files
    _printFiles();
}

// Private methods

void Vcd2Mangrove_t::_parseDeclarations(std::istream& stream)
{
    // Note: create an entry (nickname, realname) for each variable,
    // where nickname is the compact ASCII identifier used in the
    // VCD format and realname is the real name of the variable.

    string line;
    while (getline(stream, line))
    {
        stringstream sstream(trim(line)); string token;
        string nickName; string realName; int size;

        // Note: if true we reach the end of this section
        if (line.find("$enddefinitions") != string::npos
          || line.find("$dumpvars") != string::npos ) return;

        // Note: skip all the uninteresting lines (empty or not useful)
        if (isEmpty(line) || line.find("$var") == string::npos) continue;

        // Note: parse the declaration of the variable
        sstream >> token     // variable declaration
                >> token     // wire specification
                >> size      // size specification
                >> nickName  // variable nickname
                >> token;    // variable realname
       
        // Note: complete the realname of the variable
        for (; string(token).compare("$end") != 0; sstream >> token)
          realName += (realName.empty())? token : ("_" + token);  
              
        if (size == 1)
        {
            // Note: variable with a single bit
            _associations.insert(make_pair(nickName, realName));
            _ranges.insert(make_pair(nickName, make_pair(1, 1)));
        }
        else
        {
            // Note: variable with multiple bits
            size_t pos1 = realName.find('[');
            size_t pos2 = realName.find(':');
            size_t pos3 = realName.find(']');
            int firstBit = atoi(realName.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
            int lastBit = atoi(realName.substr(pos2 + 1, pos3 - pos2 - 1).c_str());
            realName = realName.substr(0, pos1 - 1);
            _associations.insert(make_pair(nickName, realName));
            _ranges.insert(make_pair(nickName, make_pair(firstBit, lastBit)));
        }
    }
}

void Vcd2Mangrove_t::_addTimeInstant()
{
    if (_values.empty())
    {
        // A. Create an empty time istant for each variable
        for (Associations::iterator varIterator = _associations.begin();
          varIterator != _associations.end(); ++varIterator)
          _values.insert(make_pair(varIterator->first, list<string>(1)));

    }
    else
    {
        // B. Create a duplicate of the last time istant
        for (Values::iterator valueIterator = _values.begin();
          valueIterator != _values.end(); ++valueIterator)
          valueIterator->second.push_back(valueIterator->second.back());
    }
}

void Vcd2Mangrove_t::_parseDumping(std::istream& stream)
{
    std::string line;
    bool inComment = false;
    std::pair<std::string, std::string> assign;

    while (getline(stream, line))
    {
        // Note: skip all the possible comments
        if (line.find("$comment") != string::npos)
        {
            inComment = true;
            continue;
        }

        if (inComment)
        {
            inComment = line.find("$end") == string::npos;
            continue;
        }

        // Note: skip uninteresting lines
        if (isEmpty(line) || inComment 
          || line.find("$end") != string::npos
          || line.find("$dumpvars") != string::npos || 
          line.find("$enddefinitions") != string::npos) continue;

        // Note: new simulation instant
        if (line.at(0) == '#')
        {
            _addTimeInstant();
            continue;
        }

        // Note: variable value
        if (_values.empty())
        {
            _addTimeInstant();
        }

        _parseAssignment(trim(line), assign);        
        _values[assign.first].pop_back();
        _values[assign.first].push_back(assign.second);
    }
}

void Vcd2Mangrove_t::_parseAssignment(const std::string &entry, 
                                      std::pair<std::string, std::string>& assign)
{
    // A. Logic value
    if (entry.at(0) == 'b')
    {
        size_t space = entry.find(" ");
        assign.first = entry.substr(space + 1);
        assign.second = entry.substr(1, space - 1);
        return;
    }

    // B. Real value
    if (entry.at(0) == 'r')
    {
        size_t space = entry.find(" ");
        assign.first = entry.substr(space + 1);
        assign.second = entry.substr(1, space - 1);
        return;
    }

    // C. Remaining type
    assign.first = entry.substr(1);
    assign.second = entry.substr(0, 1);
}

void Vcd2Mangrove_t::_printFiles()
{
    // 1. Open output stream for variables and values
    std::ofstream valueStream; std::ofstream varStream;
    openOutputStream(_fileName + ".mangrove", valueStream);
    openOutputStream(_fileName + ".variables", varStream);

    // 2. Calculate the number of variables
    unsigned int variables = 0;

    if (_mode == MODE_SPLITTED)
    {
        // Note: bitvectors must be splitted
        for (Ranges::iterator rangeIterator = _ranges.begin();
                              rangeIterator != _ranges.end(); ++rangeIterator)

            variables += (unsigned int) abs(rangeIterator->second.first -
                             rangeIterator->second.second) + 1;
    }
    else
    {
        // Note: bitvectors are not splitted
        variables = (unsigned int) _values.size();
    }

    // 3. Calculate the length of execution trace
    unsigned int traceLength = roundUp(static_cast<unsigned int>(
      ((_values.begin())->second).size()), _factor);

    // 4. Dump the first line of the execution trace
    valueStream << variables << " " << traceLength << endl;

    // 5. Print the values of the variables
    for (Associations::iterator varIterator = _associations.begin();
         varIterator != _associations.end(); ++varIterator)
    {
        // 5.1. Get the values of the variable
        list<string> varValues = _values[varIterator->first];
        unsigned int varLength = static_cast<unsigned int>(varValues.size());

        // 5.2. Remove the undefined values
        removeUndefinedValues(varValues);

        // 5.3. Get the range of the variable
        pair<int, int> range = _ranges[varIterator->first];

        // 5.4. Calculate the number of bits
        unsigned int size = (unsigned int) 
          abs(range.first - range.second) + 1U;

        // 5.4. Print the values of the variable
        if (range.first == 1 && range.second == 1)
        {
            // Note: variable with a single bit

            // 5.4.1. Print the real name
            varStream << varIterator->second << endl;

            // 5.4.2. Print the values of the variable
            for (list<string>::iterator valueIterator =  varValues.begin();
                 valueIterator != varValues.end(); valueIterator++)
            {
                assert(valueIterator->length() == 1);
                valueStream << *valueIterator << " ";
            }

            // 5.4.3. Adjust the trace length
            for (unsigned int i = varLength; i < traceLength; ++i)
                valueStream << varValues.back() << " ";
            valueStream << endl;

        }
        else
        {
            // Note: variable with multiple bits

            switch (_mode)
            {
            case MODE_ORIGINAL:
            {

                // 5.4.1. Print the real name
                varStream << varIterator->second << endl;

                // 5.4.2. Adjust the length of values
                adjustLengthOfValues(varValues, size);

                // 5.4.3. Print the values of the variable
                for (list<string>::iterator valueIterator =  varValues.begin();
                     valueIterator != varValues.end(); valueIterator++)
                {
                    assert(static_cast<int>(valueIterator->length()) == size);
                    valueStream << *valueIterator << " ";
                }

                // 5.4.4. Adjust the trace length
                for (unsigned int i = varLength; i < traceLength; ++i)
                    valueStream << varValues.back() << " ";
                valueStream << endl;

                break;
            }

            case MODE_SPLITTED:
            {

                // 5.4.1. Adjust the length of values
                adjustLengthOfValues(varValues, size);

                int j = std::max(range.first, range.second);
                int j_min = std::min(range.first, range.second);
                for (unsigned int pos = 0; j >= j_min; --j, ++pos)
                {

                    // 5.4.2. Print the real name of the variable
                    varStream << varIterator->second + "[" << j << "]" << endl;

                    // 5.4.3. Print the values of the variable
                    for (list<string>::iterator valueIterator =  varValues.begin();
                         valueIterator != varValues.end(); valueIterator++)
                        valueStream << (*valueIterator)[pos] << " ";

                    // 5.4.4. Adjust the trace length
                    for (unsigned int i = varLength; i < traceLength; ++i)
                        valueStream << varValues.back()[pos] << " ";
                    valueStream << endl;

                }

                break;
            }

            case MODE_NUMERIC:
            {

                // 5.4.1. Print the real name
                varStream << varIterator->second << endl;

                // 5.4.2. Adjust the length of values
                adjustLengthOfValues(varValues, size);

                // 5.4.3. Print the values of the variable
                for (list<string>::iterator valueIterator =  varValues.begin();
                     valueIterator != varValues.end(); valueIterator++)
                {
                    assert(static_cast<int>(valueIterator->length()) == size);
                    valueStream << bin2dec(valueIterator->c_str()) << " ";
                }

                // 5.4.4. Adjust the trace length
                for (unsigned int i = varLength; i < traceLength; ++i)
                    valueStream << bin2dec(varValues.back().c_str()) << " ";
                valueStream << endl;

                break;
            }

            default:
            {
                cerr << "FATAL ERROR: <MODE> must be only 0, 1 or 2" << endl;
                exit(1);
            }

            }
        }
    }

  // 6. Close the streams
  valueStream.close();
  varStream.close();
}

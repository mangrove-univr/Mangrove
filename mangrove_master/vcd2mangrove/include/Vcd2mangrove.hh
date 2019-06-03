#ifndef Vcd2Mangrove__HH
#define Vcd2Mangrove__HH

#include <string>
#include <list>
#include <map>

class Vcd2Mangrove_t {

  public:

    // Constructor
    Vcd2Mangrove_t(const char *fileName, unsigned int factor, int mode);

    // Destructor
    virtual ~Vcd2Mangrove_t();

    // Translate
    void run();

  private:

    // INPUTS

    // <FILE>
    std::string _fileName;

    // <FACT>
    unsigned int _factor;

    // <MODE>
    int _mode;

    // PRIVATE FIELDS
    
    // Typedefs
    typedef std::map<std::string, std::string> Associations; 
    typedef std::map<std::string, std::pair<int, int> > Ranges; 
    typedef std::map<std::string, std::list<std::string> > Values;

    // Assocations <nickname, name>
    Associations _associations;

    // Ranges <nickname, <min, max>>
    Ranges _ranges;

    // Values <nickname, values>
    Values _values;

    // PRIVATE METHODS

    // STEP1: build the map containing mappings
    void _parseDeclarations(std::istream& stream);

    // STEP2: parse the variable dumping
    void _addTimeInstant(void);
	  void _parseDumping(std::istream& stream);
	  void _parseAssignment(const std::string &entry, 
      std::pair<std::string, std::string>& assign);

    // STEP3: dump the execution trace
	  void _printFiles();
};

#endif

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

/*
* A class to read data from a csv file.
*/
class CSVReader
{
public:
  std::vector<std::vector<double>> points;
  
  CSVReader(std::string inputFileName, double scale, std::string delm = ","){    
    std::ifstream inputFile(inputFileName);
    int l = 0;
    
    while (inputFile) {
        l++;
	std::string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
	  std::istringstream ss(s);
	    std::vector<double> record;
 
            while (ss) {
	      std::string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
		  std::cout << "NaN found in file " << inputFileName << " line " << l << std::endl;
                    e.what();
                }
            }
 
            points.push_back(record);
        }
    }
    
    for (int i = 0; i < points.size(); i++){
      for (int j = 0; j < points[0].size(); j++){
	points[i][j] = points[i][j]*scale;
      }
    }
  }
  
};

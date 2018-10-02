#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<stdlib.h>

using namespace std;

string& ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
}


int main(int argc, char* argv[])
{
  string datafile="";
  if (argc>1)
    datafile=argv[1];
  unsigned int MINSZ=1000000000;
  if (argc>2)
    MINSZ=atoi(argv[2]);

  ifstream In(datafile.c_str());
  vector <string> STR;
  string line;
  while(getline(In,line))
    STR.push_back(ReplaceStringInPlace(line," ",","));

  for(unsigned int i=0;i<STR.size();++i)
    if(STR[i].size()<MINSZ)
      MINSZ=STR[i].size();


  cout<<1;
  for(unsigned int i=1;i<=MINSZ/2;++i)
    cout <<"," <<  i+1;
  cout << endl;
  for(unsigned int i=0;i<STR.size();++i)
    cout << STR[i].substr(0,MINSZ) << endl;

  
  return 0;
}



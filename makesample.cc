/*
|__) /\ |\ ||  \/  \|\/|   /__`|   |__ /  `|/  \|__) 
|  \/~~\| \||__/\__/|  |   .__/|___|___\__,|\__/|  \ 
         __     __            _____ ___ __           
 /\ |\ ||  \   /  \|  | /\ |\ ||| /|__ |__)          
/~~\| \||__/   \__X\__//~~\| \|||/_|___|  \  
*/

#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<map>
#include<random>
#include<set>
#include<stdlib.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
#include <exception>
#include <boost/program_options/option.hpp>
#include <boost/lexical_cast/try_lexical_convert.hpp>
#include <boost/program_options/value_semantic.hpp>

#define DEBUG_ 0

using namespace std;
using namespace boost::program_options;

//--------------------------

const string VERSION="Random selector & Quantizer\ncopyright Ishanu Chattopadhyay 2017\nversion 0.3141596";
const string EMPTY_ARG_MESSAGE="Exiting. Type -h or --help for usage";


//--------------------------

vector<option> ignore_numbers(vector<string>& args)
{
  vector<option> result;
  int pos = 0;
  while(!args.empty()) 
    {
      const auto& arg = args[0];
      double num;
      if(boost::conversion::try_lexical_convert(arg, num)) 
	{
	  result.push_back(option());
	  option& opt = result.back();

	  opt.position_key = pos++;
	  opt.value.push_back(arg);
	  opt.original_tokens.push_back(arg);

	  args.erase(args.begin());
	} 
      else 
	break;
    }

  return result;
}

//--------------------------




//--------------------------


string quantize(string& line,
		vector<double>& partition,
		set<char>& alphabet,
		unsigned int SKIP=1)
{
  char s='A';
  string str="",str0="";
  
  stringstream ss(line);
  string token;
  double data_pt;
  unsigned int j=0;
  while(j++<SKIP)
    ss>>token;

  while(ss>>data_pt)
    {
      for (j=0;j<partition.size();j++)
	if (data_pt<partition[j])
	  break;
      str+=","+(str0+char(s+j));
      alphabet.insert(char(s+j));
    }

  return token+str;
}

//---------------------------
void ReplaceStringInPlace(string& subject,
			  const string& search,
                          const string& replace)
{
  size_t pos = 0;
  while((pos = subject.find(search,pos))
	!=string::npos)
    {
      subject.replace(pos,search.length(),
		      replace);
      pos+=replace.length();
    }
  return;
}
//----------------------------------


int main(int argc,char* argv[])
{
  string masterfile="master.txt",
    ofile="out.txt",configfile="cfg.cfg";
  unsigned int PROB_=1;
  vector<double> partition={0.0};
  set<char> alphabet;

  options_description desc( "Random selector & Quantizer\nIshanu Chattopadhyay 2017\nishanu@uchicago.edu\nUsage: ");
  desc.add_options()
    ("help,h", "print help message.")
    ("version,V", "print version number")
    ("masterfile,f",value<string>(), 
     "masterfile [data.txt]")
    ("ofile,o",value<string>(), 
     "output file [out.txt]")
    ("pselect,M",value< unsigned int >(), 
     "select on in M [100]")
    ("partition,p",value< vector<double> >()->multitoken(), "partition");
  positional_options_description p;
  variables_map vm;
  if (argc == 1)
    {
      cout << EMPTY_ARG_MESSAGE << endl;
      return 1;
    }
  try
    {
     store(command_line_parser(argc, argv)
        .extra_style_parser(&ignore_numbers)
        .options(desc)
        .run(), vm);
      notify(vm);
    } 
  catch (std::exception &e)
    {
      cout << endl << e.what() 
	   << endl << desc << endl;
      return 1;
    }
  if (vm.count("help"))
    {
      cout << desc << endl;
      return 1;
    }
  if (vm.count("version"))
    {
      cout << VERSION << endl; 
      return 1;
    }

  if (vm.count("partition"))
    partition=vm["partition"].as<vector<double> >();
  if (vm.count("masterfile"))
    masterfile=vm["masterfile"].as<string>();
  if (vm.count("ofile"))
    ofile=vm["ofile"].as<string>();
  if (vm.count("pselect"))
    PROB_=vm["pselect"].as<unsigned int>();


  if(DEBUG_)
    {
      cout << "PARTITION: " << endl;
      for(unsigned int i=0;i<partition.size();++i)
	cout << partition[i] << " " << endl;
    }


  std::random_device rd;  //seed for the random 
  std::mt19937 gen(rd()); //mersenne_twister_engine 
  std::uniform_int_distribution<> dist(0,PROB_);

  ifstream IN(masterfile.c_str());
  string line;
  ofstream out(ofile.c_str());

  getline(IN,line);
  getline(IN,line);
  ReplaceStringInPlace(line,"\t",",");
  out << line << endl;
  getline(IN,line);
  getline(IN,line);

  while(getline(IN,line))
    if (dist(gen)<1)
      out << quantize(line,partition,alphabet) << endl;
  
  out.close();
  IN.close();

  cout << "ALPHABET: ";
  for(set<char>::iterator itr=alphabet.begin();
      itr!=alphabet.end();
      ++itr)
    cout << *itr << " " ;
  cout << endl;

  return 0;
}

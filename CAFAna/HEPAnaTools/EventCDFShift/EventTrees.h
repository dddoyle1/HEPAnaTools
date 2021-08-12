#include "TTree.h"
#include "TFile.h"
#include "CAFAna/Core/EventList.h"

#include <stdio.h> /* std::remove */

namespace hepana {
  namespace  {
    ana::Var kRun = SIMPLEVAR(hdr.run);
    ana::Var kSubrun = SIMPLEVAR(hdr.subrun);
    ana::Var kCycle = SIMPLEVAR(hdr.cycle);
    ana::Var kBatch = SIMPLEVAR(hdr.batch);
    ana::Var kEvent = SIMPLEVAR(hdr.evt);
    ana::Var kSlice = SIMPLEVAR(hdr.subevt);
  }

  void MakeEventTrees(std::string output_file_name,
		      std::map<std::string, std::string> defnames,
		      std::map<std::string, ana::Cut> _cuts,
		      std::map<std::string, ana::Var> _vars)
  {
    // convert maps to vectors of pairs for MakeEventTTreeFile
    std::vector<std::pair<std::string, ana::Cut> > cuts;
    for(auto cut_it = _cuts.begin(); cut_it != _cuts.end(); cut_it++) {
      cuts.push_back(std::pair<std::string, ana::Cut>(cut_it->first, cut_it->second));
    }

    std::vector<std::pair<std::string, ana::Var> > vars;
    // put event indices first
    vars.push_back(std::pair<std::string, ana::Var>("run", kRun));
    vars.push_back(std::pair<std::string, ana::Var>("subrun", kSubrun));
    vars.push_back(std::pair<std::string, ana::Var>("cycle", kCycle));
    vars.push_back(std::pair<std::string, ana::Var>("batch", kBatch));
    vars.push_back(std::pair<std::string, ana::Var>("event", kEvent));
    vars.push_back(std::pair<std::string, ana::Var>("slice", kSlice));
    for(auto var_it = _vars.begin(); var_it != _vars.end(); var_it++) {
      vars.push_back(std::pair<std::string, ana::Var>(var_it->first, var_it->second));
    }
    
    // Fill trees from each defname
    for(auto def_it = defnames.begin(); def_it != defnames.end(); def_it++) {
      ana::MakeEventTTreeFile(def_it->second,
			      "tmp_" + def_it->first + ".root",
			      cuts,
			      vars);
    }

    // temp files have been created and trees filled
    // read trees from files and put into one
    TFile * output = new TFile(output_file_name.c_str(), "recreate");
    std::cout << "Saving results to " << output->GetName() << std::endl;
    for(auto def_it = defnames.begin(); def_it != defnames.end(); def_it++) {
      std::string input_file_name = "tmp_" + def_it->first + ".root";
      TFile * input = TFile::Open(input_file_name.c_str());

      auto dir = output->mkdir(def_it->first.c_str());
      dir->cd();

      for(auto cut : cuts) {
	auto tree = ((TTree*) input->Get(cut.first.c_str()))->CloneTree();
	tree->Write(cut.first.c_str());
      }
      input->Close();
      delete input;
      
      if(std::remove(input_file_name.c_str())) {
	std::cerr << "WARNING: Error removing " << input_file_name << std::endl;
      }
      else {
	std::cout << "Removed " << input_file_name << std::endl;
      }
    }
    output->Close();
  }
		      

};


